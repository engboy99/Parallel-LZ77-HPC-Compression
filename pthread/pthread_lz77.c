#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <pthread.h>
#include <unistd.h>

#define WINDOW_SIZE 32768
#define LOOKAHEAD_SIZE 258
#define MIN_MATCH_LENGTH 3
#define BLOCK_SIZE (1024 * 1024)
#define OVERLAP_SIZE WINDOW_SIZE
#define MAX_THREADS 32

typedef struct {
    uint16_t offset;
    uint16_t length;
    uint8_t literal;
} LZ77_Token;

typedef struct {
    LZ77_Token *tokens;
    size_t num_tokens;
    size_t block_id;
} BlockResult;

typedef struct {
    const uint8_t *input_data;
    size_t block_start;
    size_t block_end;
    size_t total_size;
    size_t block_id;
    BlockResult *result;
    pthread_mutex_t *progress_mutex;
    size_t *completed_blocks;
    size_t total_blocks;
} ThreadArgs;

// Statistics 
typedef struct {
    size_t original_size;
    size_t compressed_size;
    double compression_ratio;
    double execution_time;
    size_t num_tokens;
    int num_threads;
} CompressionStats;

// longest match in the search buffer
void find_longest_match(const uint8_t *data, size_t position, size_t data_size,
                       size_t block_start, uint16_t *best_offset, uint16_t *best_length) {
    *best_offset = 0;
    *best_length = 0;
    
    size_t search_start = (position >= WINDOW_SIZE) ? (position - WINDOW_SIZE) : block_start;
    size_t lookahead_end = (position + LOOKAHEAD_SIZE < data_size) ? 
                           (position + LOOKAHEAD_SIZE) : data_size;
    
    for (size_t i = search_start; i < position; i++) {
        uint16_t match_length = 0;
        
        while ((position + match_length < lookahead_end) &&
               (data[i + match_length] == data[position + match_length]) &&
               (match_length < LOOKAHEAD_SIZE)) {
            match_length++;
        }
        
        if (match_length >= MIN_MATCH_LENGTH && match_length > *best_length) {
            *best_offset = position - i;
            *best_length = match_length;
        }
    }
}

// block compression
void* compress_block_thread(void *arg) {
    ThreadArgs *thread_args = (ThreadArgs*)arg;
    
    size_t block_size = thread_args->block_end - thread_args->block_start;
    thread_args->result->tokens = (LZ77_Token*)malloc(block_size * sizeof(LZ77_Token));
    thread_args->result->num_tokens = 0;
    thread_args->result->block_id = thread_args->block_id;
    
    if (!thread_args->result->tokens) {
        fprintf(stderr, "Error: Memory allocation failed for block %zu\n", 
                thread_args->block_id);
        return NULL;
    }
    
    size_t position = thread_args->block_start;
    size_t search_start = (thread_args->block_start >= OVERLAP_SIZE) ? 
                          (thread_args->block_start - OVERLAP_SIZE) : 0;
    
    while (position < thread_args->block_end) {
        uint16_t offset = 0, length = 0;
        
        find_longest_match(thread_args->input_data, position,
                          thread_args->block_end - 1, search_start, &offset, &length);

        uint8_t next_literal = thread_args->input_data[position + length];
        
        thread_args->result->tokens[thread_args->result->num_tokens].offset = offset;
        thread_args->result->tokens[thread_args->result->num_tokens].length = length;
        thread_args->result->tokens[thread_args->result->num_tokens].literal = next_literal;
        thread_args->result->num_tokens++;
        
        position += (length > 0) ? (length + 1) : 1;
    }
    
    // Update progress
    pthread_mutex_lock(thread_args->progress_mutex);
    (*thread_args->completed_blocks)++;
    printf("Block %zu/%zu completed (%.1f%%)...\r", 
           *thread_args->completed_blocks, thread_args->total_blocks,
           (*thread_args->completed_blocks * 100.0) / thread_args->total_blocks);
    fflush(stdout);
    pthread_mutex_unlock(thread_args->progress_mutex);
    
    return NULL;
}


//POSIX Threads
LZ77_Token* compress_lz77_pthread(const uint8_t *input_data, size_t input_size,
                                 size_t *num_tokens, CompressionStats *stats,
                                 int num_threads) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Calculate number of blocks
    size_t num_blocks = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Limit threads to number of blocks
    if (num_threads > (int)num_blocks) {
        num_threads = (int)num_blocks;
    }
    
    printf("Compressing with %d POSIX threads...\n", num_threads);
    stats->num_threads = num_threads;
    
    BlockResult *block_results = (BlockResult*)calloc(num_blocks, sizeof(BlockResult));
    ThreadArgs *thread_args = (ThreadArgs*)malloc(num_blocks * sizeof(ThreadArgs));
    pthread_t *threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    
    pthread_mutex_t progress_mutex = PTHREAD_MUTEX_INITIALIZER;
    size_t completed_blocks = 0;
    
    if (!block_results || !thread_args || !threads) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return NULL;
    }
    
    // Process blocks in batches
    size_t block_idx = 0;
    
    while (block_idx < num_blocks) {
        // Launch threads for current batch
        int threads_to_launch = (num_blocks - block_idx < (size_t)num_threads) ? 
                                (num_blocks - block_idx) : num_threads;
        
        for (int i = 0; i < threads_to_launch; i++) {
            size_t current_block = block_idx + i;
            size_t block_start = current_block * BLOCK_SIZE;
            size_t block_end = ((current_block + 1) * BLOCK_SIZE < input_size) ? 
                               ((current_block + 1) * BLOCK_SIZE) : input_size;
            
            thread_args[current_block].input_data = input_data;
            thread_args[current_block].block_start = block_start;
            thread_args[current_block].block_end = block_end;
            thread_args[current_block].total_size = input_size;
            thread_args[current_block].block_id = current_block;
            thread_args[current_block].result = &block_results[current_block];
            thread_args[current_block].progress_mutex = &progress_mutex;
            thread_args[current_block].completed_blocks = &completed_blocks;
            thread_args[current_block].total_blocks = num_blocks;
            
            pthread_create(&threads[i], NULL, compress_block_thread, 
                          &thread_args[current_block]);
        }
        
        // Wait for all threads 
        for (int i = 0; i < threads_to_launch; i++) {
            pthread_join(threads[i], NULL);
        }
        
        block_idx += threads_to_launch;
    }
    printf("\n");
    
    pthread_mutex_destroy(&progress_mutex);
    
    // Merge results
    size_t total_tokens = 0;
    for (size_t i = 0; i < num_blocks; i++) {
        total_tokens += block_results[i].num_tokens;
    }
    
    LZ77_Token *tokens = (LZ77_Token*)malloc(total_tokens * sizeof(LZ77_Token));
    if (!tokens) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        for (size_t i = 0; i < num_blocks; i++) {
            free(block_results[i].tokens);
        }
        free(block_results);
        free(thread_args);
        free(threads);
        return NULL;
    }
    
    size_t token_offset = 0;
    for (size_t i = 0; i < num_blocks; i++) {
        memcpy(tokens + token_offset, block_results[i].tokens, 
               block_results[i].num_tokens * sizeof(LZ77_Token));
        token_offset += block_results[i].num_tokens;
        free(block_results[i].tokens);
    }
    
    free(block_results);
    free(thread_args);
    free(threads);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    // statistics
    *num_tokens = total_tokens;
    stats->original_size = input_size;
    stats->compressed_size = total_tokens * sizeof(LZ77_Token);
    stats->compression_ratio = (double)input_size / stats->compressed_size;
    stats->execution_time = (end.tv_sec - start.tv_sec) + 
                           (end.tv_nsec - start.tv_nsec) / 1e9;
    stats->num_tokens = total_tokens;
    
    return tokens;
}


 //Decompress LZ77 tokens
uint8_t* decompress_lz77(const LZ77_Token *tokens, size_t num_tokens,
                        size_t original_size, size_t *output_size) {
    size_t estimated_size = original_size;
    uint8_t *output_data = (uint8_t*)malloc(estimated_size);
    if (!output_data) return NULL;
    
    size_t output_pos = 0;
    
    for (size_t i = 0; i < num_tokens; i++) {
        uint16_t offset = tokens[i].offset;
        uint16_t length = tokens[i].length;
        uint8_t literal = tokens[i].literal;
        
        if (length > 0 && offset > 0) {
            size_t match_pos = output_pos - offset;
            for (uint16_t j = 0; j < length; j++) {
                output_data[output_pos++] = output_data[match_pos + j];
            }
        }
        
        if (literal != 0 || i < num_tokens - 1) {
            output_data[output_pos++] = literal;
        }
    }
    
    *output_size = output_pos;
    return output_data;
}

int write_compressed_file(const char *filename, const LZ77_Token *tokens, 
                         size_t num_tokens, size_t original_size) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) return -1;
    
    fwrite(&original_size, sizeof(size_t), 1, fp);
    fwrite(&num_tokens, sizeof(size_t), 1, fp);
    fwrite(tokens, sizeof(LZ77_Token), num_tokens, fp);
    fclose(fp);
    return 0;
}

LZ77_Token* read_compressed_file(const char *filename, size_t *num_tokens, 
                                size_t *original_size) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return NULL;
    
    fread(original_size, sizeof(size_t), 1, fp);
    fread(num_tokens, sizeof(size_t), 1, fp);
    
    LZ77_Token *tokens = (LZ77_Token*)malloc(*num_tokens * sizeof(LZ77_Token));
    if (tokens) {
        fread(tokens, sizeof(LZ77_Token), *num_tokens, fp);
    }
    fclose(fp);
    return tokens;
}


// Print statistics
void print_statistics(const CompressionStats *stats, const char *input_file) {
    printf("\n===== LZ77 Compression Results =====\n");
    printf("Implementation: POSIX Threads\n");
    printf("Number of threads: %d\n", stats->num_threads);
    printf("Input file: %s\n", input_file);
    printf("Original size: %zu bytes (%.2f MB)\n", 
           stats->original_size, stats->original_size / (1024.0 * 1024.0));
    printf("Compressed size: %zu bytes (%.2f MB)\n", 
           stats->compressed_size, stats->compressed_size / (1024.0 * 1024.0));
    printf("Compression ratio: %.2f:1\n", stats->compression_ratio);
    printf("Number of tokens: %zu\n", stats->num_tokens);
    printf("Execution time: %.4f seconds\n", stats->execution_time);
    printf("Throughput: %.2f MB/s\n", 
           (stats->original_size / (1024.0 * 1024.0)) / stats->execution_time);
    printf("====================================\n\n");
}


int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_file> <output_file> [num_threads] [--decompress]\n", argv[0]);
        printf("  num_threads: number of threads to use (default: CPU cores)\n");
        return 1;
    }
    
    const char *input_file = argv[1];
    const char *output_file = argv[2];
    
    int num_threads = sysconf(_SC_NPROCESSORS_ONLN);
    int decompress_mode = 0;
    
    if (argc > 3) {
        if (strcmp(argv[3], "--decompress") != 0) {
            num_threads = atoi(argv[3]);
            if (num_threads <= 0 || num_threads > MAX_THREADS) {
                num_threads = sysconf(_SC_NPROCESSORS_ONLN);
            }
        } else {
            decompress_mode = 1;
        }
    }
    
    if (argc > 4 && strcmp(argv[4], "--decompress") == 0) {
        decompress_mode = 1;
    }
    
    if (decompress_mode) {
        printf("Decompressing %s to %s...\n", input_file, output_file);
        
        size_t num_tokens, original_size, output_size;
        LZ77_Token *tokens = read_compressed_file(input_file, &num_tokens, &original_size);
        if (!tokens) return 1;
        
        uint8_t *decompressed_data = decompress_lz77(tokens, num_tokens, original_size, &output_size);
        if (!decompressed_data) {
            free(tokens);
            return 1;
        }
        
        FILE *fp = fopen(output_file, "wb");
        if (!fp) {
            free(tokens);
            free(decompressed_data);
            return 1;
        }
        fwrite(decompressed_data, 1, output_size, fp);
        fclose(fp);
        
        printf("Decompression complete: %zu bytes\n", output_size);
        printf("Verification: %s\n", 
               (output_size == original_size) ? "PASSED ✓" : "FAILED ✗");
        
        free(tokens);
        free(decompressed_data);
        
    } else {
        printf("Compressing %s to %s...\n", input_file, output_file);
        
        FILE *fp = fopen(input_file, "rb");
        if (!fp) {
            fprintf(stderr, "Error: Cannot open input file %s\n", input_file);
            return 1;
        }
        
        fseek(fp, 0, SEEK_END);
        size_t input_size = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        
        uint8_t *input_data = (uint8_t*)malloc(input_size);
        if (!input_data) {
            fclose(fp);
            return 1;
        }
        
        fread(input_data, 1, input_size, fp);
        fclose(fp);
        
        CompressionStats stats;
        size_t num_tokens;
        LZ77_Token *tokens = compress_lz77_pthread(input_data, input_size, 
                                                   &num_tokens, &stats, num_threads);
        if (!tokens) {
            free(input_data);
            return 1;
        }
        
        if (write_compressed_file(output_file, tokens, num_tokens, input_size) != 0) {
            free(input_data);
            free(tokens);
            return 1;
        }
        
        print_statistics(&stats, input_file);
        
        free(input_data);
        free(tokens);
    }
    
    return 0;
}