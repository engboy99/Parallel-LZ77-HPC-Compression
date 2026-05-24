/**
 * OpenMP Parallel LZ77 Compression Implementation
 * Module: EE7218 / EC7207 - High Performance Computing
 * 
 * Parallel LZ77 compression using OpenMP for shared memory parallelism
 * Implements block-based parallel compression with dynamic scheduling
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <omp.h>

// Configuration parameters
#define WINDOW_SIZE 32768
#define LOOKAHEAD_SIZE 258
#define MIN_MATCH_LENGTH 3
#define BLOCK_SIZE (1024 * 1024)  // 1 MB blocks
#define OVERLAP_SIZE WINDOW_SIZE   // Overlap for maintaining compression context

// LZ77 Token structure
typedef struct {
    uint16_t offset;
    uint16_t length;
    uint8_t literal;
} LZ77_Token;

// Block compression result
typedef struct {
    LZ77_Token *tokens;
    size_t num_tokens;
    size_t block_id;
} BlockResult;

// Statistics structure
typedef struct {
    size_t original_size;
    size_t compressed_size;
    double compression_ratio;
    double execution_time;
    size_t num_tokens;
    int num_threads;
} CompressionStats;

/**
 * Find the longest match in the search buffer
 */
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

/**
 * Compress a single block of data
 */
BlockResult compress_block(const uint8_t *input_data, size_t block_start, 
                          size_t block_end, size_t total_size, size_t block_id) {
    BlockResult result;
    
    // Allocate token buffer for this block
    size_t block_size = block_end - block_start;
    result.tokens = (LZ77_Token*)malloc(block_size * sizeof(LZ77_Token));
    result.num_tokens = 0;
    result.block_id = block_id;
    
    if (!result.tokens) {
        fprintf(stderr, "Error: Memory allocation failed for block %zu\n", block_id);
        return result;
    }
    
    size_t position = block_start;
    
    while (position < block_end) {
        uint16_t offset = 0, length = 0;
        
        find_longest_match(input_data, position, block_end - 1,
                          (block_start >= OVERLAP_SIZE) ? (block_start - OVERLAP_SIZE) : 0,
                          &offset, &length);

        uint8_t next_literal = input_data[position + length];
        
        result.tokens[result.num_tokens].offset = offset;
        result.tokens[result.num_tokens].length = length;
        result.tokens[result.num_tokens].literal = next_literal;
        result.num_tokens++;
        
        position += (length > 0) ? (length + 1) : 1;
    }
    
    return result;
}

/**
 * Parallel LZ77 compression using OpenMP
 */
LZ77_Token* compress_lz77_parallel(const uint8_t *input_data, size_t input_size,
                                  size_t *num_tokens, CompressionStats *stats) {
    double start_time = omp_get_wtime();
    
    // Calculate number of blocks
    size_t num_blocks = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    BlockResult *block_results = (BlockResult*)malloc(num_blocks * sizeof(BlockResult));
    
    if (!block_results) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return NULL;
    }
    
    // Get number of threads
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    stats->num_threads = num_threads;
    
    printf("Compressing with %d OpenMP threads...\n", num_threads);
    
    // Parallel block compression
    #pragma omp parallel for schedule(dynamic) 
    for (size_t i = 0; i < num_blocks; i++) {
        size_t block_start = i * BLOCK_SIZE;
        size_t block_end = ((i + 1) * BLOCK_SIZE < input_size) ? 
                           ((i + 1) * BLOCK_SIZE) : input_size;
        
        #pragma omp critical
        {
            printf("Thread %d processing block %zu/%zu (%.1f%%)...\r", 
                   omp_get_thread_num(), i + 1, num_blocks, 
                   ((i + 1) * 100.0) / num_blocks);
            fflush(stdout);
        }
        
        block_results[i] = compress_block(input_data, block_start, block_end, 
                                         input_size, i);
    }
    printf("\n");
    
    // Merge block results
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
        return NULL;
    }
    
    // Copy tokens from all blocks
    size_t token_offset = 0;
    for (size_t i = 0; i < num_blocks; i++) {
        memcpy(tokens + token_offset, block_results[i].tokens, 
               block_results[i].num_tokens * sizeof(LZ77_Token));
        token_offset += block_results[i].num_tokens;
        free(block_results[i].tokens);
    }
    free(block_results);
    
    double end_time = omp_get_wtime();
    
    // Calculate statistics
    *num_tokens = total_tokens;
    stats->original_size = input_size;
    stats->compressed_size = total_tokens * sizeof(LZ77_Token);
    stats->compression_ratio = (double)input_size / stats->compressed_size;
    stats->execution_time = end_time - start_time;
    stats->num_tokens = total_tokens;
    
    return tokens;
}

/**
 * Decompress LZ77 tokens
 */
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

/**
 * Write compressed file
 */
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

/**
 * Read compressed file
 */
LZ77_Token* read_compressed_file(const char *filename, size_t *num_tokens, 
                                size_t *original_size) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return NULL;
    
    fread(original_size, sizeof(size_t), 1, fp);
    fread(num_tokens, sizeof(size_t), 1, fp);
    
    LZ77_Token *tokens = (LZ77_Token*)malloc(*num_tokens * sizeof(LZ77_Token));
    if (!tokens) {
        fclose(fp);
        return NULL;
    }
    
    fread(tokens, sizeof(LZ77_Token), *num_tokens, fp);
    fclose(fp);
    
    return tokens;
}

/**
 * Print statistics
 */
void print_statistics(const CompressionStats *stats, const char *input_file) {
    printf("\n===== LZ77 Compression Results =====\n");
    printf("Implementation: OpenMP Parallel\n");
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

/**
 * Main function
 */
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_file> <output_file> [--decompress]\n", argv[0]);
        printf("Set OMP_NUM_THREADS environment variable to control thread count\n");
        return 1;
    }
    
    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int decompress_mode = (argc > 3 && strcmp(argv[3], "--decompress") == 0);
    
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
        LZ77_Token *tokens = compress_lz77_parallel(input_data, input_size, 
                                                   &num_tokens, &stats);
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
