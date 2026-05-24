/**
 * MPI Distributed LZ77 Compression Implementation
 * Module: EE7218 / EC7207 - High Performance Computing
 * 
 * Distributed memory parallel LZ77 compression using MPI
 * Implements block-based data partitioning across multiple nodes
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <mpi.h>

// Configuration parameters
#define WINDOW_SIZE 32768
#define LOOKAHEAD_SIZE 258
#define MIN_MATCH_LENGTH 3
#define OVERLAP_SIZE WINDOW_SIZE

// LZ77 Token structure
typedef struct {
    uint16_t offset;
    uint16_t length;
    uint8_t literal;
} LZ77_Token;

// Statistics structure
typedef struct {
    size_t original_size;
    size_t compressed_size;
    double compression_ratio;
    double execution_time;
    size_t num_tokens;
    int num_processes;
} CompressionStats;

/**
 * Find the longest match in the search buffer
 */
void find_longest_match(const uint8_t *data, size_t position, size_t data_size,
                       size_t search_start, uint16_t *best_offset, uint16_t *best_length) {
    *best_offset = 0;
    *best_length = 0;
    
    size_t local_search_start = (position >= WINDOW_SIZE) ? 
                                (position - WINDOW_SIZE) : search_start;
    size_t lookahead_end = (position + LOOKAHEAD_SIZE < data_size) ? 
                           (position + LOOKAHEAD_SIZE) : data_size;
    
    for (size_t i = local_search_start; i < position; i++) {
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
 * Compress a block of data
 */
LZ77_Token* compress_block(const uint8_t *data, size_t start, size_t end,
                          size_t *num_tokens) {
    size_t max_tokens = end - start;
    LZ77_Token *tokens = (LZ77_Token*)malloc(max_tokens * sizeof(LZ77_Token));
    if (!tokens) return NULL;
    
    *num_tokens = 0;
    size_t position = start;
    
    while (position < end) {
        uint16_t offset = 0, length = 0;
        
        find_longest_match(data, position, end - 1, start, &offset, &length);

        uint8_t next_literal = data[position + length];
        
        tokens[*num_tokens].offset = offset;
        tokens[*num_tokens].length = length;
        tokens[*num_tokens].literal = next_literal;
        (*num_tokens)++;
        
        position += (length > 0) ? (length + 1) : 1;
    }
    
    return tokens;
}

/**
 * Parallel LZ77 compression using MPI
 */
LZ77_Token* compress_lz77_mpi(const uint8_t *input_data, size_t input_size,
                             size_t *total_tokens, CompressionStats *stats,
                             int rank, int size) {
    double start_time = MPI_Wtime();
    
    // Calculate block distribution
    size_t base_block_size = input_size / size;
    size_t remainder = input_size % size;
    
    // Calculate this process's block range (with overlap)
    size_t block_start = rank * base_block_size + (rank < (int)remainder ? rank : remainder);
    size_t block_size = base_block_size + (rank < (int)remainder ? 1 : 0);
    size_t block_end = block_start + block_size;
    
    // Add overlap for compression context
    size_t data_start = (block_start >= OVERLAP_SIZE && rank > 0) ? 
                        (block_start - OVERLAP_SIZE) : block_start;
    size_t data_end = block_end;
    size_t data_size = data_end - data_start;
    
    if (rank == 0) {
        printf("Compressing with %d MPI processes...\n", size);
        printf("Total data size: %zu bytes (%.2f MB)\n", 
               input_size, input_size / (1024.0 * 1024.0));
    }
    
    // Compress local block
    size_t local_num_tokens;
    LZ77_Token *local_tokens = compress_block(input_data + data_start, 
                                              block_start - data_start,
                                              data_end - data_start,
                                              &local_num_tokens);
    
    if (!local_tokens) {
        fprintf(stderr, "Error: Rank %d failed to compress block\n", rank);
        return NULL;
    }
    
    if (rank == 0) {
        printf("Rank %d compressed %zu tokens\n", rank, local_num_tokens);
    }
    
    // Gather token counts from all processes
    size_t *token_counts = NULL;
    if (rank == 0) {
        token_counts = (size_t*)malloc(size * sizeof(size_t));
    }
    
    MPI_Gather(&local_num_tokens, 1, MPI_UNSIGNED_LONG,
               token_counts, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    
    LZ77_Token *all_tokens = NULL;
    
    if (rank == 0) {
        // Calculate total tokens and displacements
        *total_tokens = 0;
        for (int i = 0; i < size; i++) {
            *total_tokens += token_counts[i];
        }
        
        all_tokens = (LZ77_Token*)malloc(*total_tokens * sizeof(LZ77_Token));
        if (!all_tokens) {
            fprintf(stderr, "Error: Memory allocation failed on master\n");
            free(token_counts);
            free(local_tokens);
            return NULL;
        }
        
        // Prepare for MPI_Gatherv
        int *displs = (int*)malloc(size * sizeof(int));
        int *recvcounts = (int*)malloc(size * sizeof(int));
        
        int offset = 0;
        for (int i = 0; i < size; i++) {
            recvcounts[i] = token_counts[i] * sizeof(LZ77_Token);
            displs[i] = offset;
            offset += recvcounts[i];
        }
        
        // Gather all tokens
        MPI_Gatherv(local_tokens, local_num_tokens * sizeof(LZ77_Token), MPI_BYTE,
                   all_tokens, recvcounts, displs, MPI_BYTE,
                   0, MPI_COMM_WORLD);
        
        free(displs);
        free(recvcounts);
        free(token_counts);
        
    } else {
        // Send tokens to master
        MPI_Gatherv(local_tokens, local_num_tokens * sizeof(LZ77_Token), MPI_BYTE,
                   NULL, NULL, NULL, MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    
    free(local_tokens);
    
    double end_time = MPI_Wtime();
    
    // Calculate statistics (only on master)
    if (rank == 0) {
        stats->original_size = input_size;
        stats->compressed_size = *total_tokens * sizeof(LZ77_Token);
        stats->compression_ratio = (double)input_size / stats->compressed_size;
        stats->execution_time = end_time - start_time;
        stats->num_tokens = *total_tokens;
        stats->num_processes = size;
    }
    
    return all_tokens;
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
 * File I/O functions
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

/**
 * Print statistics (only on master)
 */
void print_statistics(const CompressionStats *stats, const char *input_file) {
    printf("\n===== LZ77 Compression Results =====\n");
    printf("Implementation: MPI Distributed\n");
    printf("Number of processes: %d\n", stats->num_processes);
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
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 3) {
        if (rank == 0) {
            printf("Usage: mpirun -np <N> %s <input_file> <output_file> [--decompress]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int decompress_mode = (argc > 3 && strcmp(argv[3], "--decompress") == 0);
    
    if (decompress_mode) {
        // Only master handles decompression
        if (rank == 0) {
            printf("Decompressing %s to %s...\n", input_file, output_file);
            
            size_t num_tokens, original_size, output_size;
            LZ77_Token *tokens = read_compressed_file(input_file, &num_tokens, &original_size);
            if (!tokens) {
                MPI_Finalize();
                return 1;
            }
            
            uint8_t *decompressed_data = decompress_lz77(tokens, num_tokens, original_size, &output_size);
            if (!decompressed_data) {
                free(tokens);
                MPI_Finalize();
                return 1;
            }
            
            FILE *fp = fopen(output_file, "wb");
            if (!fp) {
                free(tokens);
                free(decompressed_data);
                MPI_Finalize();
                return 1;
            }
            fwrite(decompressed_data, 1, output_size, fp);
            fclose(fp);
            
            printf("Decompression complete: %zu bytes\n", output_size);
            printf("Verification: %s\n", 
                   (output_size == original_size) ? "PASSED ✓" : "FAILED ✗");
            
            free(tokens);
            free(decompressed_data);
        }
        
    } else {
        // Compression mode
        uint8_t *input_data = NULL;
        size_t input_size = 0;
        
        // Master reads the file
        if (rank == 0) {
            printf("Compressing %s to %s...\n", input_file, output_file);
            
            FILE *fp = fopen(input_file, "rb");
            if (!fp) {
                fprintf(stderr, "Error: Cannot open input file %s\n", input_file);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            fseek(fp, 0, SEEK_END);
            input_size = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            
            input_data = (uint8_t*)malloc(input_size);
            if (!input_data) {
                fclose(fp);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            fread(input_data, 1, input_size, fp);
            fclose(fp);
        }
        
        // Broadcast data size
        MPI_Bcast(&input_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        
        // Allocate buffer on non-master processes
        if (rank != 0) {
            input_data = (uint8_t*)malloc(input_size);
            if (!input_data) {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        
        // Broadcast the data
        MPI_Bcast(input_data, input_size, MPI_BYTE, 0, MPI_COMM_WORLD);
        
        // Compress in parallel
        CompressionStats stats;
        size_t num_tokens;
        LZ77_Token *tokens = compress_lz77_mpi(input_data, input_size, 
                                              &num_tokens, &stats, rank, size);
        
        // Master writes the output
        if (rank == 0) {
            if (!tokens || write_compressed_file(output_file, tokens, num_tokens, input_size) != 0) {
                fprintf(stderr, "Error: Failed to write output file\n");
            } else {
                print_statistics(&stats, input_file);
            }
            
            if (tokens) free(tokens);
        }
        
        free(input_data);
    }
    
    MPI_Finalize();
    return 0;
}
