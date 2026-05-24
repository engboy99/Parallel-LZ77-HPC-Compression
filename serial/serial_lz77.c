/**
 * Serial LZ77 Compression Implementation
 * Module: EE7218 / EC7207 - High Performance Computing
 * 
 * Baseline single-threaded LZ77 compression algorithm
 * Used for performance comparison with parallel implementations
 */

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

// Configuration parameters
#define WINDOW_SIZE 32768      // 32 KB search buffer
#define LOOKAHEAD_SIZE 258     // Maximum match length
#define MIN_MATCH_LENGTH 3     // Minimum match length to encode

// LZ77 Token structure
typedef struct {
    uint16_t offset;    // Distance to match in search buffer
    uint16_t length;    // Length of match
    uint8_t literal;    // Next literal character
} LZ77_Token;

// Statistics structure
typedef struct {
    size_t original_size;
    size_t compressed_size;
    double compression_ratio;
    double execution_time;
    size_t num_tokens;
} CompressionStats;

/**
 * Find the longest match in the search buffer
 */
void find_longest_match(const uint8_t *data, size_t position, size_t data_size,
                       uint16_t *best_offset, uint16_t *best_length) {
    *best_offset = 0;
    *best_length = 0;
    
    // Determine search range
    size_t search_start = (position >= WINDOW_SIZE) ? (position - WINDOW_SIZE) : 0;
    size_t lookahead_end = (position + LOOKAHEAD_SIZE < data_size) ? 
                           (position + LOOKAHEAD_SIZE) : data_size;
    
    // Search for matches in the search buffer
    for (size_t i = search_start; i < position; i++) {
        uint16_t match_length = 0;
        
        // Count matching characters
        while ((position + match_length < lookahead_end) &&
               (data[i + match_length] == data[position + match_length]) &&
               (match_length < LOOKAHEAD_SIZE)) {
            match_length++;
        }
        
        // Update best match if this is longer
        if (match_length >= MIN_MATCH_LENGTH && match_length > *best_length) {
            *best_offset = position - i;
            *best_length = match_length;
        }
    }
}

/**
 * Compress data using LZ77 algorithm
 */
LZ77_Token* compress_lz77(const uint8_t *input_data, size_t input_size,
                          size_t *num_tokens, CompressionStats *stats) {
    // Allocate token buffer (worst case: one token per byte)
    LZ77_Token *tokens = (LZ77_Token*)malloc(input_size * sizeof(LZ77_Token));
    if (!tokens) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return NULL;
    }
    
    size_t token_count = 0;
    size_t position = 0;
    
    clock_t start, end;
    start = clock();
    
    // Main compression loop
    while (position < input_size) {
        uint16_t offset = 0, length = 0;
        
        // Find longest match
        find_longest_match(input_data, position, input_size, &offset, &length);
        
        // Create token
        uint8_t next_literal = (position + length < input_size) ? 
                               input_data[position + length] : 0;
        
        tokens[token_count].offset = offset;
        tokens[token_count].length = length;
        tokens[token_count].literal = next_literal;
        token_count++;
        
        // Move forward
        position += (length > 0) ? (length + 1) : 1;
    }
    
    end = clock();
    
    // Calculate statistics
    *num_tokens = token_count;
    stats->original_size = input_size;
    stats->compressed_size = token_count * sizeof(LZ77_Token);
    stats->compression_ratio = (double)input_size / stats->compressed_size;
    stats->execution_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    stats->num_tokens = token_count;
    
    return tokens;
}

/**
 * Decompress LZ77 tokens back to original data
 */
uint8_t* decompress_lz77(const LZ77_Token *tokens, size_t num_tokens,
                        size_t original_size, size_t *output_size) {
    size_t estimated_size = original_size;
    uint8_t *output_data = (uint8_t*)malloc(estimated_size);
    if (!output_data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return NULL;
    }
    
    size_t output_pos = 0;
    
    // Decompress each token
    for (size_t i = 0; i < num_tokens; i++) {
        uint16_t offset = tokens[i].offset;
        uint16_t length = tokens[i].length;
        uint8_t literal = tokens[i].literal;
        
        // Copy matched data
        if (length > 0 && offset > 0) {
            size_t match_pos = output_pos - offset;
            for (uint16_t j = 0; j < length; j++) {
                output_data[output_pos++] = output_data[match_pos + j];
            }
        }
        
        // Add literal
        if (literal != 0 || i < num_tokens - 1) {
            output_data[output_pos++] = literal;
        }
    }
    
    *output_size = output_pos;
    return output_data;
}

/**
 * Write compressed tokens to file
 */
int write_compressed_file(const char *filename, const LZ77_Token *tokens, 
                         size_t num_tokens, size_t original_size) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open output file %s\n", filename);
        return -1;
    }
    
    // Write header
    fwrite(&original_size, sizeof(size_t), 1, fp);
    fwrite(&num_tokens, sizeof(size_t), 1, fp);
    
    // Write tokens
    fwrite(tokens, sizeof(LZ77_Token), num_tokens, fp);
    
    fclose(fp);
    return 0;
}

/**
 * Read compressed tokens from file
 */
LZ77_Token* read_compressed_file(const char *filename, size_t *num_tokens, 
                                size_t *original_size) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open input file %s\n", filename);
        return NULL;
    }
    
    // Read header
    fread(original_size, sizeof(size_t), 1, fp);
    fread(num_tokens, sizeof(size_t), 1, fp);
    
    // Read tokens
    LZ77_Token *tokens = (LZ77_Token*)malloc(*num_tokens * sizeof(LZ77_Token));
    if (!tokens) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(fp);
        return NULL;
    }
    
    fread(tokens, sizeof(LZ77_Token), *num_tokens, fp);
    fclose(fp);
    
    return tokens;
}

/**
 * Print compression statistics
 */
void print_statistics(const CompressionStats *stats, const char *input_file) {
    printf("\n===== LZ77 Compression Results =====\n");
    printf("Implementation: Serial\n");
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
        printf("  --decompress: decompress the input file instead of compressing\n");
        return 1;
    }
    
    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int decompress_mode = (argc > 3 && strcmp(argv[3], "--decompress") == 0);
    
    if (decompress_mode) {
        // Decompression mode
        printf("Decompressing %s to %s...\n", input_file, output_file);
        
        size_t num_tokens, original_size, output_size;
        LZ77_Token *tokens = read_compressed_file(input_file, &num_tokens, &original_size);
        if (!tokens) return 1;
        
        uint8_t *decompressed_data = decompress_lz77(tokens, num_tokens, original_size, &output_size);
        if (!decompressed_data) {
            free(tokens);
            return 1;
        }
        
        // Write decompressed data
        FILE *fp = fopen(output_file, "wb");
        if (!fp) {
            fprintf(stderr, "Error: Cannot open output file\n");
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
        // Compression mode
        printf("Compressing %s to %s...\n", input_file, output_file);
        
        // Read input file
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
            fprintf(stderr, "Error: Memory allocation failed\n");
            fclose(fp);
            return 1;
        }
        
        fread(input_data, 1, input_size, fp);
        fclose(fp);
        
        // Compress data
        CompressionStats stats;
        size_t num_tokens;
        LZ77_Token *tokens = compress_lz77(input_data, input_size, &num_tokens, &stats);
        if (!tokens) {
            free(input_data);
            return 1;
        }
        
        // Write compressed file
        if (write_compressed_file(output_file, tokens, num_tokens, input_size) != 0) {
            free(input_data);
            free(tokens);
            return 1;
        }
        
        // Print statistics
        print_statistics(&stats, input_file);
        
        free(input_data);
        free(tokens);
    }
    
    return 0;
}
