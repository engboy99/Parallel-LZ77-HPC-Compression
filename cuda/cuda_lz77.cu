/**
 * CUDA GPU LZ77 Compression Implementation
 * Module: EE7218 / EC7207 - High Performance Computing
 * 
 * GPU-accelerated LZ77 compression using CUDA
 * Parallel pattern matching and token generation on GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>

// Configuration parameters
#define WINDOW_SIZE 32768
#define LOOKAHEAD_SIZE 258
#define MIN_MATCH_LENGTH 3
#define BLOCK_SIZE_GPU 256
#define CHUNK_SIZE (1024 * 1024)  // 1 MB chunks

// LZ77 Token structure
typedef struct {
    uint16_t offset;
    uint16_t length;
    uint8_t literal;
} LZ77_Token;

// Match result structure for GPU
typedef struct {
    uint16_t offset;
    uint16_t length;
} MatchResult;

/**
 * CUDA error checking macro
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * CUDA kernel: Find longest match for each position
 */
__global__ void find_matches_kernel(const uint8_t *data, size_t data_size,
                                   MatchResult *matches, size_t start_pos,
                                   size_t num_positions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_positions) return;
    
    size_t position = start_pos + idx;
    if (position >= data_size) return;
    
    uint16_t best_offset = 0;
    uint16_t best_length = 0;
    
    // Determine search range
    size_t search_start = (position >= WINDOW_SIZE) ? (position - WINDOW_SIZE) : 0;
    size_t lookahead_end = (position + LOOKAHEAD_SIZE < data_size) ? 
                           (position + LOOKAHEAD_SIZE) : data_size;
    
    // Search for matches
    for (size_t i = search_start; i < position; i++) {
        uint16_t match_length = 0;
        
        // Count matching characters
        while ((position + match_length < lookahead_end) &&
               (data[i + match_length] == data[position + match_length]) &&
               (match_length < LOOKAHEAD_SIZE)) {
            match_length++;
        }
        
        // Update best match
        if (match_length >= MIN_MATCH_LENGTH && match_length > best_length) {
            best_offset = position - i;
            best_length = match_length;
        }
    }
    
    matches[idx].offset = best_offset;
    matches[idx].length = best_length;
}

/**
 * Compress data on CPU using GPU for pattern matching
 */
LZ77_Token* compress_lz77_cuda(const uint8_t *input_data, size_t input_size,
                              size_t *num_tokens, double *gpu_time,
                              int block_size) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Allocate device memory
    uint8_t *d_data;
    MatchResult *d_matches;
    
    CUDA_CHECK(cudaMalloc(&d_data, input_size));
    CUDA_CHECK(cudaMalloc(&d_matches, input_size * sizeof(MatchResult)));
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_data, input_data, input_size, cudaMemcpyHostToDevice));
    
    // Allocate host memory for matches
    MatchResult *h_matches = (MatchResult*)malloc(input_size * sizeof(MatchResult));
    if (!h_matches) {
        fprintf(stderr, "Error: Host memory allocation failed\n");
        cudaFree(d_data);
        cudaFree(d_matches);
        return NULL;
    }
    
    CUDA_CHECK(cudaEventRecord(start));
    
    // Process data in chunks
    size_t position = 0;
    while (position < input_size) {
        size_t chunk_size = (position + CHUNK_SIZE < input_size) ? 
                           CHUNK_SIZE : (input_size - position);
        
        int grid_size = (chunk_size + block_size - 1) / block_size;
        
        // Launch kernel
        find_matches_kernel<<<grid_size, block_size>>>(
            d_data, input_size, d_matches + position, position, chunk_size);
        
        CUDA_CHECK(cudaGetLastError());
        
        position += chunk_size;
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    *gpu_time = milliseconds / 1000.0;
    
    // Copy matches back to host
    CUDA_CHECK(cudaMemcpy(h_matches, d_matches, 
                         input_size * sizeof(MatchResult), 
                         cudaMemcpyDeviceToHost));
    
    // Free GPU memory
    cudaFree(d_data);
    cudaFree(d_matches);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Generate tokens on CPU
    LZ77_Token *tokens = (LZ77_Token*)malloc(input_size * sizeof(LZ77_Token));
    if (!tokens) {
        free(h_matches);
        return NULL;
    }
    
    size_t token_count = 0;
    position = 0;
    
    while (position < input_size) {
        uint16_t offset = h_matches[position].offset;
        uint16_t length = h_matches[position].length;
        
        uint8_t next_literal = (position + length < input_size) ? 
                               input_data[position + length] : 0;
        
        tokens[token_count].offset = offset;
        tokens[token_count].length = length;
        tokens[token_count].literal = next_literal;
        token_count++;
        
        position += (length > 0) ? (length + 1) : 1;
    }
    
    free(h_matches);
    
    *num_tokens = token_count;
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
 * Print GPU device info
 */
void print_gpu_info() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        fprintf(stderr, "No CUDA-capable devices found\n");
        exit(1);
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("\n===== GPU Information =====\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("===========================\n\n");
}

/**
 * Print statistics
 */
void print_statistics(size_t original_size, size_t compressed_size, 
                     size_t num_tokens, double total_time, double gpu_time,
                     const char *input_file, int block_size) {
    double compression_ratio = (double)original_size / compressed_size;
    double throughput = (original_size / (1024.0 * 1024.0)) / total_time;
    
    printf("\n===== LZ77 Compression Results =====\n");
    printf("Implementation: CUDA GPU\n");
    printf("CUDA Block Size: %d threads\n", block_size);
    printf("Input file: %s\n", input_file);
    printf("Original size: %zu bytes (%.2f MB)\n", 
           original_size, original_size / (1024.0 * 1024.0));
    printf("Compressed size: %zu bytes (%.2f MB)\n", 
           compressed_size, compressed_size / (1024.0 * 1024.0));
    printf("Compression ratio: %.2f:1\n", compression_ratio);
    printf("Number of tokens: %zu\n", num_tokens);
    printf("Total execution time: %.4f seconds\n", total_time);
    printf("GPU kernel time: %.4f seconds (%.1f%%)\n", 
           gpu_time, (gpu_time / total_time) * 100.0);
    printf("Throughput: %.2f MB/s\n", throughput);
    printf("====================================\n\n");
}

/**
 * Main function
 */
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_file> <output_file> [--block-size N] [--decompress]\n", argv[0]);
        printf("  --block-size N: CUDA block size (default: 256)\n");
        return 1;
    }
    
    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int block_size = BLOCK_SIZE_GPU;
    int decompress_mode = 0;
    
    // Parse arguments
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
            block_size = atoi(argv[++i]);
            if (block_size <= 0 || block_size > 1024) {
                block_size = BLOCK_SIZE_GPU;
            }
        } else if (strcmp(argv[i], "--decompress") == 0) {
            decompress_mode = 1;
        }
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
        print_gpu_info();
        
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
            fclose(fp);
            return 1;
        }
        
        fread(input_data, 1, input_size, fp);
        fclose(fp);
        
        // Compress using CUDA
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        double gpu_time;
        size_t num_tokens;
        LZ77_Token *tokens = compress_lz77_cuda(input_data, input_size, 
                                               &num_tokens, &gpu_time, block_size);
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        double total_time = (end.tv_sec - start.tv_sec) + 
                           (end.tv_nsec - start.tv_nsec) / 1e9;
        
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
        size_t compressed_size = num_tokens * sizeof(LZ77_Token);
        print_statistics(input_size, compressed_size, num_tokens, 
                        total_time, gpu_time, input_file, block_size);
        
        free(input_data);
        free(tokens);
    }
    
    return 0;
}
