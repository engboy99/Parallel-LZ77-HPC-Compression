/**
 * MPI + CUDA Hybrid LZ77 Compression Implementation
 * Module: EE7218 / EC7207 - High Performance Computing
 * 
 * Distributed GPU-accelerated LZ77 compression:
 * - MPI: Distribute blocks across nodes
 * - CUDA: GPU acceleration for pattern matching on each node
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <mpi.h>
#include <cuda_runtime.h>

// Configuration
#define WINDOW_SIZE 32768
#define LOOKAHEAD_SIZE 258
#define MIN_MATCH_LENGTH 3
#define OVERLAP_SIZE WINDOW_SIZE
#define BLOCK_SIZE_GPU 256
#define CHUNK_SIZE (512 * 1024)

// Structures
typedef struct {
    uint16_t offset;
    uint16_t length;
    uint8_t literal;
} LZ77_Token;

typedef struct {
    uint16_t offset;
    uint16_t length;
} MatchResult;

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while(0)

/**
 * CUDA kernel for pattern matching
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
    
    size_t search_start = (position >= WINDOW_SIZE) ? (position - WINDOW_SIZE) : 0;
    size_t lookahead_end = (position + LOOKAHEAD_SIZE < data_size) ? 
                           (position + LOOKAHEAD_SIZE) : data_size;
    
    for (size_t i = search_start; i < position; i++) {
        uint16_t match_length = 0;
        
        while ((position + match_length < lookahead_end) &&
               (data[i + match_length] == data[position + match_length]) &&
               (match_length < LOOKAHEAD_SIZE)) {
            match_length++;
        }
        
        if (match_length >= MIN_MATCH_LENGTH && match_length > best_length) {
            best_offset = position - i;
            best_length = match_length;
        }
    }
    
    matches[idx].offset = best_offset;
    matches[idx].length = best_length;
}

/**
 * Compress block using CUDA
 */
LZ77_Token* compress_block_cuda(const uint8_t *data, size_t start, size_t end,
                               size_t *num_tokens, int block_size) {
    // data[0..start-1]  = overlap context (empty for rank 0 where start=0)
    // data[start..end-1] = actual block to compress
    size_t actual_size = end - start;

    uint8_t *d_data;
    MatchResult *d_matches;

    // Upload full region (overlap context + actual block) so the GPU can
    // search the overlap when finding back-references.
    CUDA_CHECK(cudaMalloc(&d_data, end));
    CUDA_CHECK(cudaMalloc(&d_matches, actual_size * sizeof(MatchResult)));
    CUDA_CHECK(cudaMemcpy(d_data, data, end, cudaMemcpyHostToDevice));

    MatchResult *h_matches = (MatchResult*)malloc(actual_size * sizeof(MatchResult));
    if (!h_matches) {
        cudaFree(d_data);
        cudaFree(d_matches);
        return NULL;
    }

    // Limit lookahead to end-1 so every match leaves room for the literal
    // within this block (position + length <= end-2, literal at end-1 at most).
    size_t kernel_data_size = (end > 0) ? (end - 1) : 0;
    size_t position = start;  // begin at actual block start, not at overlap start
    while (position < end) {
        size_t remaining = end - position;
        size_t chunk_size = (remaining < CHUNK_SIZE) ? remaining : CHUNK_SIZE;
        int grid_size = (chunk_size + block_size - 1) / block_size;

        find_matches_kernel<<<grid_size, block_size>>>(
            d_data, kernel_data_size,
            d_matches + (position - start), position, chunk_size);
        CUDA_CHECK(cudaGetLastError());

        position += chunk_size;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_matches, d_matches,
                         actual_size * sizeof(MatchResult),
                         cudaMemcpyDeviceToHost));

    cudaFree(d_data);
    cudaFree(d_matches);

    LZ77_Token *tokens = (LZ77_Token*)malloc(actual_size * sizeof(LZ77_Token));
    if (!tokens) {
        free(h_matches);
        return NULL;
    }

    size_t token_count = 0;
    size_t local_pos = 0;  // offset within h_matches / actual block

    while (local_pos < actual_size) {
        uint16_t offset = h_matches[local_pos].offset;
        uint16_t length = h_matches[local_pos].length;

        // literal lives at data[start + local_pos + length]; always in-bounds
        // because kernel_data_size = end-1 guarantees start+local_pos+length <= end-2
        uint8_t next_literal = data[start + local_pos + length];

        tokens[token_count].offset = offset;
        tokens[token_count].length = length;
        tokens[token_count].literal = next_literal;
        token_count++;

        local_pos += (length > 0) ? (length + 1) : 1;
    }

    free(h_matches);
    *num_tokens = token_count;
    return tokens;
}

/**
 * MPI + CUDA hybrid compression
 */
LZ77_Token* compress_lz77_hybrid(const uint8_t *input_data, size_t input_size,
                                size_t *total_tokens, double *exec_time,
                                int rank, int mpi_size, int block_size) {
    double start_time = MPI_Wtime();
    
    // Calculate block distribution
    size_t base_block_size = input_size / mpi_size;
    size_t remainder = input_size % mpi_size;
    
    size_t block_start = rank * base_block_size + (rank < (int)remainder ? rank : remainder);
    size_t block_size_local = base_block_size + (rank < (int)remainder ? 1 : 0);
    size_t block_end = block_start + block_size_local;
    
    size_t data_start = (block_start >= OVERLAP_SIZE && rank > 0) ? 
                        (block_start - OVERLAP_SIZE) : block_start;
    
    if (rank == 0) {
        printf("Hybrid MPI+CUDA compression: %d processes, GPU block size: %d\n", 
               mpi_size, block_size);
    }
    
    // Compress local block with CUDA
    size_t local_num_tokens;
    LZ77_Token *local_tokens = compress_block_cuda(input_data + data_start,
                                                   block_start - data_start,
                                                   block_end - data_start,
                                                   &local_num_tokens,
                                                   block_size);
    
    if (!local_tokens) {
        fprintf(stderr, "Rank %d: CUDA compression failed\n", rank);
        return NULL;
    }
    
    // Gather results
    size_t *token_counts = NULL;
    if (rank == 0) {
        token_counts = (size_t*)malloc(mpi_size * sizeof(size_t));
    }
    
    MPI_Gather(&local_num_tokens, 1, MPI_UNSIGNED_LONG,
               token_counts, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    
    LZ77_Token *all_tokens = NULL;
    
    if (rank == 0) {
        *total_tokens = 0;
        for (int i = 0; i < mpi_size; i++) {
            *total_tokens += token_counts[i];
        }
        
        all_tokens = (LZ77_Token*)malloc(*total_tokens * sizeof(LZ77_Token));
        
        int *displs = (int*)malloc(mpi_size * sizeof(int));
        int *recvcounts = (int*)malloc(mpi_size * sizeof(int));
        
        int offset = 0;
        for (int i = 0; i < mpi_size; i++) {
            recvcounts[i] = token_counts[i] * sizeof(LZ77_Token);
            displs[i] = offset;
            offset += recvcounts[i];
        }
        
        MPI_Gatherv(local_tokens, local_num_tokens * sizeof(LZ77_Token), MPI_BYTE,
                   all_tokens, recvcounts, displs, MPI_BYTE, 0, MPI_COMM_WORLD);
        
        free(displs);
        free(recvcounts);
        free(token_counts);
    } else {
        MPI_Gatherv(local_tokens, local_num_tokens * sizeof(LZ77_Token), MPI_BYTE,
                   NULL, NULL, NULL, MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    
    free(local_tokens);
    
    *exec_time = MPI_Wtime() - start_time;
    return all_tokens;
}

/**
 * File I/O and utility functions
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
 * Main function
 */
int main(int argc, char *argv[]) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set GPU device based on available GPUs
    int num_gpus = 1;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus < 1) num_gpus = 1;
    CUDA_CHECK(cudaSetDevice(rank % num_gpus));
    
    if (argc < 3) {
        if (rank == 0) {
            printf("Usage: mpirun -np <N> %s <input_file> <output_file> [--block-size N] [--decompress]\n",
                   argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int block_size = BLOCK_SIZE_GPU;
    int decompress_mode = 0;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
            block_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--decompress") == 0) {
            decompress_mode = 1;
        }
    }

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
        uint8_t *input_data = NULL;
        size_t input_size = 0;

        // Read file on master
        if (rank == 0) {
            printf("Compressing %s to %s...\n", input_file, output_file);

            FILE *fp = fopen(input_file, "rb");
            if (!fp) {
                fprintf(stderr, "Error: Cannot open input file\n");
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

        // Broadcast data
        MPI_Bcast(&input_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            input_data = (uint8_t*)malloc(input_size);
        }

        MPI_Bcast(input_data, input_size, MPI_BYTE, 0, MPI_COMM_WORLD);

        // Compress
        size_t num_tokens;
        double exec_time;
        LZ77_Token *tokens = compress_lz77_hybrid(input_data, input_size, &num_tokens,
                                                 &exec_time, rank, size, block_size);

        // Write output
        if (rank == 0) {
            if (!tokens || write_compressed_file(output_file, tokens, num_tokens, input_size) != 0) {
                fprintf(stderr, "Error: Failed to write output\n");
            } else {
                size_t compressed_size = num_tokens * sizeof(LZ77_Token);
                double ratio = (double)input_size / compressed_size;
                double throughput = (input_size / (1024.0 * 1024.0)) / exec_time;

                printf("\n===== Results =====\n");
                printf("Implementation: MPI + CUDA Hybrid\n");
                printf("MPI Processes: %d\n", size);
                printf("GPU Block Size: %d\n", block_size);
                printf("Original: %zu bytes (%.2f MB)\n",
                       input_size, input_size / (1024.0 * 1024.0));
                printf("Compressed: %zu bytes (%.2f MB)\n",
                       compressed_size, compressed_size / (1024.0 * 1024.0));
                printf("Ratio: %.2f:1\n", ratio);
                printf("Time: %.4f seconds\n", exec_time);
                printf("Throughput: %.2f MB/s\n", throughput);
                printf("===================\n\n");
            }

            if (tokens) free(tokens);
        }

        free(input_data);
    }

    MPI_Finalize();
    return 0;
}
