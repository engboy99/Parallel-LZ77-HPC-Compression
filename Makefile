# Makefile for LZ77 Compression Project
# Module: EE7218 / EC7207 - High Performance Computing
#
# Usage:
#   make              - Build all implementations
#   make serial       - Build serial version
#   make openmp       - Build OpenMP version
#   make pthread      - Build POSIX Threads version
#   make mpi          - Build MPI version
#   make hybrid       - Build MPI+OpenMP version
#   make cuda         - Build CUDA version
#   make cuda-hybrid  - Build MPI+CUDA version
#   make clean        - Remove all executables
#   make test         - Generate test data and run quick tests

# Compiler settings
CC = gcc
MPICC = mpicc
NVCC = nvcc
CFLAGS = -O3 -Wall
OMPFLAGS = -fopenmp
PTHREADFLAGS = -pthread
CUDAFLAGS = -arch=sm_75
LDFLAGS = -lrt

# MPI flags for CUDA compilation
MPI_COMPILE_FLAGS = $(shell mpicc --showme:compile 2>/dev/null || echo "")
MPI_LINK_FLAGS = $(shell mpicc --showme:link 2>/dev/null || echo "-lmpi")

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    EXT =
    RM = rm -f
else
    EXT = .exe
    RM = del /Q
endif

# Target executables
SERIAL_TARGET = serial/serial_lz77$(EXT)
OPENMP_TARGET = openmp/openmp_lz77$(EXT)
PTHREAD_TARGET = pthread/pthread_lz77$(EXT)
MPI_TARGET = mpi/mpi_lz77$(EXT)
HYBRID_TARGET = hybrid/mpi_openmp_lz77$(EXT)
CUDA_TARGET = cuda/cuda_lz77$(EXT)
CUDA_HYBRID_TARGET = hybrid/mpi_cuda_lz77$(EXT)

# Source files
SERIAL_SRC = serial/serial_lz77.c
OPENMP_SRC = openmp/openmp_lz77.c
PTHREAD_SRC = pthread/pthread_lz77.c
MPI_SRC = mpi/mpi_lz77.c
HYBRID_SRC = hybrid/mpi_openmp_lz77.c
CUDA_SRC = cuda/cuda_lz77.cu
CUDA_HYBRID_SRC = hybrid/mpi_cuda_lz77.cu

# Default target - build all
.PHONY: all
all: serial openmp pthread mpi hybrid

# Build with GPU support
.PHONY: all-gpu
all-gpu: all cuda cuda-hybrid

# Individual targets
.PHONY: serial
serial: $(SERIAL_TARGET)

$(SERIAL_TARGET): $(SERIAL_SRC)
	@echo "Building Serial LZ77..."
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "✓ Serial build complete"

.PHONY: openmp
openmp: $(OPENMP_TARGET)

$(OPENMP_TARGET): $(OPENMP_SRC)
	@echo "Building OpenMP LZ77..."
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDFLAGS)
	@echo "✓ OpenMP build complete"

.PHONY: pthread
pthread: $(PTHREAD_TARGET)

$(PTHREAD_TARGET): $(PTHREAD_SRC)
	@echo "Building POSIX Threads LZ77..."
	$(CC) $(CFLAGS) $(PTHREADFLAGS) -o $@ $< $(LDFLAGS)
	@echo "✓ POSIX Threads build complete"

.PHONY: mpi
mpi: $(MPI_TARGET)

$(MPI_TARGET): $(MPI_SRC)
	@echo "Building MPI LZ77..."
	$(MPICC) $(CFLAGS) -o $@ $<
	@echo "✓ MPI build complete"

.PHONY: hybrid
hybrid: $(HYBRID_TARGET)

$(HYBRID_TARGET): $(HYBRID_SRC)
	@echo "Building MPI+OpenMP Hybrid LZ77..."
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -o $@ $<
	@echo "✓ MPI+OpenMP build complete"

.PHONY: cuda
cuda: $(CUDA_TARGET)

$(CUDA_TARGET): $(CUDA_SRC)
	@echo "Building CUDA LZ77..."
	$(NVCC) -O3 $(CUDAFLAGS) -o $@ $<
	@echo "✓ CUDA build complete"

.PHONY: cuda-hybrid
cuda-hybrid: $(CUDA_HYBRID_TARGET)

$(CUDA_HYBRID_TARGET): $(CUDA_HYBRID_SRC)
	@echo "Building MPI+CUDA Hybrid LZ77..."
	$(NVCC) -O3 $(CUDAFLAGS) $(MPI_COMPILE_FLAGS) -o $@ $< $(MPI_LINK_FLAGS)
	@echo "✓ MPI+CUDA build complete"

# Clean targets
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	-$(RM) $(SERIAL_TARGET)
	-$(RM) $(OPENMP_TARGET)
	-$(RM) $(PTHREAD_TARGET)
	-$(RM) $(MPI_TARGET)
	-$(RM) $(HYBRID_TARGET)
	-$(RM) $(CUDA_TARGET)
	-$(RM) $(CUDA_HYBRID_TARGET)
	-$(RM) output/*.lz77 output/*.txt
	@echo "✓ Clean complete"

# Create necessary directories
.PHONY: dirs
dirs:
	@mkdir -p output results datasets/test_files

# Generate test data
.PHONY: testdata
testdata: dirs
	@echo "Generating test data..."
	python3 scripts/generate_test_data.py
	@echo "✓ Test data generated"

# Quick test
.PHONY: test
test: serial testdata
	@echo "Running quick test..."
	@echo "Testing Serial implementation..."
	./$(SERIAL_TARGET) datasets/test_files/1MB_mixed.txt output/test_serial.lz77
	./$(SERIAL_TARGET) output/test_serial.lz77 output/test_decompressed.txt --decompress
	@if diff -q datasets/test_files/1MB_mixed.txt output/test_decompressed.txt > /dev/null 2>&1; then \
		echo "✓ Compression/Decompression verification passed!"; \
	else \
		echo "✗ Verification failed!"; \
	fi

# Test OpenMP
.PHONY: test-openmp
test-openmp: openmp testdata
	@echo "Testing OpenMP with 4 threads..."
	OMP_NUM_THREADS=4 ./$(OPENMP_TARGET) datasets/test_files/10MB_mixed.txt output/test_openmp.lz77

# Test MPI
.PHONY: test-mpi
test-mpi: mpi testdata
	@echo "Testing MPI with 2 processes..."
	mpirun -np 2 ./$(MPI_TARGET) datasets/test_files/10MB_mixed.txt output/test_mpi.lz77

# Test CUDA
.PHONY: test-cuda
test-cuda: cuda testdata
	@echo "Testing CUDA..."
	./$(CUDA_TARGET) datasets/test_files/10MB_mixed.txt output/test_cuda.lz77

# Benchmark all implementations
.PHONY: benchmark
benchmark: all testdata
	@echo "Running comprehensive benchmarks..."
	@echo "This may take several minutes..."
	./scripts/benchmark.sh

# Display build information
.PHONY: info
info:
	@echo "===== LZ77 Compression Build System ====="
	@echo "OS: $(UNAME_S)"
	@echo "C Compiler: $(CC)"
	@echo "MPI Compiler: $(MPICC)"
	@echo "CUDA Compiler: $(NVCC)"
	@echo "Targets available:"
	@echo "  make all        - Build CPU implementations"
	@echo "  make all-gpu    - Build all including GPU"
	@echo "  make serial     - Build serial version"
	@echo "  make openmp     - Build OpenMP version"
	@echo "  make pthread    - Build PThreads version"
	@echo "  make mpi        - Build MPI version"
	@echo "  make hybrid     - Build MPI+OpenMP version"
	@echo "  make cuda       - Build CUDA version"
	@echo "  make cuda-hybrid- Build MPI+CUDA version"
	@echo "  make test       - Run quick tests"
	@echo "  make benchmark  - Run full benchmarks"
	@echo "  make clean      - Remove executables"
	@echo "=========================================="

# Help target
.PHONY: help
help: info

# Install dependencies (for Debian/Ubuntu)
.PHONY: install-deps
install-deps:
	@echo "Installing dependencies..."
	sudo apt-get update
	sudo apt-get install -y gcc g++ make
	sudo apt-get install -y libgomp1
	sudo apt-get install -y openmpi-bin openmpi-common libopenmpi-dev
	sudo apt-get install -y python3 python3-pip
	@echo "✓ Dependencies installed"
	@echo "Note: CUDA must be installed separately from NVIDIA"

# Check compiler versions
.PHONY: check
check:
	@echo "Checking compiler versions..."
	@$(CC) --version | head -n1
	@$(MPICC) --version | head -n1 || echo "MPI not found"
	@$(NVCC) --version | tail -n1 || echo "CUDA not found"
	@python3 --version || echo "Python3 not found"

.DEFAULT_GOAL := help
