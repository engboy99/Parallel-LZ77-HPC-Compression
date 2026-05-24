# Parallel LZ77 HPC Compression

## Overview

This project presents a High-Performance Computing (HPC) implementation of the LZ77 compression algorithm using:

- MPI (Message Passing Interface)
- OpenMP Parallelism
- Sequential Baseline Execution

The project evaluates scalability, execution time, speedup, and computational efficiency across different parallel configurations.

---

# Features

- Sequential LZ77 implementation
- MPI distributed-memory parallel execution
- OpenMP shared-memory multithreading
- Benchmarking and performance evaluation
- Graph generation and scalability analysis
- HPC-oriented runtime analysis

---

# Technologies Used

- Python
- MPI (mpi4py)
- OpenMP
- Linux / WSL
- Matplotlib
- NumPy

---

# Folder Structure

```bash
.
├── sequential_lz77.py
├── mpi_lz77.py
├── openmp_lz77.py
├── benchmark.py
├── results/
├── graphs/
├── README.md
└── requirements.txt
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Parallel-LZ77-HPC-Compression.git
cd Parallel-LZ77-HPC-Compression
```

---

# Create Virtual Environment

```bash
python -m venv hpc_env
```

Activate environment:

## Linux / WSL

```bash
source hpc_env/bin/activate
```

## Windows

```bash
hpc_env\Scripts\activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Sequential Execution

```bash
time python sequential_lz77.py
```

---

## MPI Parallel Execution

```bash
mpirun -np 4 python mpi_lz77.py
```

---

## OpenMP Parallel Execution

```bash
OMP_NUM_THREADS=4 python openmp_lz77.py
```

---

# Benchmarking

Run benchmark analysis:

```bash
python benchmark.py
```

This generates:

- Execution time analysis
- Speedup graphs
- Efficiency graphs
- Scalability evaluation

---

# Performance Metrics

The project evaluates:

- Execution Time
- Speedup
- Parallel Efficiency
- Scalability
- Communication Overhead

---

# HPC Concepts Demonstrated

- Distributed-memory parallelism
- Shared-memory multithreading
- Workload decomposition
- Synchronization overhead
- Scalability analysis
- Runtime optimization

---

# Example Performance Analysis

## Speedup Formula

\[
Speedup = \frac{T_{sequential}}{T_{parallel}}
\]

## Efficiency Formula

\[
Efficiency = \frac{Speedup}{Processors}
\]

---

# Future Improvements

- GPU acceleration using CUDA
- Hybrid MPI + OpenMP architecture
- Dynamic workload balancing
- Real-time distributed compression
- Cloud HPC deployment

---

# Research Significance

This project demonstrates how HPC techniques can improve the performance of computationally intensive compression algorithms through parallel execution and scalable processing architectures.

---

# Author

Prabhath

---

# License

This project is licensed under the MIT License.