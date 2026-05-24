#!/usr/bin/env bash
#
# LZ77 Comprehensive Benchmark Script
#
# Runs all 7 implementations across all test files with varying parallelism
# parameters, then measures compression time, decompression time, compression
# ratio, throughput, and accuracy (% match + RMSE vs original).
#
# Output: test_results_<timestamp>.csv in the project root.
#
# Usage:
#   chmod +x scripts/benchmark.sh
#   ./scripts/benchmark.sh
#
# Optional env vars:
#   SKIP_LARGE=1   Skip 50MB and 100MB files (much faster run)
#   MPI_EXTRA=""   Extra mpirun flags, e.g. "--allow-run-as-root"
#

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Binaries ──────────────────────────────────────────────────────────────────
SERIAL="$PROJECT_ROOT/serial/serial_lz77"
OPENMP="$PROJECT_ROOT/openmp/openmp_lz77"
PTHREAD="$PROJECT_ROOT/pthread/pthread_lz77"
MPI_BIN="$PROJECT_ROOT/mpi/mpi_lz77"
MPI_OMP="$PROJECT_ROOT/hybrid/mpi_openmp_lz77"
CUDA_BIN="$PROJECT_ROOT/cuda/cuda_lz77"
MPI_CUDA="$PROJECT_ROOT/hybrid/mpi_cuda_lz77"

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_DIR="$PROJECT_ROOT/scripts/datasets/test_files"
ACCURACY_SCRIPT="$SCRIPT_DIR/compare_accuracy.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WORK_DIR="$PROJECT_ROOT/output/bench_$TIMESTAMP"
CSV_FILE="$PROJECT_ROOT/test_results_$TIMESTAMP.csv"
MPI_EXTRA="${MPI_EXTRA:-}"

mkdir -p "$WORK_DIR"

# ── Logging ───────────────────────────────────────────────────────────────────
log()  { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*"; }
warn() { printf "[WARN] %s\n" "$*" >&2; }

# ── Timing ────────────────────────────────────────────────────────────────────
now_ns()    { date +%s%N; }
elapsed_s() { python3 -c "print(f'{($2 - $1)/1e9:.6f}')"; }

# ── File helpers ──────────────────────────────────────────────────────────────
file_size() { stat -c%s "$1" 2>/dev/null || echo 0; }

size_mb() {
    local bytes="$1"
    python3 -c "print(f'{$bytes/1048576:.1f}')"
}

# ── Accuracy: fast path uses cmp(1), slow path uses Python ───────────────────
compute_accuracy() {
    local original="$1" decompressed="$2"
    if cmp -s "$original" "$decompressed" 2>/dev/null; then
        echo "100.0000,0.0000"
    else
        python3 "$ACCURACY_SCRIPT" "$original" "$decompressed"
    fi
}

# ── CSV setup ─────────────────────────────────────────────────────────────────
echo "timestamp,implementation,config,test_file,file_size_bytes,compressed_size_bytes,\
compression_ratio,compression_time_s,decompression_time_s,\
compress_throughput_mbps,decompress_throughput_mbps,\
accuracy_percent,rmse" \
    | tr -d ' ' > "$CSV_FILE"

append_row() {
    echo "$(date +%Y-%m-%dT%H:%M:%S),$1,$2,$3,$4,$5,$6,$7,$8,$9,${10},${11},${12}" >> "$CSV_FILE"
}

# ── Core run function ─────────────────────────────────────────────────────────
# run_test <impl> <config> <input_file> <compress_cmd> <decompress_cmd> <comp_path> <decomp_path>
run_test() {
    local impl="$1"      config="$2"    input="$3"
    local comp_cmd="$4"  decomp_cmd="$5"
    local comp_path="$6" decomp_path="$7"

    local fname fsize
    fname=$(basename "$input")
    fsize=$(file_size "$input")

    log "  [$impl | $config] $fname"

    # ── Compress ──────────────────────────────────────────────────────────────
    local t0 t1 ctime
    t0=$(now_ns)
    if ! eval "$comp_cmd" >/dev/null 2>&1; then
        warn "$impl [$config] compression FAILED: $fname"
        append_row "$impl" "$config" "$fname" "$fsize" "0" "0" \
                   "FAILED" "FAILED" "0" "0" "0" "9999.0000"
        return
    fi
    t1=$(now_ns)
    ctime=$(elapsed_s "$t0" "$t1")

    local csize ratio c_tput
    csize=$(file_size "$comp_path")
    ratio=$(python3 -c "print(f'{$fsize / max($csize,1):.4f}')")
    c_tput=$(python3 -c "print(f'{($fsize/1048576)/$ctime:.4f}' if float('$ctime')>0 else '0.0000')")

    # ── Decompress ────────────────────────────────────────────────────────────
    local t2 t3 dtime
    t2=$(now_ns)
    if ! eval "$decomp_cmd" >/dev/null 2>&1; then
        warn "$impl [$config] decompression FAILED: $fname"
        append_row "$impl" "$config" "$fname" "$fsize" "$csize" "$ratio" \
                   "$ctime" "FAILED" "$c_tput" "0" "0" "9999.0000"
        rm -f "$comp_path"
        return
    fi
    t3=$(now_ns)
    dtime=$(elapsed_s "$t2" "$t3")
    local d_tput
    d_tput=$(python3 -c "print(f'{($fsize/1048576)/$dtime:.4f}' if float('$dtime')>0 else '0.0000')")

    # ── Accuracy vs original ──────────────────────────────────────────────────
    local metrics accuracy rmse
    metrics=$(compute_accuracy "$input" "$decomp_path")
    accuracy="${metrics%%,*}"
    rmse="${metrics##*,}"

    append_row "$impl" "$config" "$fname" "$fsize" "$csize" "$ratio" \
               "$ctime" "$dtime" "$c_tput" "$d_tput" "$accuracy" "$rmse"

    rm -f "$comp_path" "$decomp_path"
}

# ── Availability ──────────────────────────────────────────────────────────────
HAS_MPI=false;  command -v mpirun      &>/dev/null && HAS_MPI=true
HAS_CUDA=false; command -v nvidia-smi  &>/dev/null && HAS_CUDA=true

log "LZ77 Benchmark — $TIMESTAMP"
log "  serial     : $([ -x "$SERIAL"   ] && echo OK || echo MISSING)"
log "  openmp     : $([ -x "$OPENMP"   ] && echo OK || echo MISSING)"
log "  pthread    : $([ -x "$PTHREAD"  ] && echo OK || echo MISSING)"
log "  mpi        : $([ -x "$MPI_BIN"  ] && $HAS_MPI  && echo OK || echo MISSING)"
log "  mpi+openmp : $([ -x "$MPI_OMP"  ] && $HAS_MPI  && echo OK || echo MISSING)"
log "  cuda       : $([ -x "$CUDA_BIN" ] && $HAS_CUDA && echo OK || echo MISSING/NO-GPU)"
log "  mpi+cuda   : $([ -x "$MPI_CUDA" ] && $HAS_MPI && $HAS_CUDA && echo OK || echo MISSING/NO-GPU)"
log "  CSV output : $CSV_FILE"
[ "${SKIP_LARGE:-0}" = "1" ] && log "  SKIP_LARGE=1 — skipping 50MB and 100MB files"
echo ""

# ── Gather test files ─────────────────────────────────────────────────────────
mapfile -t TEST_FILES < <(find "$TEST_DIR" -maxdepth 1 -name "*.txt" -printf "%s\t%p\n" | sort -n | cut -f2-)
if [ ${#TEST_FILES[@]} -eq 0 ]; then
    echo "No test files found in $TEST_DIR"; exit 1
fi

TOTAL_START=$(now_ns)

for input_file in "${TEST_FILES[@]}"; do
    fname=$(basename "$input_file")
    fsize=$(file_size "$input_file")

    # Optionally skip large files
    if [ "${SKIP_LARGE:-0}" = "1" ] && [ "$fsize" -ge 41943040 ]; then
        log "=== SKIP $fname ($(size_mb "$fsize") MB) ==="
        continue
    fi

    log "=== $fname — $(size_mb "$fsize") MB ==="

    # ── 1. Serial (baseline) ─────────────────────────────────────────────────
    if [ -x "$SERIAL" ]; then
        comp="$WORK_DIR/serial_${fname}.lz77"
        dec="$WORK_DIR/serial_${fname}.out"
        run_test "serial" "serial" "$input_file" \
            "\"$SERIAL\" \"$input_file\" \"$comp\"" \
            "\"$SERIAL\" \"$comp\" \"$dec\" --decompress" \
            "$comp" "$dec"
    fi

    # ── 2. OpenMP — threads: 4, 8, 16 ────────────────────────────────────────
    if [ -x "$OPENMP" ]; then
        for t in 4 8 16; do
            comp="$WORK_DIR/openmp_t${t}_${fname}.lz77"
            dec="$WORK_DIR/openmp_t${t}_${fname}.out"
            run_test "openmp" "threads=$t" "$input_file" \
                "env OMP_NUM_THREADS=$t \"$OPENMP\" \"$input_file\" \"$comp\"" \
                "env OMP_NUM_THREADS=1  \"$OPENMP\" \"$comp\" \"$dec\" --decompress" \
                "$comp" "$dec"
        done
    fi

    # ── 3. POSIX Threads — threads: 4, 8, 16 ─────────────────────────────────
    if [ -x "$PTHREAD" ]; then
        for t in 4 8 16; do
            comp="$WORK_DIR/pthread_t${t}_${fname}.lz77"
            dec="$WORK_DIR/pthread_t${t}_${fname}.out"
            run_test "pthread" "threads=$t" "$input_file" \
                "\"$PTHREAD\" \"$input_file\" \"$comp\" $t" \
                "\"$PTHREAD\" \"$comp\" \"$dec\" --decompress" \
                "$comp" "$dec"
        done
    fi

    # ── 4. MPI — processes: 2, 4, 8 ──────────────────────────────────────────
    if [ -x "$MPI_BIN" ] && $HAS_MPI; then
        for p in 2 4 8; do
            comp="$WORK_DIR/mpi_p${p}_${fname}.lz77"
            dec="$WORK_DIR/mpi_p${p}_${fname}.out"
            run_test "mpi" "procs=$p" "$input_file" \
                "mpirun $MPI_EXTRA -np $p \"$MPI_BIN\" \"$input_file\" \"$comp\"" \
                "\"$SERIAL\" \"$comp\" \"$dec\" --decompress" \
                "$comp" "$dec"
        done
    fi

    # ── 5. MPI+OpenMP Hybrid — (procs × threads): 2×4, 4×2, 4×4 ─────────────
    if [ -x "$MPI_OMP" ] && $HAS_MPI; then
        for combo in "2:4" "4:2" "4:4"; do
            p="${combo%%:*}"; t="${combo##*:}"
            comp="$WORK_DIR/mpi_omp_p${p}t${t}_${fname}.lz77"
            dec="$WORK_DIR/mpi_omp_p${p}t${t}_${fname}.out"
            run_test "mpi_openmp" "procs=$p,threads=$t" "$input_file" \
                "env OMP_NUM_THREADS=$t mpirun $MPI_EXTRA -np $p \"$MPI_OMP\" \"$input_file\" \"$comp\"" \
                "\"$SERIAL\" \"$comp\" \"$dec\" --decompress" \
                "$comp" "$dec"
        done
    fi

    # ── 6. CUDA — block sizes: 128, 256, 512 ─────────────────────────────────
    if [ -x "$CUDA_BIN" ] && $HAS_CUDA; then
        for bs in 128 256 512; do
            comp="$WORK_DIR/cuda_bs${bs}_${fname}.lz77"
            dec="$WORK_DIR/cuda_bs${bs}_${fname}.out"
            run_test "cuda" "block_size=$bs" "$input_file" \
                "\"$CUDA_BIN\" \"$input_file\" \"$comp\" --block-size $bs" \
                "\"$CUDA_BIN\" \"$comp\" \"$dec\" --decompress" \
                "$comp" "$dec"
        done
    fi

    # ── 7. MPI+CUDA Hybrid — processes: 1, 2, 4 ──────────────────────────────
    if [ -x "$MPI_CUDA" ] && $HAS_MPI && $HAS_CUDA; then
        for p in 1 2 4; do
            comp="$WORK_DIR/mpi_cuda_p${p}_${fname}.lz77"
            dec="$WORK_DIR/mpi_cuda_p${p}_${fname}.out"
            run_test "mpi_cuda" "procs=$p,block_size=256" "$input_file" \
                "mpirun $MPI_EXTRA -np $p \"$MPI_CUDA\" \"$input_file\" \"$comp\"" \
                "\"$SERIAL\" \"$comp\" \"$dec\" --decompress" \
                "$comp" "$dec"
        done
    fi

    echo ""
done

# Remove work dir if empty
rmdir "$WORK_DIR" 2>/dev/null || true

TOTAL_END=$(now_ns)
TOTAL_TIME=$(elapsed_s "$TOTAL_START" "$TOTAL_END")

log "Benchmark complete in ${TOTAL_TIME}s"
log "Results: $CSV_FILE"
