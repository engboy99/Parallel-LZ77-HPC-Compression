#!/bin/bash

# LZ77 Compression Performance Test Script
# Tests serial and OpenMP implementations on 1MB and 10MB files

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "  LZ77 Compression Performance Testing"
echo "======================================================================"
echo ""

# Check if gcc is available
if ! command -v gcc &> /dev/null; then
    echo -e "${RED}Error: gcc not found${NC}"
    echo "Please install gcc:"
    echo "  Alpine: apk add gcc make libc-dev"
    echo "  Ubuntu: sudo apt-get install build-essential"
    exit 1
fi

# Compile serial version
echo -e "${BLUE}Compiling serial implementation...${NC}"
cd serial
gcc -O3 -o serial_lz77 serial_lz77.c
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to compile serial_lz77${NC}"
    exit 1
fi
echo -e "${GREEN}Serial compilation successful${NC}"

# Compile OpenMP version
echo -e "${BLUE}Compiling OpenMP implementation...${NC}"
cd ../openmp
gcc -O3 -fopenmp -o openmp_lz77 openmp_lz77.c
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to compile openmp_lz77${NC}"
    exit 1
fi
echo -e "${GREEN}OpenMP compilation successful${NC}"
echo ""

cd ..

# Function to extract execution time from output
extract_time() {
    output="$1"
    # Remove ANSI color codes and extract time
    cleaned=$(echo "$output" | sed 's/\x1b\[[0-9;]*m//g')
    time=$(echo "$cleaned" | grep -E 'Execution time: [0-9.]+' | sed 's/.*Execution time: \([0-9.]*\).*/\1/' | head -1)
    echo "$time"
}

# Create temp file to store results
results_file=$(mktemp)

# Run tests
echo -e "${BLUE}Running compression tests...${NC}"
echo ""

# Test each file
for file in scripts/datasets/test_files/1MB_mixed.txt \
            scripts/datasets/test_files/1MB_random.txt \
            scripts/datasets/test_files/1MB_repetitive.txt \
            scripts/datasets/test_files/10MB_mixed.txt \
            scripts/datasets/test_files/10MB_random.txt \
            scripts/datasets/test_files/10MB_repetitive.txt
do
    if [ ! -f "$file" ]; then
        echo -e "${YELLOW}Warning: $file not found, skipping${NC}"
        continue
    fi
    
    filename=$(basename "$file")
    echo -e "Testing: ${GREEN}$filename${NC}"
    
    # Test serial
    echo -n "  Serial...   "
    output=$(./serial/serial_lz77 "$file" /dev/null 2>&1)
    serial_time=$(extract_time "$output")
    if [ -n "$serial_time" ] && [ "$serial_time" != "0" ]; then
        echo -e "${GREEN}$serial_time s${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        serial_time="0"
    fi
    
    # Test OpenMP
    echo -n "  OpenMP...   "
    output=$(./openmp/openmp_lz77 "$file" /dev/null 2>&1)
    openmp_time=$(extract_time "$output")
    if [ -n "$openmp_time" ] && [ "$openmp_time" != "0" ]; then
        echo -e "${GREEN}$openmp_time s${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        openmp_time="0"
    fi
    
    # Store results
    echo "$filename,$serial_time,$openmp_time" >> "$results_file"
    
    echo ""
done

# Generate results table
echo ""
echo "======================================================================"
echo "                          RESULTS"
echo "======================================================================"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
printf "%-25s %12s %12s %12s\n" "Test File" "Serial (s)" "OpenMP (s)" "Speedup"
echo "═══════════════════════════════════════════════════════════════════════"

# Print results
echo "--- 1MB Files ---"
while IFS=, read -r filename serial openmp; do
    case "$filename" in
        1MB_*.txt)
            if [ "$serial" != "0" ] && [ "$openmp" != "0" ]; then
                speedup=$(echo "scale=2; $serial / $openmp" | bc)
                comparison=$(echo "$speedup > 2.0" | bc)
                if [ "$comparison" -eq 1 ]; then
                    mark=" ✓"
                else
                    mark=""
                fi
                name=$(echo "$filename" | sed 's/\.txt$//')
                printf "%-25s %12.4f %12.4f %11.2fx%s\n" "$name" "$serial" "$openmp" "$speedup" "$mark"
            fi
            ;;
    esac
done < "$results_file"

echo ""
echo "--- 10MB Files ---"
while IFS=, read -r filename serial openmp; do
    case "$filename" in
        10MB_*.txt)
            if [ "$serial" != "0" ] && [ "$openmp" != "0" ]; then
                speedup=$(echo "scale=2; $serial / $openmp" | bc)
                comparison=$(echo "$speedup > 2.0" | bc)
                if [ "$comparison" -eq 1 ]; then
                    mark=" ✓"
                else
                    mark=""
                fi
                name=$(echo "$filename" | sed 's/\.txt$//')
                printf "%-25s %12.4f %12.4f %11.2fx%s\n" "$name" "$serial" "$openmp" "$speedup" "$mark"
            fi
            ;;
    esac
done < "$results_file"

echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Save to CSV
csv_file="test_results_$(date +%Y%m%d_%H%M%S).csv"
echo "Test File,Serial (s),OpenMP (s),Speedup" > "$csv_file"

echo "--- 1MB Files ---,,,," >> "$csv_file"
while IFS=, read -r filename serial openmp; do
    case "$filename" in
        1MB_*.txt)
            if [ "$serial" != "0" ] && [ "$openmp" != "0" ]; then
                speedup=$(echo "scale=2; $serial / $openmp" | bc)
                name=$(echo "$filename" | sed 's/\.txt$//')
                echo "$name,$serial,$openmp,${speedup}x" >> "$csv_file"
            fi
            ;;
    esac
done < "$results_file"

echo "--- 10MB Files ---,,,," >> "$csv_file"
while IFS=, read -r filename serial openmp; do
    case "$filename" in
        10MB_*.txt)
            if [ "$serial" != "0" ] && [ "$openmp" != "0" ]; then
                speedup=$(echo "scale=2; $serial / $openmp" | bc)
                name=$(echo "$filename" | sed 's/\.txt$//')
                echo "$name,$serial,$openmp,${speedup}x" >> "$csv_file"
            fi
            ;;
    esac
done < "$results_file"

# Cleanup
rm -f "$results_file"

echo -e "${GREEN}Results saved to: $csv_file${NC}"
echo ""
