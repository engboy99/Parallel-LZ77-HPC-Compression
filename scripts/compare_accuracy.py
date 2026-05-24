#!/usr/bin/env python3
"""
Compare an original file against a decompressed file and report accuracy metrics.

Usage: python3 compare_accuracy.py <original_file> <decompressed_file>
Output (stdout, comma-separated): accuracy_percent,rmse
  - accuracy_percent: percentage of bytes in original that are reproduced correctly
  - rmse: root-mean-square error treating each byte as a value 0-255;
          missing bytes (short output) are treated as 0
"""

import sys
import math


def main():
    if len(sys.argv) < 3:
        print("0.0000,9999.0000")
        sys.exit(1)

    try:
        with open(sys.argv[1], "rb") as f:
            orig = bytearray(f.read())
        with open(sys.argv[2], "rb") as f:
            decomp = bytearray(f.read())
    except OSError as e:
        sys.stderr.write(f"compare_accuracy: {e}\n")
        print("0.0000,9999.0000")
        sys.exit(0)

    n_orig  = len(orig)
    n_decomp = len(decomp)
    n_cmp   = min(n_orig, n_decomp)

    if n_orig == 0:
        print("100.0000,0.0000")
        return

    # Use numpy when available for speed on large files (50-100 MB)
    try:
        import numpy as np
        o = np.frombuffer(bytes(orig[:n_cmp]),   dtype=np.uint8)
        d = np.frombuffer(bytes(decomp[:n_cmp]), dtype=np.uint8)

        matches = int(np.sum(o == d))
        diff    = o.astype(np.int32) - d.astype(np.int32)
        sq_sum  = float(np.sum(diff ** 2))

        # Penalty for truncated output (missing tail bytes vs original)
        if n_decomp < n_orig:
            tail = np.frombuffer(bytes(orig[n_cmp:]), dtype=np.uint8).astype(np.int32)
            sq_sum += float(np.sum(tail ** 2))

    except ImportError:
        matches = 0
        sq_sum  = 0.0
        for a, b in zip(orig[:n_cmp], decomp[:n_cmp]):
            if a == b:
                matches += 1
            sq_sum += (a - b) ** 2
        if n_decomp < n_orig:
            for byte in orig[n_cmp:]:
                sq_sum += byte * byte

    accuracy = matches / n_orig * 100.0
    rmse     = math.sqrt(sq_sum / n_orig)

    print(f"{accuracy:.4f},{rmse:.4f}")


if __name__ == "__main__":
    main()
