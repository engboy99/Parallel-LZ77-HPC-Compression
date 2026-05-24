#!/usr/bin/env python3

import os
import random
import string
import multiprocessing

def generate_random_text(size):
    return ''.join(random.choices(string.ascii_letters + string.digits + ' \n', k=size))

def generate_repetitive_text(size):
    patterns = [
        "The quick brown fox jumps over the lazy dog. ",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ ",
        "0123456789" * 10,
        "compression test data " * 5
    ]

    result = []
    current_len = 0

    while current_len < size:
        p = random.choice(patterns)
        result.append(p)
        current_len += len(p)

    return ''.join(result)[:size]


def generate_mixed_text(size):
    chunks = []
    remaining = size

    while remaining > 0:
        chunk_size = min(random.randint(100, 1000), remaining)

        if random.random() < 0.5:
            pattern = random.choice([
                "ABC" * 50,
                "Test data " * 20,
                "Compression benchmark " * 10
            ])
            chunks.append(pattern[:chunk_size])
        else:
            chunks.append(generate_random_text(chunk_size))

        remaining -= chunk_size

    return ''.join(chunks)


def generate_binary_data(size):
    return bytes([random.randint(0,255) for _ in range(size)])


def generate_test_file(config):
    size, data_type, filepath = config

    print(f"Generating {filepath} ({size/(1024*1024):.2f} MB, type: {data_type})...")

    if data_type == 'random':
        data = generate_random_text(size).encode()

    elif data_type == 'repetitive':
        data = generate_repetitive_text(size).encode()

    elif data_type == 'mixed':
        data = generate_mixed_text(size).encode()

    elif data_type == 'binary':
        data = generate_binary_data(size)

    else:
        raise ValueError("Unknown type")

    with open(filepath, 'wb') as f:
        f.write(data[:size])

    actual_size = os.path.getsize(filepath)

    print(f"  Created: {actual_size} bytes ({actual_size/(1024*1024):.2f} MB)")


def main():

    os.makedirs('datasets/test_files', exist_ok=True)

    test_configs = [
        (1 * 1024 * 1024, 'repetitive', 'datasets/test_files/1MB_repetitive.txt'),
        (1 * 1024 * 1024, 'random', 'datasets/test_files/1MB_random.txt'),
        (1 * 1024 * 1024, 'mixed', 'datasets/test_files/1MB_mixed.txt'),

        (10 * 1024 * 1024, 'repetitive', 'datasets/test_files/10MB_repetitive.txt'),
        (10 * 1024 * 1024, 'random', 'datasets/test_files/10MB_random.txt'),
        (10 * 1024 * 1024, 'mixed', 'datasets/test_files/10MB_mixed.txt'),

        (50 * 1024 * 1024, 'repetitive', 'datasets/test_files/50MB_repetitive.txt'),
        (50 * 1024 * 1024, 'mixed', 'datasets/test_files/50MB_mixed.txt'),

        (100 * 1024 * 1024, 'repetitive', 'datasets/test_files/100MB_repetitive.txt'),
        (100 * 1024 * 1024, 'mixed', 'datasets/test_files/100MB_mixed.txt'),
    ]

    print("="*60)
    print("Parallel LZ77 Test Data Generator")
    print("="*60)

    cpu_count = multiprocessing.cpu_count()
    print(f"Using {cpu_count} CPU cores\n")

    with multiprocessing.Pool(cpu_count) as pool:
        pool.map(generate_test_file, test_configs)

    print("\nDone.")


if __name__ == "__main__":
    main()
