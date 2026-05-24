#!/usr/bin/env python3
"""
Test Data Generator for LZ77 Compression
Generates various test files with different sizes and patterns
"""

import os
import random
import string
import sys

def generate_random_text(size):
    """Generate random alphanumeric text"""
    return ''.join(random.choices(string.ascii_letters + string.digits + ' \n', k=size))

def generate_repetitive_text(size):
    """Generate text with high repetition (good compression)"""
    patterns = [
        "The quick brown fox jumps over the lazy dog. ",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ ",
        "0123456789" * 10,
        "compression test data " * 5
    ]
    result = []
    while len(''.join(result)) < size:
        result.append(random.choice(patterns))
    return ''.join(result)[:size]

def generate_mixed_text(size):
    """Generate mixed content (medium compression)"""
    chunks = []
    remaining = size
    while remaining > 0:
        chunk_size = min(random.randint(100, 1000), remaining)
        if random.random() < 0.5:
            # Repetitive chunk
            pattern = random.choice([
                "ABC" * 50,
                "Test data " * 20,
                "Compression benchmark " * 10
            ])
            chunks.append(pattern[:chunk_size])
        else:
            # Random chunk
            chunks.append(generate_random_text(chunk_size))
        remaining -= chunk_size
    return ''.join(chunks)

def generate_binary_data(size):
    """Generate random binary data"""
    return bytes([random.randint(0, 255) for _ in range(size)])

def generate_test_file(filename, size, data_type):
    """Generate a test file of specified size and type"""
    print(f"Generating {filename} ({size / (1024*1024):.2f} MB, type: {data_type})...")
    
    if data_type == 'random':
        data = generate_random_text(size).encode('utf-8')
    elif data_type == 'repetitive':
        data = generate_repetitive_text(size).encode('utf-8')
    elif data_type == 'mixed':
        data = generate_mixed_text(size).encode('utf-8')
    elif data_type == 'binary':
        data = generate_binary_data(size)
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    with open(filename, 'wb') as f:
        f.write(data[:size])
    
    actual_size = os.path.getsize(filename)
    print(f"  Created: {actual_size} bytes ({actual_size / (1024*1024):.2f} MB)")

def main():
    """Generate all test datasets"""
    
    # Create datasets directory
    os.makedirs('datasets/test_files', exist_ok=True)
    
    # Define test file configurations
    test_configs = [
        # Size (bytes), Type, Filename suffix
        (1 * 1024 * 1024, 'repetitive', '1MB_repetitive.txt'),
        (1 * 1024 * 1024, 'random', '1MB_random.txt'),
        (1 * 1024 * 1024, 'mixed', '1MB_mixed.txt'),
        
        (10 * 1024 * 1024, 'repetitive', '10MB_repetitive.txt'),
        (10 * 1024 * 1024, 'random', '10MB_random.txt'),
        (10 * 1024 * 1024, 'mixed', '10MB_mixed.txt'),
        
        (50 * 1024 * 1024, 'repetitive', '50MB_repetitive.txt'),
        (50 * 1024 * 1024, 'mixed', '50MB_mixed.txt'),
        
        (100 * 1024 * 1024, 'repetitive', '100MB_repetitive.txt'),
        (100 * 1024 * 1024, 'mixed', '100MB_mixed.txt'),
    ]
    
    print("=" * 60)
    print("LZ77 Compression Test Data Generator")
    print("=" * 60)
    print()
    
    for size, data_type, filename in test_configs:
        filepath = os.path.join('datasets', 'test_files', filename)
        generate_test_file(filepath, size, data_type)
    
    print()
    print("=" * 60)
    print("Test data generation complete!")
    print(f"Files created in: datasets/test_files/")
    print("=" * 60)

if __name__ == '__main__':
    main()
