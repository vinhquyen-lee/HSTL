#!/usr/bin/env python3
"""
Training Log Processor
Processes HSTL training log files into organized summaries.

Usage: python process_logs.py <absolute_path_to_logs_folder>
"""

import os
import sys
import re
from pathlib import Path
from datetime import datetime

def parse_restore_hint(lines):
    """Return restore_hint value from config lines; fallback to 'N/A'."""
    for line in lines:
        match = re.search(r'\'restore_hint\'\s*[:=]\s*(\d+)', line)
        if match:
            return match.group(1)
    return 'N/A'

def extract_timestamp_from_filename(filename):
    """Extract datetime from filename like '2025-12-29-04-21-51.txt'"""
    match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.txt$', filename)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y-%m-%d-%H-%M-%S')
        except ValueError:
            return None
    return None


def preprocess_log_content(lines):
    """
    Fix formatting issues where array elements are split across lines.
    Merges lines that are continuations of numpy arrays.
    """
    processed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this line contains an incomplete numpy array (ends with number or comma, no closing bracket)
        # and the next line starts with spaces followed by numbers and ends with ]
        if i + 1 < len(lines):
            # Pattern: line ends with array opening or continuation without closing bracket
            # Next line starts with whitespace and contains array continuation
            if ('[' in line and ']' not in line.split('[')[-1]) or \
               (re.search(r'\d+\.\d+$', line.rstrip())):
                next_line = lines[i + 1]
                print("incomplete line:",next_line)
                # Check if next line is a continuation (starts with whitespace and has array closing)
                if re.match(r'^\s+[\d\.\s\]]+', next_line):
                    # Merge the lines
                    merged = line.rstrip() + ' ' + next_line.lstrip()
                    processed_lines.append(merged)
                    i += 2  # Skip next line since we merged it
                    continue

        processed_lines.append(line)
        i += 1

    return processed_lines


def split_log_content(lines):
    """
    Split log content into three parts:
    1. Metadata/Config (from start until Model Initialization Finished)
    2. Training logs (Iteration entries)
    3. Testing logs (Running test entries and results)

    Returns: (config_lines, train_lines, test_lines)
    """
    config_lines = []
    train_lines = []
    test_lines = []

    in_config = True
    in_test = False
    test_buffer = []

    for line in lines:
        # Check if we're leaving config section
        if in_config:
            config_lines.append(line)
            if 'Model Initialization Finished!' in line:
                in_config = False
            continue

        # Check for test section markers
        if 'Running test...' in line:
            in_test = True
            test_buffer = [line]
            continue

        # If in test section
        if in_test:
            test_buffer.append(line)
            # Check if test section ended (next iteration starts or file ends)
            if line.strip().startswith('[') and 'Iteration' in line:
                # Test section ended, save buffer and add this line to train
                test_lines.extend(test_buffer[:-1])  # Don't include the iteration line
                train_lines.append(test_buffer[-1])  # Add iteration to train
                test_buffer = []
                in_test = False
            elif re.match(r'^\[.*\] \[INFO\]: (NM|BG|CL):', line):
                # Still in test results, continue
                continue
            elif line.strip() == '' or (not line.strip().startswith('[')):
                # Empty or continuation line in test
                continue
            elif 'Rank-1' in line or '===' in line:
                # Test header line
                continue
        else:
            # Regular training iteration line
            if line.strip():  # Ignore empty lines
                train_lines.append(line)

    # If we ended while still in test section, add remaining buffer
    if test_buffer:
        test_lines.extend(test_buffer)

    return config_lines, train_lines, test_lines


def process_single_log_file(file_path):
    """Process a single log file and return (config, train, test) content"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Preprocess to fix formatting issues
    lines = preprocess_log_content(lines)

    # Split into three parts
    config_lines, train_lines, test_lines = split_log_content(lines)

    return config_lines, train_lines, test_lines, parse_restore_hint(config_lines)


def find_log_files(logs_dir):
    """Find all .txt log files recursively in the logs directory"""
    log_files = []

    for root, dirs, files in os.walk(logs_dir):
        for file in files:
            if file.endswith('.txt'):
                full_path = os.path.join(root, file)
                timestamp = extract_timestamp_from_filename(file)
                if timestamp:
                    log_files.append((timestamp, full_path))

    # Sort by timestamp (oldest first)
    log_files.sort(key=lambda x: x[0])

    return [path for _, path in log_files]


def process_logs(logs_dir_path):
    """Main processing function"""
    logs_dir = Path(logs_dir_path).resolve()

    if not logs_dir.exists() or not logs_dir.is_dir():
        print(f"Error: Directory does not exist: {logs_dir}")
        sys.exit(1)

    # Find all log files
    log_files = find_log_files(logs_dir)

    if not log_files:
        print(f"No log files found in: {logs_dir}")
        sys.exit(1)

    print(f"Found {len(log_files)} log file(s)")
    for i, log_file in enumerate(log_files, 1):
        print(f"  {i}. {Path(log_file).name}")

    # Create summary directory at parent level of input
    summary_dir = logs_dir.parent / 'summary'
    summary_dir.mkdir(exist_ok=True)

    print(f"\nProcessing logs...")
    print(f"Output directory: {summary_dir}")

    # Aggregate content from all files
    all_config_lines = []
    all_train_lines = []
    all_test_lines = []

    for i, log_file in enumerate(log_files, 1):
        print(f"Processing {i}/{len(log_files)}: {Path(log_file).name}")
        config, train, test, restore_hint = process_single_log_file(log_file)

        init_line = f"INIT CHECKPOINT: {restore_hint}\n"

        # Add separator between files (except for first file)
        if i > 1:
            separator = f"\n{'='*80}\n"
            separator += f"{'='*80}\n"
            separator += f"LOG FILE: {Path(log_file).name}\n"
            separator += init_line
            separator += f"{'='*80}\n"
            separator += f"{'='*80}\n\n"

            all_config_lines.append(separator)
            all_train_lines.append(separator)
            all_test_lines.append(separator)
        else:
            # First file header
            header = f"{'='*80}\n"
            header += f"LOG FILE: {Path(log_file).name}\n"
            header += init_line
            header += f"{'='*80}\n\n"
            all_config_lines.append(header)
            all_train_lines.append(header)
            all_test_lines.append(header)

        all_config_lines.extend(config)
        all_train_lines.extend(train)
        all_test_lines.extend(test)

    # Write to output files
    config_file = summary_dir / 'config.txt'
    train_file = summary_dir / 'train_log.txt'
    test_file = summary_dir / 'test_log.txt'

    print(f"\nWriting output files...")

    with open(config_file, 'w', encoding='utf-8') as f:
        f.writelines(all_config_lines)
    print(f"  ✓ {config_file.name} ({len(all_config_lines)} lines)")

    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(all_train_lines)
    print(f"  ✓ {train_file.name} ({len(all_train_lines)} lines)")

    with open(test_file, 'w', encoding='utf-8') as f:
        f.writelines(all_test_lines)
    print(f"  ✓ {test_file.name} ({len(all_test_lines)} lines)")

    print(f"\n✓ Processing complete!")
    print(f"Summary files saved to: {summary_dir}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python process_logs.py <absolute_path_to_logs_folder>")
        print("\nExample:")
        print("  python process_logs.py C:/Users/Admin/Documents/CV/biometrics/src/HSTL/logs/p3d-circleloss/output/CASIA-B/HSTL/p3d-circleloss/logs")
        sys.exit(1)

    logs_dir = sys.argv[1]
    process_logs(logs_dir)


if __name__ == '__main__':
    main()