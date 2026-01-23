#!/usr/bin/env python3
"""
Pickle File Comparison Tool for CASIA-B Preprocessed Data
Compares two pickle files deeply, checking numpy arrays inside.
"""

import os
import pickle
import numpy as np
from pathlib import Path
import hashlib
import sys


def get_file_hash(filepath):
    """Calculate MD5 hash of file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compare_single_pkl(file1, file2, verbose=False):
    """
    Compare two pickle files deeply.
    
    Returns:
        dict: Comparison results with status and details
    """
    result = {
        'identical': False,
        'file1': str(file1),
        'file2': str(file2),
        'issues': []
    }
    
    # Check file existence
    if not os.path.exists(file1):
        result['issues'].append(f"File 1 does not exist: {file1}")
        return result
    if not os.path.exists(file2):
        result['issues'].append(f"File 2 does not exist: {file2}")
        return result
    
    # Compare file sizes
    size1 = os.path.getsize(file1)
    size2 = os.path.getsize(file2)
    if size1 != size2:
        result['issues'].append(f"File sizes differ: {size1} vs {size2} bytes")
        if verbose:
            print(f"  File sizes: {size1} vs {size2}")
    
    # Quick binary comparison using hash
    hash1 = get_file_hash(file1)
    hash2 = get_file_hash(file2)
    
    if hash1 == hash2:
        result['identical'] = True
        if verbose:
            print(f"Y - Files are byte-for-byte identical (MD5: {hash1})")
        return result
    
    if verbose:
        print(f"  MD5 hashes differ: {hash1} vs {hash2}")
        print(f"  Loading pickle contents for deep comparison...")
    
    # Load pickle contents
    try:
        with open(file1, 'rb') as f:
            data1 = pickle.load(f)
        with open(file2, 'rb') as f:
            data2 = pickle.load(f)
    except Exception as e:
        result['issues'].append(f"Error loading pickle files: {e}")
        return result
    
    # Check types
    if type(data1) != type(data2):
        result['issues'].append(f"Data types differ: {type(data1)} vs {type(data2)}")
        return result
    
    # For numpy arrays (expected case for CASIA-B)
    if isinstance(data1, np.ndarray):
        # Check shapes
        if data1.shape != data2.shape:
            result['issues'].append(f"Array shapes differ: {data1.shape} vs {data2.shape}")
            return result
        
        # Check dtypes
        if data1.dtype != data2.dtype:
            result['issues'].append(f"Array dtypes differ: {data1.dtype} vs {data2.dtype}")
            return result
        
        # Check exact equality
        if np.array_equal(data1, data2):
            result['identical'] = True
            if verbose:
                print(f"Y - Arrays are identical (shape: {data1.shape}, dtype: {data1.dtype})")
            return result
        
        # Check for numerical differences
        diff_count = np.sum(data1 != data2)
        max_diff = np.max(np.abs(data1.astype(float) - data2.astype(float)))
        
        result['issues'].append(
            f"Arrays differ: {diff_count}/{data1.size} elements "
            f"({100*diff_count/data1.size:.2f}%), max diff: {max_diff}"
        )
        
        if verbose:
            print(f"  Different elements: {diff_count}/{data1.size}")
            print(f"  Maximum difference: {max_diff}")
            
            # Show some examples of differences
            diff_mask = data1 != data2
            diff_indices = np.where(diff_mask)
            if len(diff_indices[0]) > 0:
                print(f"  First 5 differences:")
                for i in range(min(5, len(diff_indices[0]))):
                    idx = tuple(d[i] for d in diff_indices)
                    print(f"    [{idx}]: {data1[idx]} vs {data2[idx]}")
    else:
        # For non-numpy data
        if data1 == data2:
            result['identical'] = True
            if verbose:
                print(f"Y - Data is identical")
        else:
            result['issues'].append("Data differs (non-numpy comparison)")
    
    return result


def compare_directories(dir1, dir2, pattern="**/*.pkl", verbose=False):
    """
    Compare all matching pickle files in two directories.
    
    Args:
        dir1: First directory path
        dir2: Second directory path  
        pattern: Glob pattern for files to compare
        verbose: Print detailed comparison info
    """
    dir1 = Path(dir1)
    dir2 = Path(dir2)
    
    # Find all pickle files in dir1
    pkl_files1 = sorted(dir1.glob(pattern))
    
    print(f"Found {len(pkl_files1)} pickle files in {dir1}")
    print(f"Comparing with corresponding files in {dir2}")
    print("="*80)
    
    results = {
        'total': 0,
        'identical': 0,
        'different': 0,
        'missing': 0,
        'errors': 0
    }
    
    different_files = []
    
    for pkl1 in pkl_files1:
        # Get relative path
        rel_path = pkl1.relative_to(dir1)
        pkl2 = dir2 / rel_path
        
        results['total'] += 1
        
        if verbose:
            print(f"\n[{results['total']}/{len(pkl_files1)}] Comparing: {rel_path}")
        
        if not pkl2.exists():
            results['missing'] += 1
            if not verbose:
                print(f"X - Missing in dir2: {rel_path}")
            else:
                print(f"  X - File missing in second directory")
            continue
        
        # Compare files
        comparison = compare_single_pkl(pkl1, pkl2, verbose=verbose)
        
        if comparison['identical']:
            results['identical'] += 1
            if not verbose:
                print(f"Y - {rel_path}")
        else:
            results['different'] += 1
            different_files.append((rel_path, comparison['issues']))
            if not verbose:
                print(f"X - DIFFER: {rel_path}")
                for issue in comparison['issues']:
                    print(f"    {issue}")
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"Total files compared: {results['total']}")
    print(f"Identical:            {results['identical']} ({100*results['identical']/results['total']:.1f}%)")
    print(f"Different:            {results['different']} ({100*results['different']/results['total']:.1f}%)")
    print(f"Missing in dir2:      {results['missing']}")
    
    if different_files:
        print(f"\n{len(different_files)} files differ:")
        for filepath, issues in different_files[:10]:  # Show first 10
            print(f"  - {filepath}")
            for issue in issues:
                print(f"      {issue}")
        if len(different_files) > 10:
            print(f"  ... and {len(different_files) - 10} more")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare pickle files from CASIA-B preprocessing')
    parser.add_argument('path1', help='First pickle file or directory')
    parser.add_argument('path2', help='Second pickle file or directory')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-r', '--recursive', action='store_true', help='Compare directories recursively')
    
    args = parser.parse_args()
    
    path1 = Path(args.path1)
    path2 = Path(args.path2)
    
    if path1.is_file() and path2.is_file():
        # Compare single files
        print(f"Comparing files:")
        print(f"  File 1: {path1}")
        print(f"  File 2: {path2}")
        print("="*80)
        
        result = compare_single_pkl(path1, path2, verbose=True)
        
        if result['identical']:
            print("\nY - Files are IDENTICAL")
            sys.exit(0)
        else:
            print("\nX - Files are DIFFERENT")
            for issue in result['issues']:
                print(f"  - {issue}")
            sys.exit(1)
    
    elif path1.is_dir() and path2.is_dir():
        # Compare directories
        if not args.recursive:
            print("Note: Use -r/--recursive flag to compare directories")
            sys.exit(1)
        
        results = compare_directories(path1, path2, verbose=args.verbose)
        
        if results['different'] == 0 and results['missing'] == 0:
            print("\nY - All files are IDENTICAL")
            sys.exit(0)
        else:
            print("\nX - Some files differ")
            sys.exit(1)
    else:
        print("Error: Both paths must be either files or directories")
        sys.exit(1)


if __name__ == '__main__':
    main()