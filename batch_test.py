#!/usr/bin/env python3
"""
Batch Testing Script for HSTL Models
Tests multiple checkpoints and saves results in organized format.
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
import re


def load_config(config_path='batch_test_config.yaml'):
    """Load batch testing configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_test_output(output):
    """
    Extract test results from command output.
    Preserves exact format from log file.
    """
    results = {
        'rank1_include': {},
        'rank1_exclude': {},
        'rank1_angles': {}
    }
    
    lines = output.split('\n')
    
    for i, line in enumerate(lines):
        # Parse Rank-1 (Include identical-view cases)
        if 'Rank-1 (Include identical-view cases)' in line:
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                match = re.search(r'NM:\s+([\d.]+),\s+BG:\s+([\d.]+),\s+CL:\s+([\d.]+)', next_line)
                if match:
                    results['rank1_include'] = {
                        'NM': float(match.group(1)),
                        'BG': float(match.group(2)),
                        'CL': float(match.group(3))
                    }
        
        # Parse Rank-1 (Exclude identical-view cases)
        if 'Rank-1 (Exclude identical-view cases)' in line:
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                match = re.search(r'NM:\s+([\d.]+),\s+BG:\s+([\d.]+),\s+CL:\s+([\d.]+)', next_line)
                if match:
                    results['rank1_exclude'] = {
                        'NM': float(match.group(1)),
                        'BG': float(match.group(2)),
                        'CL': float(match.group(3))
                    }
        
        # Parse angle-wise results
        if 'Rank-1 of each angle' in line:
            for j in range(i + 1, min(i + 10, len(lines))):
                if 'NM:' in lines[j] and '[' in lines[j]:
                    results['rank1_angles']['NM'] = lines[j].split('NM:')[1].strip()
                if 'BG:' in lines[j] and '[' in lines[j]:
                    results['rank1_angles']['BG'] = lines[j].split('BG:')[1].strip()
                if 'CL:' in lines[j] and '[' in lines[j]:
                    results['rank1_angles']['CL'] = lines[j].split('CL:')[1].strip()
    
    return results


def run_single_test(model_name, iteration, checkpoint_path, dataset_root, dataset_partition, output_dir):
    """Run testing for a single checkpoint."""
    
    print(f"\n{'='*80}")
    print(f"Testing: {model_name} - Iteration {iteration}")
    print(f"{'='*80}")
    
    # Construct checkpoint filename
    checkpoint_file = f"{model_name}-{iteration:05d}.pt"
    full_checkpoint_path = os.path.join(checkpoint_path, checkpoint_file)
    
    # Check if checkpoint exists
    if not os.path.exists(full_checkpoint_path):
        print(f"WARNING: Checkpoint not found: {full_checkpoint_path}")
        return None
    
    # Create temporary config for this test
    temp_config = f"temp_test_{model_name}_{iteration}.yaml"
    
    config_content = f"""data_cfg:
  dataset_name: CASIA-B
  dataset_root: {dataset_root}
  dataset_partition: {dataset_partition}
  num_workers: 4
  remove_no_gallery: false
  test_dataset_name: CASIA-B

evaluator_cfg:
  enable_distributed: false
  enable_float16: false
  restore_ckpt_strict: true
  restore_hint: {iteration}
  save_name: {model_name}
  sampler:
    batch_size: 1
    sample_type: all_ordered
    type: InferenceSampler

model_cfg:
  model: HSTL
  channels: [32, 64, 128]
  class_num: 74
"""
    
    with open(temp_config, 'w') as f:
        f.write(config_content)
    
    # Run test
    cmd = [
        sys.executable, 'lib/main.py',
        '--cfgs', temp_config,
        '--phase', 'test',
        '--iter', str(iteration)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        output = result.stdout + result.stderr
        
        # Parse results
        results = parse_test_output(output)
        
        # Save detailed log
        log_dir = os.path.join(output_dir, 'logs', model_name)
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'{model_name}_{iteration}.log')
        with open(log_file, 'w') as f:
            f.write(output)
        
        print(f"Test completed. Log saved to: {log_file}")
        
        # Clean up temp config
        os.remove(temp_config)
        
        return results
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Test timed out after 600 seconds")
        os.remove(temp_config)
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        if os.path.exists(temp_config):
            os.remove(temp_config)
        return None


def format_results_table(all_results):
    """Format results as a nice table."""
    
    lines = []
    lines.append("="*100)
    lines.append("BATCH TESTING RESULTS SUMMARY")
    lines.append("="*100)
    lines.append("")
    
    for model_name, iterations_results in all_results.items():
        lines.append(f"\nModel: {model_name}")
        lines.append("-"*100)
        lines.append(f"{'Iteration':<12} {'NM (Incl)':<12} {'BG (Incl)':<12} {'CL (Incl)':<12} {'NM (Excl)':<12} {'BG (Excl)':<12} {'CL (Excl)':<12}")
        lines.append("-"*100)
        
        for iteration, results in sorted(iterations_results.items()):
            if results is None:
                lines.append(f"{iteration:<12} {'FAILED':<12}")
                continue
            
            nm_incl = results['rank1_include'].get('NM', 0.0)
            bg_incl = results['rank1_include'].get('BG', 0.0)
            cl_incl = results['rank1_include'].get('CL', 0.0)
            nm_excl = results['rank1_exclude'].get('NM', 0.0)
            bg_excl = results['rank1_exclude'].get('BG', 0.0)
            cl_excl = results['rank1_exclude'].get('CL', 0.0)
            
            lines.append(f"{iteration:<12} {nm_incl:<12.1f} {bg_incl:<12.1f} {cl_incl:<12.1f} {nm_excl:<12.1f} {bg_excl:<12.1f} {cl_excl:<12.1f}")
    
    lines.append("")
    lines.append("="*100)
    
    return "\n".join(lines)


def save_detailed_results(all_results, output_dir):
    """Save detailed results with angle-wise accuracy."""
    
    detail_file = os.path.join(output_dir, 'detailed_results.txt')
    
    with open(detail_file, 'w') as f:
        f.write("DETAILED BATCH TESTING RESULTS\n")
        f.write("="*100 + "\n\n")
        
        for model_name, iterations_results in all_results.items():
            f.write(f"\n{'='*100}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"{'='*100}\n\n")
            
            for iteration, results in sorted(iterations_results.items()):
                if results is None:
                    f.write(f"Iteration {iteration}: FAILED\n\n")
                    continue
                
                f.write(f"Iteration {iteration}:\n")
                f.write(f"-"*80 + "\n")
                
                # Rank-1 Include
                f.write(f"===Rank-1 (Include identical-view cases)===\n")
                if results['rank1_include']:
                    f.write(f"NM: {results['rank1_include']['NM']:.1f},\t")
                    f.write(f"BG: {results['rank1_include']['BG']:.1f},\t")
                    f.write(f"CL: {results['rank1_include']['CL']:.1f}\n")
                
                # Rank-1 Exclude
                f.write(f"===Rank-1 (Exclude identical-view cases)===\n")
                if results['rank1_exclude']:
                    f.write(f"NM: {results['rank1_exclude']['NM']:.1f},\t")
                    f.write(f"BG: {results['rank1_exclude']['BG']:.1f},\t")
                    f.write(f"CL: {results['rank1_exclude']['CL']:.1f}\n")
                
                # Angle-wise
                f.write(f"===Rank-1 of each angle (Exclude identical-view cases)===\n")
                if results['rank1_angles']:
                    if 'NM' in results['rank1_angles']:
                        f.write(f"NM: {results['rank1_angles']['NM']}\n")
                    if 'BG' in results['rank1_angles']:
                        f.write(f"BG: {results['rank1_angles']['BG']}\n")
                    if 'CL' in results['rank1_angles']:
                        f.write(f"CL: {results['rank1_angles']['CL']}\n")
                
                f.write(f"\n")
    
    print(f"\nDetailed results saved to: {detail_file}")


def main():
    """Main batch testing function."""
    
    print("="*100)
    print("HSTL BATCH TESTING TOOL")
    print("="*100)
    
    # Load configuration
    config = load_config()
    
    # Create output directory
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Store all results
    all_results = {}
    
    # Iterate through all models and iterations
    for model_name, model_config in config['models'].items():
        print(f"\n\nProcessing model: {model_name}")
        print(f"Checkpoint path: {model_config['checkpoint_path']}")
        
        all_results[model_name] = {}
        
        for iteration in model_config['iterations']:
            results = run_single_test(
                model_name=model_config['save_name'],
                iteration=iteration,
                checkpoint_path=model_config['checkpoint_path'],
                dataset_root=config['dataset_root'],
                dataset_partition=config['dataset_partition'],
                output_dir=output_dir
            )
            
            all_results[model_name][iteration] = results
    
    # Generate summary
    print("\n\n")
    summary = format_results_table(all_results)
    print(summary)
    
    # Save summary
    summary_file = os.path.join(output_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(summary)
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Save detailed results
    save_detailed_results(all_results, output_dir)
    
    print(f"\n\nAll results saved to: {output_dir}")
    print("="*100)


if __name__ == '__main__':
    main()

