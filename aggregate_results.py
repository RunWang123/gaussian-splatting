#!/usr/bin/env python3
"""
Aggregate test results from all scene_case directories (Vanilla Gaussian Splatting).

This script processes RGB metrics (PSNR, SSIM, LPIPS) from vanilla 3DGS evaluation.
It computes:
- Per-scene metrics (averaged across cases)
- Overall metrics (averaged across all scenes and cases)
- Statistics (mean, std, min, max)

Expected directory structure:
  output_dir/
    scene_name_case0/
      results.json          # Contains RGB metrics
    scene_name_case1/
      results.json
    ...

Usage:
  python aggregate_results.py --results_dir /path/to/output_dir
  python aggregate_results.py --results_dir /path/to/output_dir --output results_summary.json
"""

import json
import os
import sys
from pathlib import Path
import argparse
import numpy as np
from collections import defaultdict

def load_results_json(json_path):
    """Load results.json file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {json_path}: {e}")
        return None

def extract_scene_and_case(dirname):
    """
    Extract scene name and case number from directory name.
    Example: 'scene0686_01_case5' -> ('scene0686_01', 5)
    """
    parts = dirname.split('_case')
    if len(parts) == 2:
        scene_name = parts[0]
        try:
            case_id = int(parts[1])
            return scene_name, case_id
        except ValueError:
            return None, None
    return None, None

def collect_all_results(results_dir):
    """
    Collect all results from scene_case directories.
    
    Returns:
        dict: {scene_name: {case_id: {
            'ours_7000': {'PSNR': ..., 'SSIM': ..., 'LPIPS': ...},
            'ours_30000': {'PSNR': ..., 'SSIM': ..., 'LPIPS': ...}
        }}}
    """
    results_dir = Path(results_dir)
    all_results = defaultdict(dict)
    
    # Find all scene_case directories
    scene_case_dirs = sorted([d for d in results_dir.iterdir() 
                             if d.is_dir() and '_case' in d.name])
    
    print(f"Found {len(scene_case_dirs)} scene_case directories")
    print(f"Collecting results...\n")
    
    missing_count = 0
    
    for scene_case_dir in scene_case_dirs:
        scene_name, case_id = extract_scene_and_case(scene_case_dir.name)
        
        if scene_name is None:
            continue
        
        # Load results.json
        results_json_path = scene_case_dir / "results.json"
        if results_json_path.exists():
            results = load_results_json(results_json_path)
            if results is not None:
                all_results[scene_name][case_id] = results
        else:
            missing_count += 1
            print(f"⚠️  Missing: {scene_case_dir.name}/results.json")
    
    if missing_count > 0:
        print(f"\n⚠️  Warning: {missing_count} cases missing results.json")
    
    return dict(all_results)

def compute_scene_statistics(scene_results):
    """
    Compute statistics for a single scene across all cases.
    
    Args:
        scene_results: dict {case_id: {
            'ours_7000': {'PSNR': ..., 'SSIM': ..., 'LPIPS': ...},
            'ours_30000': {'PSNR': ..., 'SSIM': ..., 'LPIPS': ...}
        }}
    
    Returns:
        dict: {iteration: {metric: stats}}
    """
    # Organize metrics by iteration
    iteration_metrics = defaultdict(lambda: defaultdict(list))
    
    for case_id, case_data in scene_results.items():
        for iteration_name, metrics in case_data.items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    iteration_metrics[iteration_name][metric_name].append(value)
    
    # Compute statistics
    stats = {}
    for iteration_name, metrics in iteration_metrics.items():
        stats[iteration_name] = {}
        for metric_name, values in metrics.items():
            values = np.array(values)
            if len(values) > 0:
                stats[iteration_name][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values),
                    'values': values.tolist()
                }
    
    return stats

def compute_overall_statistics(all_scene_stats):
    """
    Compute overall statistics across all scenes.
    
    Args:
        all_scene_stats: dict {scene_name: {
            'ours_7000': {metric: stats},
            'ours_30000': {metric: stats}
        }}
    
    Returns:
        dict: {iteration: {metric: stats}}
    """
    # Organize by iteration
    iteration_overall = defaultdict(lambda: defaultdict(list))
    
    for scene_name, scene_stats in all_scene_stats.items():
        for iteration_name, metrics in scene_stats.items():
            for metric_name, stats in metrics.items():
                # Use mean from each scene
                iteration_overall[iteration_name][metric_name].append(stats['mean'])
    
    # Compute overall statistics
    overall_stats = {}
    for iteration_name, metrics in iteration_overall.items():
        overall_stats[iteration_name] = {}
        for metric_name, values in metrics.items():
            values = np.array(values)
            if len(values) > 0:
                overall_stats[iteration_name][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
    
    return overall_stats

def print_results_table(overall_stats, title="Overall Results"):
    """Print results in a formatted table."""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}")
    
    # Sort iterations (ours_7000, ours_30000)
    iterations = sorted(overall_stats.keys())
    
    for iteration_name in iterations:
        metrics = overall_stats[iteration_name]
        print(f"\n{iteration_name}:")
        print(f"  {'Metric':<15} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
        print(f"  {'-'*63}")
        
        # Print in order: PSNR, SSIM, LPIPS
        for metric_name in ['PSNR', 'SSIM', 'LPIPS']:
            if metric_name in metrics:
                stat = metrics[metric_name]
                print(f"  {metric_name:<15} {stat['mean']:>12.4f} {stat['std']:>12.4f} "
                      f"{stat['min']:>12.4f} {stat['max']:>12.4f}")

def save_summary_json(output_path, scene_stats, overall_stats, metadata):
    """Save complete summary to JSON file."""
    summary = {
        'metadata': metadata,
        'overall_statistics': overall_stats,
        'per_scene_statistics': scene_stats
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Full summary saved to: {output_path}")

def save_csv(output_path, overall_stats):
    """Save results in CSV format."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'Metric', 'Mean', 'Std', 'Min', 'Max', 'Count'])
        
        for iteration_name, metrics in sorted(overall_stats.items()):
            for metric_name, stat in sorted(metrics.items()):
                writer.writerow([
                    iteration_name,
                    metric_name,
                    f"{stat['mean']:.6f}",
                    f"{stat['std']:.6f}",
                    f"{stat['min']:.6f}",
                    f"{stat['max']:.6f}",
                    stat['count']
                ])
    
    print(f"✅ CSV saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate test results from vanilla gaussian_splatting evaluation"
    )
    parser.add_argument('--results_dir', '-r', type=str, required=True,
                       help='Directory containing scene_case subdirectories with results')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                       help='Output directory for summary files (default: results_dir)')
    parser.add_argument('--formats', nargs='+', default=['json', 'csv'],
                       choices=['json', 'csv'],
                       help='Output formats (default: json csv)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_dir.is_dir():
        print(f"❌ Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Vanilla Gaussian Splatting Results Aggregation")
    print(f"{'='*80}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"{'='*80}\n")
    
    # Collect all results
    all_results = collect_all_results(results_dir)
    
    if not all_results:
        print("❌ No results found!")
        sys.exit(1)
    
    num_scenes = len(all_results)
    num_cases = sum(len(cases) for cases in all_results.values())
    
    print(f"\n📊 Dataset Summary:")
    print(f"  Total scenes: {num_scenes}")
    print(f"  Total cases:  {num_cases}")
    print(f"  Average cases per scene: {num_cases / num_scenes:.1f}")
    
    # Compute per-scene statistics
    print(f"\n📈 Computing per-scene statistics...")
    scene_stats = {}
    for scene_name, scene_results in all_results.items():
        scene_stats[scene_name] = compute_scene_statistics(scene_results)
    
    # Compute overall statistics
    print(f"📈 Computing overall statistics...")
    overall_stats = compute_overall_statistics(scene_stats)
    
    # Print results
    print_results_table(overall_stats, "Overall Results (RGB Metrics - Test Split)")
    
    # Save outputs
    metadata = {
        'results_directory': str(results_dir),
        'num_scenes': num_scenes,
        'num_cases': num_cases,
        'scene_list': sorted(all_results.keys()),
        'note': 'Vanilla Gaussian Splatting - RGB metrics only (PSNR, SSIM, LPIPS)'
    }
    
    if 'json' in args.formats:
        save_summary_json(
            output_dir / 'aggregate_results.json',
            scene_stats,
            overall_stats,
            metadata
        )
    
    if 'csv' in args.formats:
        save_csv(output_dir / 'aggregate_results.csv', overall_stats)
    
    print(f"\n{'='*80}")
    print(f"✅ Aggregation Complete!")
    print(f"{'='*80}\n")
    
    # Print quick reference
    print("📁 Output files:")
    if 'json' in args.formats:
        print(f"  - aggregate_results.json   (detailed per-scene + overall stats)")
    if 'csv' in args.formats:
        print(f"  - aggregate_results.csv    (overall stats in CSV format)")
    print()

if __name__ == '__main__':
    main()

