#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPZ File Visualization Script
Usage:
    python vis_npz.py path/to/file.npz                          # Show NPZ structure and basic visualization
    python vis_npz.py path/to/file.npz --mode inspect          # Show file structure and statistics only
    python vis_npz.py path/to/file.npz --mode plot             # Plot all arrays
    python vis_npz.py path/to/file.npz --mode compare --keys key1 key2  # Compare two arrays
    python vis_npz.py path/to/file.npz --mode heatmap --key array_name   # Generate heatmap
"""

import argparse
import os
import sys
from typing import Dict, Any, List
import numpy as np
import matplotlib
# Use non-interactive backend for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm


def load_npz(file_path: str) -> Dict[str, Any]:
    """Load NPZ file"""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    try:
        data = np.load(file_path, allow_pickle=True)
        return dict(data)
    except Exception as e:
        print(f"Error: Failed to load NPZ: {e}")
        sys.exit(1)


def inspect_npz(data: Dict[str, Any]) -> None:
    """Print detailed NPZ file information"""
    print("\n" + "=" * 80)
    print("NPZ File Structure and Statistics")
    print("=" * 80)
    
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"\n[{key}]")
            print(f"   Shape: {value.shape}")
            print(f"   Data type: {value.dtype}")
            print(f"   Memory: {value.nbytes / 1024:.2f} KB")
            
            if value.size > 0:
                if np.issubdtype(value.dtype, np.number):
                    print(f"   Range: [{value.min():.6f}, {value.max():.6f}]")
                    print(f"   Mean: {value.mean():.6f}, Std: {value.std():.6f}")
                    
                    # Display first few elements
                    if value.ndim == 0:
                        # 0-dimensional array (scalar)
                        print(f"   Value: {value.item()}")
                    elif value.ndim == 1:
                        print(f"   First 5 values: {value[:5]}")
                    else:
                        print(f"   First 2 rows: \n{value[:2]}")
        else:
            # Handle non-array types
            print(f"\n[{key}]")
            print(f"   Type: {type(value)}")
            print(f"   Value: {value}")


def plot_arrays(data: Dict[str, Any], output_path: str = None) -> None:
    """Plot all arrays visualization"""
    numeric_arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number)}
    
    if not numeric_arrays:
        print("Error: No numeric arrays found")
        return
    
    num_plots = len(numeric_arrays)
    cols = min(3, num_plots)
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    
    for idx, (key, arr) in enumerate(numeric_arrays.items()):
        ax = axes[idx]
        
        # Choose appropriate plot type based on array dimension
        if arr.ndim == 0:
            # 0-dimensional array (scalar)
            ax.text(0.5, 0.5, f"Scalar: {arr.item():.6f}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        elif arr.ndim == 1:
            # 1D array: line plot
            ax.plot(arr, linewidth=1.5, marker='o', markersize=3)
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            # Limit y-axis for infer_times_ms
            if "infer_times_ms" in key.lower() or "infer_time" in key.lower():
                ax.set_ylim(top=300, bottom=200)
                ax.set_ylabel("Time (ms)")
        elif arr.ndim == 2:
            if arr.shape[0] > arr.shape[1]:
                # Time series data (T, D): multiple lines
                for i in range(min(arr.shape[1], 10)):  # Show at most 10 lines
                    ax.plot(arr[:, i], linewidth=1, alpha=0.7, label=f"Dim {i}")
                if arr.shape[1] > 1:
                    ax.legend(fontsize=8, loc='best')
                ax.set_xlabel("Time Step")
                ax.set_ylabel("Value")
            else:
                # Matrix: heatmap
                im = ax.imshow(arr, aspect='auto', cmap='viridis', interpolation='nearest')
                ax.set_xlabel("Col")
                ax.set_ylabel("Row")
                fig.colorbar(im, ax=ax)
        else:
            # High-dimensional array: show first slice
            if arr.ndim == 3:
                im = ax.imshow(arr[0], cmap='viridis', interpolation='nearest')
                ax.set_title(f"{key} (Slice[0])")
                fig.colorbar(im, ax=ax)
            else:
                ax.text(0.5, 0.5, f"Dimension too high\n{arr.shape}", 
                       ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title(f"{key}\n{arr.shape}")
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(len(numeric_arrays), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Success: Visualization saved to {output_path}")
        plt.close()
    else:
        print("Note: No output path specified. Use --output to save the visualization.")
        print("Available arrays to visualize:")
        for key in numeric_arrays.keys():
            print(f"  - {key}: {numeric_arrays[key].shape}")


def compare_arrays(data: Dict[str, Any], keys: List[str], output_path: str = None) -> None:
    """Compare two or more arrays"""
    if len(keys) < 2:
        print("Error: Specify at least 2 arrays for comparison")
        return
    
    for key in keys:
        if key not in data:
            print(f"Error: Key not found: {key}")
            return
    
    arrays = [data[key] for key in keys]
    
    # Check if shapes match
    shapes = [arr.shape for arr in arrays]
    if len(set(shapes)) > 1:
        print(f"Warning: Array shapes mismatch: {dict(zip(keys, shapes))}")
    
    # Create comparison plot
    if all(arr.ndim >= 1 for arr in arrays):
        fig, axes = plt.subplots(len(keys), 1, figsize=(12, 4 * len(keys)))
        axes = np.atleast_1d(axes)
        
        for ax, key, arr in zip(axes, keys, arrays):
            if arr.ndim == 0:
                # 0-dimensional array (scalar)
                ax.text(0.5, 0.5, f"Scalar: {arr.item():.6f}", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
            elif arr.ndim == 1:
                ax.plot(arr, linewidth=2, marker='o', markersize=3)
                ax.set_ylabel(key)
                ax.grid(True, alpha=0.3)
                # Limit y-axis for infer_times_ms
                if "infer_times_ms" in key.lower() or "infer_time" in key.lower():
                    ax.set_ylim(top=300, bottom=200)
            elif arr.ndim == 2:
                im = ax.imshow(arr, aspect='auto', cmap='viridis', interpolation='nearest')
                ax.set_ylabel(f"{key} ({arr.shape})")
                fig.colorbar(im, ax=ax)
            
            ax.set_title(f"{key} - Shape: {arr.shape}")
        
        # If two arrays with same shape, add difference analysis
        if len(keys) == 2 and arrays[0].shape == arrays[1].shape and arrays[0].ndim >= 1:
            diff = np.abs(arrays[0] - arrays[1])
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            
            if arrays[0].ndim == 1:
                ax2.plot(diff, linewidth=2, color='red', label='Absolute diff')
                ax2.fill_between(range(len(diff)), diff, alpha=0.3, color='red')
                ax2.set_ylabel("Absolute difference")
                ax2.set_xlabel("Index")
            else:
                im = ax2.imshow(diff, aspect='auto', cmap='hot', interpolation='nearest')
                ax2.set_title("Absolute difference heatmap")
                fig2.colorbar(im, ax=ax2)
            
            ax2.set_title(f"Difference analysis: |{keys[0]} - {keys[1]}|\nMax diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Success: Comparison plot saved to {output_path}")
            plt.close(fig)
            plt.close(fig2) if 'fig2' in locals() else None
        else:
            print("Note: No output path specified. Use --output to save the comparison.")
            plt.close()


def plot_heatmap(data: Dict[str, Any], key: str, output_path: str = None) -> None:
    """Plot heatmap"""
    if key not in data:
        print(f"Error: Key not found: {key}")
        return
    
    arr = data[key]
    
    if not isinstance(arr, np.ndarray) or not np.issubdtype(arr.dtype, np.number):
        print(f"Error: {key} is not a numeric array")
        return
    
    # Handle 0-dimensional array (scalar)
    if arr.ndim == 0:
        print(f"Warning: {key} is a scalar value: {arr.item():.6f}, cannot plot heatmap")
        return
    
    if arr.ndim == 1:
        # Convert 1D array to 2D
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        # For high-dimensional array, take first slice
        arr = arr[0] if arr.shape[0] > 1 else arr.reshape(arr.shape[1:])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(arr, aspect='auto', cmap='viridis', interpolation='bilinear')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Value")
    
    # Labels
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(f"{key} - Heatmap (Shape: {arr.shape})\nMin: {arr.min():.6f}, Max: {arr.max():.6f}")
    
    # If matrix is small enough, show values
    if arr.shape[0] <= 20 and arr.shape[1] <= 20:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                text = ax.text(j, i, f'{arr[i, j]:.2f}',
                             ha="center", va="center", color="w", fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Success: Heatmap saved to {output_path}")
        plt.close()
    else:
        print("Note: No output path specified. Use --output to save the heatmap.")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="NPZ File Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show NPZ file info and basic visualization
  python vis_npz.py data.npz
  
  # Check file structure only
  python vis_npz.py data.npz --mode inspect
  
  # Plot all arrays
  python vis_npz.py data.npz --mode plot --output result.png
  
  # Compare two arrays
  python vis_npz.py data.npz --mode compare --keys gt_actions pred_actions --output compare.png
  
  # Plot heatmap
  python vis_npz.py data.npz --mode heatmap --key gt_actions --output heatmap.png
        """
    )
    
    parser.add_argument("input", help="Path to NPZ file")
    parser.add_argument("--mode", choices=["inspect", "plot", "compare", "heatmap"], default="inspect",
                       help="Visualization mode")
    parser.add_argument("--keys", nargs="+", help="Keys to compare (use with mode=compare)")
    parser.add_argument("--key", help="Key for heatmap (use with mode=heatmap)")
    parser.add_argument("--output", help="Output file path (save image if specified, else display)")
    
    args = parser.parse_args()
    
    print(f"Loading NPZ file: {args.input}")
    data = load_npz(args.input)
    
    if args.mode == "inspect":
        inspect_npz(data)
    elif args.mode == "plot":
        plot_arrays(data, args.output)
    elif args.mode == "compare":
        if not args.keys:
            print("Error: --keys argument required for compare mode")
            sys.exit(1)
        compare_arrays(data, args.keys, args.output)
    elif args.mode == "heatmap":
        if not args.key:
            print("Error: --key argument required for heatmap mode")
            sys.exit(1)
        plot_heatmap(data, args.key, args.output)


if __name__ == "__main__":
    main()
