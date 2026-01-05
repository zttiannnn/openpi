"""
轨迹可视化工具

用于分析 TOPP-RA 轨迹平滑效果，对比有/无平滑的差异。

使用方法：
1. 运行推理脚本时添加 --record_trajectory 参数，会保存轨迹数据到 NPZ 文件
2. 运行本脚本进行可视化：
   python scripts/visualize_trajectory.py --input trajectory_record.npz
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Optional, List


def compute_derivatives(positions: np.ndarray, dt: float) -> tuple:
    """计算位置序列的速度、加速度、jerk"""
    if len(positions) < 2:
        zeros = np.zeros_like(positions)
        return zeros, zeros, zeros
    
    # 速度 = 一阶差分
    velocity = np.diff(positions, axis=0) / dt
    
    # 加速度 = 二阶差分
    acceleration = np.diff(velocity, axis=0) / dt if len(velocity) > 1 else np.zeros_like(velocity)
    
    # Jerk = 三阶差分
    jerk = np.diff(acceleration, axis=0) / dt if len(acceleration) > 1 else np.zeros_like(acceleration)
    
    return velocity, acceleration, jerk


def plot_single_trajectory(
    positions: np.ndarray,
    dt: float,
    title: str = "Trajectory",
    joint_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    """绘制单条轨迹的位置、速度、加速度、jerk"""
    H, D = positions.shape
    if joint_names is None:
        joint_names = [f"Joint {i}" for i in range(D)]
    
    velocity, acceleration, jerk = compute_derivatives(positions, dt)
    
    times = np.arange(H) * dt
    times_vel = times[:-1] + dt/2 if len(velocity) > 0 else times
    times_acc = times[:-2] + dt if len(acceleration) > 0 else times
    times_jerk = times[:-3] + dt*1.5 if len(jerk) > 0 else times
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=14)
    
    # 只绘制前6个关节（排除 gripper）
    n_joints = min(D, 6)
    colors = plt.cm.tab10(np.linspace(0, 1, n_joints))
    
    # Position
    for j in range(n_joints):
        axes[0].plot(times, positions[:, j], color=colors[j], label=joint_names[j], linewidth=1.5)
    axes[0].set_ylabel("Position (pulse)")
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Velocity
    if len(velocity) > 0:
        for j in range(n_joints):
            axes[1].plot(times_vel, velocity[:, j], color=colors[j], linewidth=1)
    axes[1].set_ylabel("Velocity (pulse/s)")
    axes[1].grid(True, alpha=0.3)
    
    # Acceleration
    if len(acceleration) > 0:
        for j in range(n_joints):
            axes[2].plot(times_acc, acceleration[:, j], color=colors[j], linewidth=1)
    axes[2].set_ylabel("Acceleration (pulse/s²)")
    axes[2].grid(True, alpha=0.3)
    
    # Jerk
    if len(jerk) > 0:
        for j in range(n_joints):
            axes[3].plot(times_jerk, jerk[:, j], color=colors[j], linewidth=1)
    axes[3].set_ylabel("Jerk (pulse/s³)")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_comparison(
    original: np.ndarray,
    smoothed: np.ndarray,
    dt: float,
    title: str = "Trajectory Comparison",
    save_path: Optional[str] = None,
):
    """对比原始轨迹和平滑后轨迹的速度/加速度/jerk"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)
    
    vel_orig, acc_orig, jerk_orig = compute_derivatives(original, dt)
    vel_smooth, acc_smooth, jerk_smooth = compute_derivatives(smoothed, dt)
    
    H = len(original)
    times = np.arange(H) * dt
    
    # 只看第一个关节（最容易观察抖动）
    joint_idx = 0
    
    # Velocity
    axes[0, 0].set_title("Original Velocity")
    if len(vel_orig) > 0:
        axes[0, 0].plot(times[:-1], vel_orig[:, joint_idx], 'b-', linewidth=1)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title("Smoothed Velocity")
    if len(vel_smooth) > 0:
        axes[0, 1].plot(times[:-1], vel_smooth[:, joint_idx], 'g-', linewidth=1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Acceleration
    axes[1, 0].set_title("Original Acceleration")
    if len(acc_orig) > 0:
        axes[1, 0].plot(times[:-2], acc_orig[:, joint_idx], 'b-', linewidth=1)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title("Smoothed Acceleration")
    if len(acc_smooth) > 0:
        axes[1, 1].plot(times[:-2], acc_smooth[:, joint_idx], 'g-', linewidth=1)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Jerk
    axes[2, 0].set_title("Original Jerk")
    if len(jerk_orig) > 0:
        axes[2, 0].plot(times[:-3], jerk_orig[:, joint_idx], 'b-', linewidth=1)
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].set_title("Smoothed Jerk")
    if len(jerk_smooth) > 0:
        axes[2, 1].plot(times[:-3], jerk_smooth[:, joint_idx], 'g-', linewidth=1)
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def compute_smoothness_metrics(positions: np.ndarray, dt: float) -> dict:
    """计算平滑度指标"""
    velocity, acceleration, jerk = compute_derivatives(positions, dt)
    
    metrics = {
        "max_velocity": np.max(np.abs(velocity)) if len(velocity) > 0 else 0,
        "max_acceleration": np.max(np.abs(acceleration)) if len(acceleration) > 0 else 0,
        "max_jerk": np.max(np.abs(jerk)) if len(jerk) > 0 else 0,
        "mean_abs_jerk": np.mean(np.abs(jerk)) if len(jerk) > 0 else 0,
        "jerk_std": np.std(jerk) if len(jerk) > 0 else 0,
    }
    return metrics


def analyze_trajectory_file(filepath: str, dt: float = 0.033):
    """分析保存的轨迹文件"""
    data = np.load(filepath, allow_pickle=True)
    
    print(f"\n{'='*60}")
    print(f"Trajectory File: {filepath}")
    print(f"{'='*60}")
    
    keys = list(data.keys())
    print(f"Available keys: {keys}")
    
    for key in keys:
        arr = data[key]
        if arr.ndim >= 2:
            print(f"\n--- {key} ---")
            print(f"Shape: {arr.shape}")
            
            if arr.shape[0] > 5:  # 假设是轨迹数据
                metrics = compute_smoothness_metrics(arr, dt)
                print(f"Max Velocity: {metrics['max_velocity']:.2f}")
                print(f"Max Acceleration: {metrics['max_acceleration']:.2f}")
                print(f"Max Jerk: {metrics['max_jerk']:.2f}")
                print(f"Mean |Jerk|: {metrics['mean_abs_jerk']:.2f}")
    
    return data


def main():
    parser = argparse.ArgumentParser(description="Trajectory Visualization Tool")
    parser.add_argument("--input", type=str, required=True, help="Input NPZ file")
    parser.add_argument("--dt", type=float, default=0.033, help="Time step (1/fps)")
    parser.add_argument("--output_dir", type=str, default="./trajectory_plots", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 分析文件
    data = analyze_trajectory_file(args.input, args.dt)
    
    # 绘制图表
    if "original" in data:
        fig = plot_single_trajectory(
            data["original"], args.dt,
            title="Original Trajectory",
            save_path=os.path.join(args.output_dir, "original.png")
        )
    
    if "smoothed" in data:
        fig = plot_single_trajectory(
            data["smoothed"], args.dt,
            title="Smoothed Trajectory",
            save_path=os.path.join(args.output_dir, "smoothed.png")
        )
    
    if "original" in data and "smoothed" in data:
        fig = plot_comparison(
            data["original"], data["smoothed"], args.dt,
            title="Original vs Smoothed",
            save_path=os.path.join(args.output_dir, "comparison.png")
        )
    
    if args.show:
        plt.show()
    
    print(f"\nPlots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
