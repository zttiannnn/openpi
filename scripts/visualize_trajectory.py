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


def compute_smoothness_metrics(positions: np.ndarray, dt: float, skip_start: int = 30) -> dict:
    """
    计算平滑度指标
    
    Args:
        positions: (H, D) 位置序列
        dt: 时间步长
        skip_start: 跳过开头多少步（避免起步瞬态影响）
    """
    velocity, acceleration, jerk = compute_derivatives(positions, dt)
    
    # 排除起始段（避免静止到运动的瞬态污染统计）
    if len(jerk) > skip_start:
        jerk_stable = jerk[skip_start:]
        acc_stable = acceleration[skip_start:]
        vel_stable = velocity[skip_start:]
    else:
        jerk_stable = jerk
        acc_stable = acceleration
        vel_stable = velocity
    
    # 计算各维度的绝对值
    jerk_abs = np.abs(jerk_stable) if len(jerk_stable) > 0 else np.zeros((1, positions.shape[1]))
    acc_abs = np.abs(acc_stable) if len(acc_stable) > 0 else np.zeros((1, positions.shape[1]))
    vel_abs = np.abs(vel_stable) if len(vel_stable) > 0 else np.zeros((1, positions.shape[1]))
    
    metrics = {
        # 峰值指标（仍然有参考价值）
        "max_velocity": np.max(vel_abs),
        "max_acceleration": np.max(acc_abs),
        "max_jerk": np.max(jerk_abs),
        
        # Mean 指标（更能反映整体平滑度）
        "mean_velocity": np.mean(vel_abs),
        "mean_acceleration": np.mean(acc_abs),
        "mean_jerk": np.mean(jerk_abs),
        
        # 中位数（对异常值更鲁棒）
        "median_jerk": np.median(jerk_abs),
        
        # 百分位数（看分布）
        "jerk_p90": np.percentile(jerk_abs, 90),
        "jerk_p99": np.percentile(jerk_abs, 99),
        
        # 标准差（看波动程度）
        "jerk_std": np.std(jerk_abs),
        "acc_std": np.std(acc_abs),
        
        # 总能量（积分指标）
        "jerk_energy": np.sum(jerk_abs**2) * dt,
    }
    return metrics


def compare_metrics(metrics1: dict, metrics2: dict, name1: str = "A", name2: str = "B"):
    """对比两组指标"""
    print(f"\n{'='*60}")
    print(f"Comparison: {name1} vs {name2}")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {name1:>15} {name2:>15} {'Change':>10}")
    print("-" * 65)
    
    for key in metrics1:
        v1 = metrics1[key]
        v2 = metrics2[key]
        if v1 != 0:
            change = (v2 - v1) / v1 * 100
            change_str = f"{change:+.1f}%"
        else:
            change_str = "N/A"
        print(f"{key:<25} {v1:>15.2f} {v2:>15.2f} {change_str:>10}")


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
                print(f"Mean |Jerk|: {metrics['mean_jerk']:.2f}")
    
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
    
    # 获取输入文件名（不含扩展名）用于命名
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    
    # 绘制图表
    if "executed" in data:
        fig = plot_single_trajectory(
            data["executed"], args.dt,
            title=f"Trajectory: {base_name} (executed)",
            save_path=os.path.join(args.output_dir, f"{base_name}.png")
        )

    if "original" in data:
        fig = plot_single_trajectory(
            data["original"], args.dt,
            title=f"Trajectory: {base_name} (original)",
            save_path=os.path.join(args.output_dir, f"{base_name}_original.png")
        )
    
    if "smoothed" in data:
        fig = plot_single_trajectory(
            data["smoothed"], args.dt,
            title=f"Trajectory: {base_name} (smoothed)",
            save_path=os.path.join(args.output_dir, f"{base_name}_smoothed.png")
        )
    
    if "original" in data and "smoothed" in data:
        fig = plot_comparison(
            data["original"], data["smoothed"], args.dt,
            title=f"Comparison: {base_name}",
            save_path=os.path.join(args.output_dir, f"{base_name}_comparison.png")
        )
    
    if args.show:
        plt.show()
    
    print(f"\nPlots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
