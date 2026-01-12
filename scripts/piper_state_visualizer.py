#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Piper 机械臂数据离线可视化工具

读取 inference_smooth_0912.py 生成的日志文件，可视化：
- 发送的动作 (action_sent_*.csv)
- 末端位姿 (end_pose_*.csv): X, Y, Z, RX, RY, RZ
- 关节状态 (joint_state_*.csv): J1-J6

用法:
    python piper_state_visualizer.py --log_dir /home/test/test_tra --timestamp 20260112_100000
    python piper_state_visualizer.py --action_csv path/to/action.csv --pose_csv path/to/pose.csv --joint_csv path/to/joint.csv
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


def find_latest_logs(log_dir: str, timestamp: Optional[str] = None):
    """查找指定目录下的日志文件"""
    if timestamp:
        action_csv = os.path.join(log_dir, f"action_sent_{timestamp}.csv")
        pose_csv = os.path.join(log_dir, f"end_pose_{timestamp}.csv")
        joint_csv = os.path.join(log_dir, f"joint_state_{timestamp}.csv")
    else:
        # 查找最新的日志文件
        action_files = sorted(glob.glob(os.path.join(log_dir, "action_sent_*.csv")))
        pose_files = sorted(glob.glob(os.path.join(log_dir, "end_pose_*.csv")))
        joint_files = sorted(glob.glob(os.path.join(log_dir, "joint_state_*.csv")))
        
        action_csv = action_files[-1] if action_files else None
        pose_csv = pose_files[-1] if pose_files else None
        joint_csv = joint_files[-1] if joint_files else None
    
    return action_csv, pose_csv, joint_csv


def load_csv(path: str, header=None):
    """加载 CSV 文件"""
    if path and os.path.exists(path):
        return pd.read_csv(path, header=header)
    return None


def plot_visualization(action_csv: str, pose_csv: str, joint_csv: str, output: str):
    """生成可视化图表"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Piper 机械臂数据可视化", fontsize=14, fontweight='bold')
    
    # 加载数据
    action_df = load_csv(action_csv)
    pose_df = load_csv(pose_csv)
    joint_df = load_csv(joint_csv)
    
    # 子图1: 发送的动作 (关节1-3)
    ax1 = axes[0, 0]
    if action_df is not None and action_df.shape[1] >= 4:
        # 列: timestamp, j1, j2, j3, j4, j5, j6, gripper
        t = action_df.iloc[:, 0] - action_df.iloc[0, 0]  # 相对时间
        ax1.plot(t, action_df.iloc[:, 1], 'r-', label='J1 cmd', alpha=0.8)
        ax1.plot(t, action_df.iloc[:, 2], 'g-', label='J2 cmd', alpha=0.8)
        ax1.plot(t, action_df.iloc[:, 3], 'b-', label='J3 cmd', alpha=0.8)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position (Piper units)")
        ax1.legend(loc='upper right')
    ax1.set_title("发送动作指令 (J1-J3)")
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 发送的动作 (关节4-6)
    ax2 = axes[0, 1]
    if action_df is not None and action_df.shape[1] >= 7:
        t = action_df.iloc[:, 0] - action_df.iloc[0, 0]
        ax2.plot(t, action_df.iloc[:, 4], 'r-', label='J4 cmd', alpha=0.8)
        ax2.plot(t, action_df.iloc[:, 5], 'g-', label='J5 cmd', alpha=0.8)
        ax2.plot(t, action_df.iloc[:, 6], 'b-', label='J6 cmd', alpha=0.8)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Position (Piper units)")
        ax2.legend(loc='upper right')
    ax2.set_title("发送动作指令 (J4-J6)")
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 末端位置 (X, Y, Z)
    ax3 = axes[1, 0]
    if pose_df is not None and pose_df.shape[1] >= 4:
        # 列: timestamp, X, Y, Z, RX, RY, RZ
        t = pose_df.iloc[:, 0] - pose_df.iloc[0, 0]
        ax3.plot(t, pose_df.iloc[:, 1], 'r-', label='X (mm)', alpha=0.8)
        ax3.plot(t, pose_df.iloc[:, 2], 'g-', label='Y (mm)', alpha=0.8)
        ax3.plot(t, pose_df.iloc[:, 3], 'b-', label='Z (mm)', alpha=0.8)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Position (mm)")
        ax3.legend(loc='upper right')
    ax3.set_title("末端位置 (X, Y, Z)")
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 末端姿态 (RX, RY, RZ)
    ax4 = axes[1, 1]
    if pose_df is not None and pose_df.shape[1] >= 7:
        t = pose_df.iloc[:, 0] - pose_df.iloc[0, 0]
        ax4.plot(t, pose_df.iloc[:, 4], 'r-', label='RX (deg)', alpha=0.8)
        ax4.plot(t, pose_df.iloc[:, 5], 'g-', label='RY (deg)', alpha=0.8)
        ax4.plot(t, pose_df.iloc[:, 6], 'b-', label='RZ (deg)', alpha=0.8)
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Orientation (deg)")
        ax4.legend(loc='upper right')
    ax4.set_title("末端姿态 (RX, RY, RZ)")
    ax4.grid(True, alpha=0.3)
    
    # 子图5: 关节反馈 (J1-J3)
    ax5 = axes[2, 0]
    if joint_df is not None and joint_df.shape[1] >= 4:
        # 列: timestamp, j1, j2, j3, j4, j5, j6
        t = joint_df.iloc[:, 0] - joint_df.iloc[0, 0]
        ax5.plot(t, joint_df.iloc[:, 1], 'r-', label='J1 (deg)', alpha=0.8)
        ax5.plot(t, joint_df.iloc[:, 2], 'g-', label='J2 (deg)', alpha=0.8)
        ax5.plot(t, joint_df.iloc[:, 3], 'b-', label='J3 (deg)', alpha=0.8)
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Angle (deg)")
        ax5.legend(loc='upper right')
    ax5.set_title("关节角度反馈 (J1-J3)")
    ax5.grid(True, alpha=0.3)
    
    # 子图6: 关节反馈 (J4-J6)
    ax6 = axes[2, 1]
    if joint_df is not None and joint_df.shape[1] >= 7:
        t = joint_df.iloc[:, 0] - joint_df.iloc[0, 0]
        ax6.plot(t, joint_df.iloc[:, 4], 'r-', label='J4 (deg)', alpha=0.8)
        ax6.plot(t, joint_df.iloc[:, 5], 'g-', label='J5 (deg)', alpha=0.8)
        ax6.plot(t, joint_df.iloc[:, 6], 'b-', label='J6 (deg)', alpha=0.8)
        ax6.set_xlabel("Time (s)")
        ax6.set_ylabel("Angle (deg)")
        ax6.legend(loc='upper right')
    ax6.set_title("关节角度反馈 (J4-J6)")
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"[OK] 图表已保存到 {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Piper 机械臂数据离线可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从日志目录读取最新的日志
  python piper_state_visualizer.py --log_dir /home/test/test_tra
  
  # 指定时间戳
  python piper_state_visualizer.py --log_dir /home/test/test_tra --timestamp 20260112_100000
  
  # 直接指定 CSV 文件
  python piper_state_visualizer.py --action_csv action.csv --pose_csv pose.csv --joint_csv joint.csv
        """
    )
    parser.add_argument("--log_dir", type=str, help="日志目录路径")
    parser.add_argument("--timestamp", type=str, help="日志时间戳 (如 20260112_100000)")
    parser.add_argument("--action_csv", type=str, help="action_sent CSV 文件路径")
    parser.add_argument("--pose_csv", type=str, help="end_pose CSV 文件路径")
    parser.add_argument("--joint_csv", type=str, help="joint_state CSV 文件路径")
    parser.add_argument("--output", type=str, default="", help="输出图片路径 (留空则显示窗口)")
    args = parser.parse_args()
    
    # 确定 CSV 文件路径
    if args.log_dir:
        action_csv, pose_csv, joint_csv = find_latest_logs(args.log_dir, args.timestamp)
    else:
        action_csv = args.action_csv
        pose_csv = args.pose_csv
        joint_csv = args.joint_csv
    
    print(f"Action CSV: {action_csv}")
    print(f"Pose CSV: {pose_csv}")
    print(f"Joint CSV: {joint_csv}")
    
    if not any([action_csv, pose_csv, joint_csv]):
        print("[ERR] 未找到任何日志文件")
        return
    
    # 生成可视化
    plot_visualization(action_csv, pose_csv, joint_csv, args.output)


if __name__ == "__main__":
    main()
