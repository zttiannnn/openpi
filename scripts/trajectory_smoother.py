"""
TOPP-RA + Ruckig 轨迹平滑模块（简化版）

使用 TOPP-RA 进行时间最优路径规划，使用 Ruckig (社区版)进行 jerk-limited 执行平滑。
社区版 Ruckig 不支持 intermediate waypoints，因此采用分段点到点平滑策略。

流程：
    模型输出动作 → TOPP-RA时间最优规划 → 重采样回原步数 → cubic_transition → Ruckig执行平滑(可选) → 发送给机器人
"""

import numpy as np
import logging
from typing import Tuple, Optional, List

# ============ 检测可用的库 ============

# TOPP-RA
try:
    import toppra as ta
    import toppra.constraint as ta_constraint
    import toppra.algorithm as ta_algo
    TOPPRA_AVAILABLE = True
except ImportError:
    TOPPRA_AVAILABLE = False
    logging.warning("toppra not installed. Install via 'pip install toppra' to enable time-optimal trajectory generation.")

# Ruckig
try:
    from ruckig import InputParameter, OutputParameter, Result, Ruckig
    RUCKIG_AVAILABLE = True
except ImportError:
    RUCKIG_AVAILABLE = False
    logging.warning("ruckig not installed. Install via 'pip install ruckig' to enable jerk-limited trajectory generation.")


# ============ 辅助函数 ============

def resample_trajectory(trajectory: np.ndarray, target_steps: int) -> np.ndarray:
    """
    将轨迹重采样到指定步数。
    
    Args:
        trajectory: (N, D) 原始轨迹
        target_steps: 目标步数
    
    Returns:
        resampled: (target_steps, D) 重采样后的轨迹
    """
    trajectory = np.asarray(trajectory, dtype=float)
    N, D = trajectory.shape
    
    if N == target_steps:
        return trajectory
    
    if N < 2:
        return np.tile(trajectory, (target_steps, 1))
    
    # 线性插值重采样
    old_indices = np.linspace(0, 1, N)
    new_indices = np.linspace(0, 1, target_steps)
    
    resampled = np.zeros((target_steps, D), dtype=float)
    for d in range(D):
        resampled[:, d] = np.interp(new_indices, old_indices, trajectory[:, d])
    
    return resampled


# ============ TOPP-RA 时间最优规划 ============

def toppra_time_optimal(
    waypoints: np.ndarray,
    max_velocity: np.ndarray | float = 1.0,
    max_acceleration: np.ndarray | float = 2.0,
    control_cycle: float = 0.033,
    start_velocity: np.ndarray | None = None,
    maintain_end_velocity: bool = True,  # 是否保持终点速度（不减速到零）
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    使用 TOPP-RA 进行时间最优路径参数化。
    
    将输入的航点通过样条插值构建几何路径，然后使用 TOPP-RA
    在速度/加速度约束下求解时间最优轨迹。
    
    Args:
        waypoints: (H, D) 航点数组
        max_velocity: 每个自由度的最大速度，标量或 (D,) 数组
        max_acceleration: 每个自由度的最大加速度，标量或 (D,) 数组
        control_cycle: 控制周期（秒），用于采样
        start_velocity: (D,) 起始速度，None 表示零速度
        maintain_end_velocity: 如果为 True，终点保持匀速（不减速到零）
    
    Returns:
        trajectory: (N, D) 时间最优轨迹（按控制周期采样）
        duration: 轨迹总时长（秒）
        final_velocity: (D,) 轨迹终点速度
    """
    t0 = None
    try:
        t0 = __import__("time").perf_counter()
    except Exception:
        t0 = None
    if not TOPPRA_AVAILABLE:
        logging.warning("TOPP-RA not available, returning original waypoints")
        zero_vel = np.zeros(waypoints.shape[1]) if waypoints.ndim > 1 else np.zeros(1)
        return waypoints, len(waypoints) * control_cycle, zero_vel
    
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim == 1:
        waypoints = waypoints[None, :]
    
    H, D = waypoints.shape
    if H < 2:
        zero_vel = np.zeros(D)
        return waypoints, 0.0, zero_vel
    
    # 构建速度/加速度约束
    if np.isscalar(max_velocity):
        max_vel = np.full(D, max_velocity)
    else:
        max_vel = np.asarray(max_velocity, dtype=float)
    
    if np.isscalar(max_acceleration):
        max_acc = np.full(D, max_acceleration)
    else:
        max_acc = np.asarray(max_acceleration, dtype=float)
    
    try:
        # 创建路径参数 s ∈ [0, 1]
        ss = np.linspace(0, 1, H)
        
        # 使用样条插值创建几何路径
        path = ta.SplineInterpolator(ss, waypoints)
        
        # 定义约束
        vel_limits = np.column_stack([-max_vel, max_vel])
        acc_limits = np.column_stack([-max_acc, max_acc])
        
        pc_vel = ta_constraint.JointVelocityConstraint(vel_limits)
        pc_acc = ta_constraint.JointAccelerationConstraint(acc_limits)
        
        # TOPP-RA 算法求解
        instance = ta_algo.TOPPRA([pc_vel, pc_acc], path)
        
        # 计算路径参数化
        # sd (ds/dt) 是路径参数的速度
        # 计算路径长度用于估算合适的 sd
        path_length = np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))
        expected_duration = H * control_cycle  # 预期执行时间
        
        # 估算起始 sd
        sd_start = 0.0
        if start_velocity is not None:
            start_vel_norm = np.linalg.norm(start_velocity)
            if start_vel_norm > 0 and path_length > 0:
                # sd = |dq/dt| / |dq/ds| ≈ |velocity| / (path_length / 1.0)
                sd_start = start_vel_norm / path_length
        
        # 估算终点 sd
        if maintain_end_velocity and path_length > 0:
            # 使用"匀速"假设：整个路径以恒定速度执行
            # sd_uniform = 1.0 / expected_duration (因为 s 从 0 到 1)
            sd_end = 1.0 / expected_duration
            # 如果有起始速度，终点速度可以保持相似
            if sd_start > 0:
                sd_end = max(sd_end, sd_start * 0.8)  # 略微减速但不归零
        else:
            sd_end = 0.0  # 传统行为：终点减速到零
        
        # 使用 compute_parameterization 计算路径参数化
        try:
            # 先计算可行的 sd 范围
            feasible_range = instance.compute_feasible_sets()
            if feasible_range is not None:
                # 确保 sd_start 和 sd_end 在可行范围内
                sd_max_start = feasible_range[0, 1] if len(feasible_range) > 0 else 1.0
                sd_max_end = feasible_range[-1, 1] if len(feasible_range) > 0 else 1.0
                sd_start = min(sd_start, sd_max_start * 0.9)  # 留一点余量
                sd_end = min(sd_end, sd_max_end * 0.9)
            
            # 计算路径参数化
            parameterization = instance.compute_parameterization(sd_start, sd_end)
            if parameterization is None:
                # 回退：尝试零终点速度
                logging.debug("TOPP-RA parameterization with non-zero end velocity failed, trying zero end velocity")
                parameterization = instance.compute_parameterization(sd_start, 0.0)
            
            if parameterization is None:
                # 再次回退：使用默认
                traj = instance.compute_trajectory()
            else:
                # 从参数化创建轨迹
                traj = instance.compute_trajectory(sd_start, sd_end)
                
        except Exception as e:
            logging.debug(f"compute_parameterization failed: {e}, falling back to default")
            traj = instance.compute_trajectory()
        
        if traj is None:
            logging.warning("TOPP-RA failed to compute trajectory, returning original waypoints")
            return waypoints, H * control_cycle, np.zeros(D)
        
        # 获取时间最优轨迹的总时长
        duration = traj.duration
        
        # 按控制周期采样
        n_samples = max(2, int(np.ceil(duration / control_cycle)) + 1)
        times = np.linspace(0, duration, n_samples)
        trajectory = np.array([traj(t) for t in times])
        
        # 计算终点速度（用于传递给下一个动作块）
        # 使用数值差分估计
        if len(trajectory) >= 2:
            final_velocity = (trajectory[-1] - trajectory[-2]) / control_cycle
        else:
            final_velocity = np.zeros(D)
        
        if t0 is not None:
            t1 = __import__("time").perf_counter()
            solve_ms = (t1 - t0) * 1000.0
        else:
            solve_ms = None

        if solve_ms is None:
            logging.info(
                "TOPP-RA: sd_start=%.4f sd_end=%.4f duration=%.3fs final_vel_norm=%.4f",
                sd_start,
                sd_end,
                duration,
                float(np.linalg.norm(final_velocity)),
            )
        else:
            logging.info(
                "TOPP-RA: sd_start=%.4f sd_end=%.4f duration=%.3fs final_vel_norm=%.4f solve=%.1fms",
                sd_start,
                sd_end,
                duration,
                float(np.linalg.norm(final_velocity)),
                solve_ms,
            )
        
        return trajectory, duration, final_velocity
        
    except Exception as e:
        logging.exception(f"TOPP-RA failed: {e}")
        return waypoints, H * control_cycle, np.zeros(D)


# ============ Ruckig 分段执行平滑 ============

def ruckig_segment_smooth(
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    start_velocity: np.ndarray | None = None,
    start_acceleration: np.ndarray | None = None,
    max_velocity: np.ndarray | float = 1.0,
    max_acceleration: np.ndarray | float = 2.0,
    max_jerk: np.ndarray | float = 5.0,
    control_cycle: float = 0.033,
    max_duration: float = 10.0,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    使用 Ruckig 社区版进行单段点到点 jerk-limited 轨迹生成。
    
    Args:
        start_pos: (D,) 起始位置
        target_pos: (D,) 目标位置
        start_velocity: (D,) 起始速度，None 则为零
        start_acceleration: (D,) 起始加速度，None 则为零
        max_velocity: 最大速度
        max_acceleration: 最大加速度
        max_jerk: 最大 jerk
        control_cycle: 控制周期
        max_duration: 最大允许时长
    
    Returns:
        trajectory: List of (D,) 位置点
        final_velocity: (D,) 终点速度
        final_acceleration: (D,) 终点加速度
    """
    if not RUCKIG_AVAILABLE:
        logging.warning("Ruckig not available, returning linear interpolation")
        trajectory = [start_pos, target_pos]
        return trajectory, np.zeros_like(start_pos), np.zeros_like(start_pos)
    
    start_pos = np.asarray(start_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    D = len(start_pos)
    
    # 初始化速度和加速度
    vel = np.zeros(D) if start_velocity is None else np.asarray(start_velocity, dtype=float)
    acc = np.zeros(D) if start_acceleration is None else np.asarray(start_acceleration, dtype=float)
    
    # 构建约束
    if np.isscalar(max_velocity):
        max_vel = np.full(D, max_velocity)
    else:
        max_vel = np.asarray(max_velocity, dtype=float)
    
    if np.isscalar(max_acceleration):
        max_acc = np.full(D, max_acceleration)
    else:
        max_acc = np.asarray(max_acceleration, dtype=float)
    
    if np.isscalar(max_jerk):
        max_jrk = np.full(D, max_jerk)
    else:
        max_jrk = np.asarray(max_jerk, dtype=float)
    
    try:
        # 创建 Ruckig 实例（社区版：无中间点）
        ruckig = Ruckig(D, control_cycle)
        inp = InputParameter(D)
        out = OutputParameter(D)
        
        # 设置输入
        inp.current_position = start_pos.tolist()
        inp.current_velocity = vel.tolist()
        inp.current_acceleration = acc.tolist()
        inp.target_position = target_pos.tolist()
        inp.target_velocity = [0.0] * D  # 目标速度为零
        inp.target_acceleration = [0.0] * D
        inp.max_velocity = max_vel.tolist()
        inp.max_acceleration = max_acc.tolist()
        inp.max_jerk = max_jrk.tolist()
        
        # 生成轨迹
        trajectory = []
        max_steps = int(max_duration / control_cycle)
        
        for _ in range(max_steps):
            result = ruckig.update(inp, out)
            trajectory.append(np.array(out.new_position))
            
            if result == Result.Finished:
                break
            elif result == Result.Working:
                out.pass_to_input(inp)
            else:
                logging.warning(f"Ruckig returned error: {result}")
                break
        
        if not trajectory:
            trajectory = [start_pos, target_pos]
            return trajectory, np.zeros(D), np.zeros(D)
        
        # 获取终点状态
        final_vel = np.array(out.new_velocity)
        final_acc = np.array(out.new_acceleration)
        
        return trajectory, final_vel, final_acc
        
    except Exception as e:
        logging.exception(f"Ruckig segment smooth failed: {e}")
        return [start_pos, target_pos], np.zeros(D), np.zeros(D)


def ruckig_piecewise_smooth(
    waypoints: np.ndarray,
    current_velocity: np.ndarray | None = None,
    current_acceleration: np.ndarray | None = None,
    max_velocity: np.ndarray | float = 1.0,
    max_acceleration: np.ndarray | float = 2.0,
    max_jerk: np.ndarray | float = 5.0,
    control_cycle: float = 0.033,
    target_steps: int | None = None,
) -> np.ndarray:
    """
    使用 Ruckig 社区版分段进行 jerk-limited 轨迹平滑。
    
    由于社区版不支持 intermediate waypoints，采用分段点到点策略：
    对每对相邻航点调用 Ruckig，并串联生成完整轨迹。
    
    Args:
        waypoints: (H, D) 航点数组
        current_velocity: (D,) 当前速度，用于第一段
        current_acceleration: (D,) 当前加速度，用于第一段
        max_velocity: 最大速度
        max_acceleration: 最大加速度
        max_jerk: 最大 jerk
        control_cycle: 控制周期
        target_steps: 重采样目标步数，None 则不重采样
    
    Returns:
        smoothed: (N, D) 平滑后的轨迹
    """
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim == 1:
        waypoints = waypoints[None, :]
    
    H, D = waypoints.shape
    if H < 2:
        if target_steps is not None:
            return np.tile(waypoints, (target_steps, 1))
        return waypoints
    
    # 初始状态
    vel = np.zeros(D) if current_velocity is None else np.asarray(current_velocity, dtype=float)
    acc = np.zeros(D) if current_acceleration is None else np.asarray(current_acceleration, dtype=float)
    
    # 分段生成轨迹
    full_trajectory = []
    
    for i in range(H - 1):
        segment, vel, acc = ruckig_segment_smooth(
            start_pos=waypoints[i],
            target_pos=waypoints[i + 1],
            start_velocity=vel,
            start_acceleration=acc,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            max_jerk=max_jerk,
            control_cycle=control_cycle,
        )
        
        # 添加到完整轨迹（避免重复起点）
        if i == 0:
            full_trajectory.extend(segment)
        else:
            full_trajectory.extend(segment[1:])  # 跳过起点（与上一段终点重复）
    
    full_trajectory = np.array(full_trajectory)
    
    # 重采样
    if target_steps is not None:
        full_trajectory = resample_trajectory(full_trajectory, target_steps)
    
    return full_trajectory


# ============ 完整轨迹平滑器（简化版） ============

class TrajectorySmoother:
    """
    TOPP-RA + Ruckig 组合轨迹平滑器（简化版）
    
    工作流程：
    1. 使用 TOPP-RA 进行时间最优路径规划
    2. 强制重采样回输入步数（保证时序对齐）
    3. (可选) 使用 Ruckig 进行 jerk-limited 执行平滑
    """
    
    def __init__(
        self,
        dof: int,
        control_cycle: float = 0.033,
        # TOPP-RA 参数
        toppra_max_velocity: float = 1.0,
        toppra_max_acceleration: float = 2.0,
        # Ruckig 参数
        ruckig_max_velocity: float = 1.0,
        ruckig_max_acceleration: float = 2.0,
        ruckig_max_jerk: float = 5.0,
        # 模式控制
        use_toppra: bool = True,
        use_ruckig: bool = False,  # Ruckig 默认关闭，作为可选增强
    ):
        self.dof = dof
        self.control_cycle = control_cycle
        
        # TOPP-RA
        self.toppra_max_velocity = toppra_max_velocity
        self.toppra_max_acceleration = toppra_max_acceleration
        
        # Ruckig
        self.ruckig_max_velocity = ruckig_max_velocity
        self.ruckig_max_acceleration = ruckig_max_acceleration
        self.ruckig_max_jerk = ruckig_max_jerk
        
        # 模式
        self.use_toppra = use_toppra and TOPPRA_AVAILABLE
        self.use_ruckig = use_ruckig and RUCKIG_AVAILABLE
    
    def smooth(
        self,
        actions: np.ndarray,
        current_velocity: np.ndarray | None = None,
        current_acceleration: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        对动作块进行完整平滑处理。
        
        Args:
            actions: (H, D) 动作块
            current_velocity: (D,) 当前速度
            current_acceleration: (D,) 当前加速度
        
        Returns:
            smoothed: (H, D) 平滑后的动作块（保持原始步数）
        """
        actions = np.asarray(actions, dtype=float)
        if actions.ndim == 1:
            actions = actions[None, :]
        
        H, D = actions.shape
        if H < 2:
            return actions
        
        original_steps = H  # 记录原始步数，用于强制重采样
        result = actions.copy()
        
        # Step 1: TOPP-RA 时间最优规划
        if self.use_toppra:
            result, duration, final_vel = toppra_time_optimal(
                waypoints=result,
                max_velocity=self.toppra_max_velocity,
                max_acceleration=self.toppra_max_acceleration,
                control_cycle=self.control_cycle,
                start_velocity=current_velocity,  # 传入当前速度
            )
            logging.debug(f"TOPP-RA output: {len(result)} points, duration={duration:.3f}s")
            
            # 强制重采样回原始步数（保证时序对齐）
            result = resample_trajectory(result, original_steps)
            logging.debug(f"Resampled to {len(result)} points")
        
        # Step 2: Ruckig jerk-limited 执行平滑（可选）
        if self.use_ruckig:
            result = ruckig_piecewise_smooth(
                waypoints=result,
                current_velocity=current_velocity,
                current_acceleration=current_acceleration,
                max_velocity=self.ruckig_max_velocity,
                max_acceleration=self.ruckig_max_acceleration,
                max_jerk=self.ruckig_max_jerk,
                control_cycle=self.control_cycle,
                target_steps=original_steps,  # 再次重采样保证步数
            )
            logging.debug(f"Ruckig output: {len(result)} points")
        
        return result


# ============ 便捷函数 ============

def smooth_action_chunk(
    actions: np.ndarray,
    current_velocity: np.ndarray | None = None,
    current_acceleration: np.ndarray | None = None,
    control_cycle: float = 0.033,
    toppra_max_velocity: float = 1.0,
    toppra_max_acceleration: float = 2.0,
    ruckig_max_velocity: float = 1.0,
    ruckig_max_acceleration: float = 2.0,
    ruckig_max_jerk: float = 5.0,
    use_toppra: bool = True,
    use_ruckig: bool = False,
) -> np.ndarray:
    """
    便捷函数：对动作块进行 TOPP-RA + Ruckig 平滑。
    
    输出保证与输入同样的步数（强制重采样）。
    
    Args:
        actions: (H, D) 动作块 (不含 gripper)
        current_velocity: 当前速度
        current_acceleration: 当前加速度
        control_cycle: 控制周期
        toppra_max_velocity: TOPP-RA 最大速度
        toppra_max_acceleration: TOPP-RA 最大加速度
        ruckig_max_velocity: Ruckig 最大速度
        ruckig_max_acceleration: Ruckig 最大加速度
        ruckig_max_jerk: Ruckig 最大 jerk
        use_toppra: 是否使用 TOPP-RA
        use_ruckig: 是否使用 Ruckig (可选增强)
    
    Returns:
        smoothed: (H, D) 平滑后的动作块（与输入步数相同）
    """
    actions = np.asarray(actions, dtype=float)
    if actions.ndim == 1:
        actions = actions[None, :]
    
    D = actions.shape[1]
    
    smoother = TrajectorySmoother(
        dof=D,
        control_cycle=control_cycle,
        toppra_max_velocity=toppra_max_velocity,
        toppra_max_acceleration=toppra_max_acceleration,
        ruckig_max_velocity=ruckig_max_velocity,
        ruckig_max_acceleration=ruckig_max_acceleration,
        ruckig_max_jerk=ruckig_max_jerk,
        use_toppra=use_toppra,
        use_ruckig=use_ruckig,
    )
    
    return smoother.smooth(
        actions=actions,
        current_velocity=current_velocity,
        current_acceleration=current_acceleration,
    )
