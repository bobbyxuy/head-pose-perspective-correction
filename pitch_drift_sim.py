"""
精确的 Pitch 漂移仿真 v3
核心思路：
  - 真实世界中，驾驶员头部姿态由旋转矩阵 R_head 描述（相对于相机坐标系）
  - 当相机从 A 柱侧视时，R_head 包含了相机安装偏置 + 真实头部姿态
  - Naive 模型（BBox 裁剪后）拿到的是 R_head，但它以为相机是正视的
    所以它直接从 R_head 提取欧拉角，得到的 Pitch 是"含相机偏置的混合角"
  - 矫正后，我们先乘以 R_cam_inv，把相机偏置剥离，再提取真实 Pitch
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
from scipy.spatial.transform import Rotation

# ─────────────────────────────────────────────
# 场景参数
# ─────────────────────────────────────────────
# 相机安装参数（A 柱侧视）
CAM_YAW_DEG   =  30.0   # 相机向左偏置 30°（从驾驶员视角看，相机在左前方）
CAM_PITCH_DEG = -10.0   # 相机俯视 10°

# 相机安装旋转矩阵：描述相机坐标系相对于"驾驶员正前方"坐标系的旋转
R_cam = Rotation.from_euler('YX', [CAM_YAW_DEG, CAM_PITCH_DEG], degrees=True).as_matrix()
R_cam_inv = R_cam.T  # 正交矩阵的逆 = 转置

TRUE_PITCH = 5.0   # 驾驶员真实 Pitch（固定）

yaw_angles = np.linspace(-45, 45, 181)
pitch_naive_list     = []
pitch_corrected_list = []

for yaw_deg in yaw_angles:
    # ── 构造真实的头部旋转矩阵（相对于世界/车辆坐标系）──
    # 驾驶员头部：先转 Yaw，再转 Pitch（真实 Pitch 固定）
    R_head_world = Rotation.from_euler('YX', [yaw_deg, TRUE_PITCH], degrees=True).as_matrix()

    # ── 相机看到的头部旋转矩阵 ──
    # R_observed = R_cam^-1 * R_head_world
    # 即：在相机坐标系中，头部的旋转
    R_observed = R_cam_inv @ R_head_world

    # ── Naive 模型的做法 ──
    # 模型以为相机是正视的，直接从 R_observed 提取欧拉角
    # 使用 'YXZ' 顺序（Yaw-Pitch-Roll，head pose 领域最常用）
    euler_naive = Rotation.from_matrix(R_observed).as_euler('YXZ', degrees=True)
    pitch_naive = euler_naive[1]  # 第二个分量是 Pitch
    pitch_naive_list.append(pitch_naive)

    # ── 矫正后的做法 ──
    # 先乘以 R_cam 的逆，把相机安装偏置剥离
    R_corrected = R_cam @ R_observed  # = R_head_world
    euler_corrected = Rotation.from_matrix(R_corrected).as_euler('YXZ', degrees=True)
    pitch_corrected = euler_corrected[1]
    pitch_corrected_list.append(pitch_corrected)

pitch_naive_arr     = np.array(pitch_naive_list)
pitch_corrected_arr = np.array(pitch_corrected_list)

print(f"相机安装偏置: Yaw={CAM_YAW_DEG}°, Pitch={CAM_PITCH_DEG}°")
print(f"真实 Pitch: {TRUE_PITCH}°")
print(f"\n未矫正 Pitch 范围: [{pitch_naive_arr.min():.2f}°, {pitch_naive_arr.max():.2f}°]")
print(f"未矫正 Pitch 漂移量: {pitch_naive_arr.max() - pitch_naive_arr.min():.2f}°")
print(f"\n矫正后 Pitch 范围: [{pitch_corrected_arr.min():.2f}°, {pitch_corrected_arr.max():.2f}°]")
print(f"矫正后 Pitch 漂移量: {pitch_corrected_arr.max() - pitch_corrected_arr.min():.4f}°")

# ─────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#1a1a2e')

# ── 左图：未矫正 ──
ax1 = axes[0]
ax1.plot(yaw_angles, pitch_naive_arr, color='#ff6b6b', linewidth=2.5,
         label='未矫正模型预测 Pitch', zorder=3)
ax1.axhline(y=TRUE_PITCH, color='#51cf66', linewidth=2.5, linestyle='--',
            label=f'真实 Pitch = {TRUE_PITCH}°', zorder=2)
ax1.fill_between(yaw_angles, pitch_naive_arr, TRUE_PITCH,
                 alpha=0.35, color='#ff6b6b', label='漂移误差', zorder=1)

# 标注最大值和最小值
p_max = pitch_naive_arr.max()
p_min = pitch_naive_arr.min()
ax1.annotate(f'max = {p_max:.1f}°',
             xy=(yaw_angles[np.argmax(pitch_naive_arr)], p_max),
             xytext=(yaw_angles[np.argmax(pitch_naive_arr)] + 3, p_max + 0.5),
             fontsize=11, color='yellow',
             arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5))
ax1.annotate(f'min = {p_min:.1f}°',
             xy=(yaw_angles[np.argmin(pitch_naive_arr)], p_min),
             xytext=(yaw_angles[np.argmin(pitch_naive_arr)] + 3, p_min - 1.5),
             fontsize=11, color='yellow',
             arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5))

ax1.set_xlabel('驾驶员 Yaw 角 (°)', fontsize=12, color='white')
ax1.set_ylabel('预测 Pitch (°)', fontsize=12, color='white')
ax1.set_title(f'未矫正：Pitch 随 Yaw 漂移\n（A 柱侧视，相机偏置 Yaw={CAM_YAW_DEG}° / Pitch={CAM_PITCH_DEG}°）',
              fontsize=12, color='white', fontweight='bold')
ax1.legend(fontsize=11, facecolor='#16213e', labelcolor='white', loc='upper right')
ax1.set_facecolor('#16213e')
ax1.tick_params(colors='gray')
ax1.grid(True, alpha=0.3, color='gray')
for spine in ax1.spines.values():
    spine.set_edgecolor('#444')
ax1.set_xlim(-45, 45)

# 添加漂移范围标注
drift_range = p_max - p_min
ax1.text(0, (p_max + p_min) / 2,
         f'漂移范围\n{drift_range:.1f}°',
         fontsize=13, color='white', ha='center', va='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#cc3333', alpha=0.8, edgecolor='white'))

# ── 右图：矫正后 ──
ax2 = axes[1]
ax2.plot(yaw_angles, pitch_corrected_arr, color='#51cf66', linewidth=2.5,
         label='矫正后预测 Pitch', zorder=3)
ax2.axhline(y=TRUE_PITCH, color='#51cf66', linewidth=2.5, linestyle='--',
            label=f'真实 Pitch = {TRUE_PITCH}°', zorder=2)

residual_max = np.abs(pitch_corrected_arr - TRUE_PITCH).max()
ax2.fill_between(yaw_angles, pitch_corrected_arr, TRUE_PITCH,
                 alpha=0.25, color='#51cf66',
                 label=f'残余误差 (max={residual_max:.4f}°)', zorder=1)

ax2.set_xlabel('驾驶员 Yaw 角 (°)', fontsize=12, color='white')
ax2.set_ylabel('预测 Pitch (°)', fontsize=12, color='white')
ax2.set_title('透视矫正后：Pitch 完全稳定\n（残余误差 ≈ 0°，数值精度级别）',
              fontsize=12, color='white', fontweight='bold')
ax2.legend(fontsize=11, facecolor='#16213e', labelcolor='white', loc='upper right')
ax2.set_facecolor('#16213e')
ax2.tick_params(colors='gray')
ax2.grid(True, alpha=0.3, color='gray')
for spine in ax2.spines.values():
    spine.set_edgecolor('#444')
ax2.set_xlim(-45, 45)
ax2.set_ylim(ax1.get_ylim())

fig.suptitle(
    f'A 柱侧视相机（偏置 Yaw={CAM_YAW_DEG}° / Pitch={CAM_PITCH_DEG}°）透视矫正效果仿真\n'
    f'真实 Pitch 固定为 {TRUE_PITCH}°，驾驶员 Yaw 在 [-45°, 45°] 范围内变化',
    fontsize=14, color='white', fontweight='bold', y=1.02
)

plt.tight_layout()
out = "/home/ubuntu/head_pose_research/pitch_drift_simulation_v3.png"
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"\n仿真图已保存至: {out}")
