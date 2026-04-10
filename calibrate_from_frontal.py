"""
calibrate_from_frontal.py
=========================
从"驾驶员正视前方"图片估计相机外参旋转矩阵，并验证补偿效果。
推理后端：SemiUHPE（TPAMI 2025）开源模型。

【使用场景】
  只有 A 柱相机视角下驾驶员正视前方的图片，没有视频，没有转头数据。

【核心原理】
  当驾驶员正视前方时，定义此时头部姿态为世界坐标系零点。
  此时模型预测的 R_pred = R_extrinsic（相机安装旋转矩阵）。
  估计出 R_extrinsic 后，对任意帧的预测结果做：

      R_world = R_extrinsic.T @ R_pred

  注意：使用转置（旋转矩阵的逆），而不是直接相乘。
  原因：R_pred 描述"将标准正脸旋转到当前头部姿态（在相机坐标系中）"，
  R_extrinsic.T 将其从相机坐标系变换回世界坐标系。

【SemiUHPE DAD3DHeads 欧拉角约定】
  官方 predict.py 中的转换（train_labeled == "DAD3DHeads"）：
      rot_mat_2 = np.transpose(rot_mat)
      angle = Rotation.from_matrix(rot_mat_2).as_euler("xyz", degrees=True)
      roll, pitch, yaw = angle[2], angle[0] - 180, angle[1]

  重要：在此约定下，正视前方（世界 pitch=0°）对应的 rot_mat 使得提取出的
  pitch ≈ ±180°，而不是 0°。本脚本在输出时自动做 ±180° 偏移修正，
  使得正视前方显示为 pitch=0°，符合直觉。

【快速开始】
  # 0. 安装依赖
  pip install torch torchvision scipy opencv-python pillow matplotlib pandas huggingface_hub

  # 1. 克隆 SemiUHPE 仓库
  git clone https://github.com/hnuzhy/SemiUHPE.git

  # 2. 下载预训练权重（自动从 HuggingFace 下载）
  python calibrate_from_frontal.py --mode download_weights --semiuhpe_root ./SemiUHPE

  # 3. 估计外参（从正视前方图片）
  python calibrate_from_frontal.py --mode calibrate \
      --images frontal1.jpg frontal2.jpg frontal3.jpg \
      --semiuhpe_root ./SemiUHPE \
      --weights weights/DAD-COCOHead-EffNetV2-S-best.pth \
      --network effinetv2 \
      --output extrinsic.npy

  # 4. 验证补偿效果（补偿后 Pitch 应接近 0°，Yaw 应反映实际转头角度）
  python calibrate_from_frontal.py --mode verify \
      --images look_left.jpg look_right.jpg \
      --semiuhpe_root ./SemiUHPE \
      --weights weights/DAD-COCOHead-EffNetV2-S-best.pth \
      --network effinetv2 \
      --extrinsic extrinsic.npy

  # 5. 批量补偿伪标签 CSV
  python calibrate_from_frontal.py --mode compensate \
      --label_csv pseudo_labels.csv \
      --extrinsic extrinsic.npy \
      --output_csv pseudo_labels_compensated.csv
"""

import argparse
import os
import sys
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# -- 可选：matplotlib 用于可视化 --
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# SemiUHPE DAD3DHeads 欧拉角约定（完整复现官方 predict.py）
# =============================================================================

def _limit_angle(angle, pi=180.0):
    """将角度限制在 [-pi, pi] 范围内（来自 SemiUHPE src/utils.py）"""
    if angle < -pi:
        k = -2 * (int(angle / pi) // 2)
        angle = angle + k * pi
    if angle > pi:
        k = 2 * ((int(angle / pi) + 1) // 2)
        angle = angle - k * pi
    return angle


def rotmat_to_euler_raw(rot_mat):
    """
    从 SemiUHPE 旋转矩阵提取原始欧拉角（官方 predict.py DAD3DHeads 分支）。

    官方代码：
        rot_mat_2 = np.transpose(rot_mat)
        angle = Rotation.from_matrix(rot_mat_2).as_euler("xyz", degrees=True)
        roll, pitch, yaw = angle[2], angle[0]-180, angle[1]

    返回：(pitch_raw, yaw, roll)，其中 pitch_raw 在正视前方时约为 ±180°。
    """
    rot_mat_2 = np.transpose(rot_mat)
    angle = Rotation.from_matrix(rot_mat_2).as_euler("xyz", degrees=True)
    pitch_raw = _limit_angle(angle[0] - 180)
    yaw       = _limit_angle(angle[1])
    roll      = _limit_angle(angle[2])
    return pitch_raw, yaw, roll


def normalize_pitch(pitch_raw):
    """
    将 DAD3D 约定的原始 Pitch 转换为直觉俯仰角（正视前方=0°）。

    在 DAD3D 约定下，正视前方对应 pitch_raw ≈ ±180°。
    本函数将其偏移 ±180°，使正视前方显示为 0°。
    """
    if pitch_raw > 90:
        return pitch_raw - 180.0
    elif pitch_raw < -90:
        return pitch_raw + 180.0
    return pitch_raw


def rotmat_to_euler(rot_mat):
    """
    从 SemiUHPE 旋转矩阵提取欧拉角（度），输出直觉俯仰角。

    返回：(pitch, yaw, roll)
        pitch : 俯仰角，正视前方=0°，低头为负，抬头为正
        yaw   : 偏航角，正视前方=0°，右转为正，左转为负
        roll  : 滚转角，正视前方=0°
    """
    pitch_raw, yaw, roll = rotmat_to_euler_raw(rot_mat)
    return normalize_pitch(pitch_raw), yaw, roll


def euler_to_rotmat(pitch, yaw, roll):
    """
    将直觉欧拉角（度）还原为 SemiUHPE 旋转矩阵（DAD3D 约定的逆向）。

    参数：
        pitch : 俯仰角（正视前方=0°）
        yaw   : 偏航角
        roll  : 滚转角

    返回：rot_mat（3x3 numpy array）
    """
    # 逆向：pitch_raw = pitch + 180（如果 pitch <= 0）或 pitch - 180（如果 pitch > 0）
    # 等价于：pitch_raw = normalize_pitch 的逆运算
    if pitch > 0:
        pitch_raw = pitch - 180.0
    else:
        pitch_raw = pitch + 180.0
    rot_mat_2 = Rotation.from_euler('xyz', [pitch_raw + 180, yaw, roll], degrees=True).as_matrix()
    return rot_mat_2.T


# =============================================================================
# SemiUHPE 模型加载与推理
# =============================================================================

def setup_semiuhpe_path(semiuhpe_root):
    """将 SemiUHPE 仓库根目录加入 Python 路径"""
    semiuhpe_root = os.path.abspath(semiuhpe_root)
    if not os.path.isdir(semiuhpe_root):
        raise FileNotFoundError(
            f"SemiUHPE 仓库目录不存在: {semiuhpe_root}\n"
            "请先克隆仓库: git clone https://github.com/hnuzhy/SemiUHPE.git"
        )
    if semiuhpe_root not in sys.path:
        sys.path.insert(0, semiuhpe_root)


def load_semiuhpe_model(semiuhpe_root, weights_path, network='effinetv2', device=None):
    """
    加载 SemiUHPE 预训练模型。

    参数:
        semiuhpe_root : SemiUHPE 仓库根目录路径
        weights_path  : 权重文件路径（相对于 semiuhpe_root 或绝对路径）
        network       : 骨干网络，可选 'effinetv2'（默认）、'resnet50'、'repvgg'
        device        : 'cuda' 或 'cpu'，None 时自动选择

    返回:
        (net, device) : 模型对象和设备字符串

    可用预训练权重（来源：https://huggingface.co/HoyerChou/SemiUHPE）:
        DAD-COCOHead-EffNetV2-S-best.pth  336 MB  -- 推荐（--network effinetv2）
        DAD-WildHead-EffNetV2-S-best.pth  336 MB  -- 野外场景更强（--network effinetv2）
        DAD-COCOHead-ResNet50-best.pth    395 MB  -- (--network resnet50)
        DAD-COCOHead-RepVGG-best.pth      719 MB  -- (--network repvgg)
    """
    import torch

    setup_semiuhpe_path(semiuhpe_root)

    if not os.path.isabs(weights_path):
        weights_path = os.path.join(semiuhpe_root, weights_path)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"权重文件不存在: {weights_path}\n"
            "请先运行: python calibrate_from_frontal.py --mode download_weights"
        )

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[模型] 使用设备: {device}")

    class MinimalConfig:
        num_classes = 9

    config = MinimalConfig()
    config.network = network

    from src.networks import get_network
    net = get_network(config)

    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and 'ema_net' in ckpt:
        state_dict = ckpt['ema_net']
    elif isinstance(ckpt, dict) and 'net' in ckpt:
        state_dict = ckpt['net']
    else:
        state_dict = ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    net.load_state_dict(new_state_dict, strict=True)

    if device == 'cuda':
        net = net.cuda()
    net.eval()
    print(f"[模型] 权重加载成功: {os.path.basename(weights_path)}")
    return net, device


def predict_rotation_matrix(img_bgr, net, device):
    """
    对单张裁剪好的头部图片做推理，返回旋转矩阵 rot_mat（3x3 numpy array）。

    参数:
        img_bgr : BGR 格式 numpy 图像（已裁剪为头部区域，任意分辨率）
        net     : 已加载的 SemiUHPE 模型
        device  : 'cuda' 或 'cpu'

    返回:
        rot_mat : 3x3 旋转矩阵（与官方 predict.py 中的 rot_mat 一致）

    内部流程（与 SemiUHPE predict.py 一致）:
        1. BGR -> RGB -> PIL -> resize(224,224)
        2. ToTensor + Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        3. net(img) -> 9 维 Fisher 向量
        4. batch_torch_A_to_R(fisher_out) -> 3x3 旋转矩阵（SVD 正交化）
    """
    import torch
    import torchvision.transforms as tfs
    from PIL import Image
    from src.fisher.fisher_utils import batch_torch_A_to_R

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((224, 224))
    img_tensor = tfs.ToTensor()(img_pil)
    img_tensor = tfs.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])(img_tensor)
    img_tensor = img_tensor.unsqueeze(0)
    if device == 'cuda':
        img_tensor = img_tensor.cuda()

    with torch.no_grad():
        fisher_out = net(img_tensor)
        pd_m = batch_torch_A_to_R(fisher_out)

    return pd_m.detach().cpu().numpy()[0]


# =============================================================================
# 旋转矩阵工具函数
# =============================================================================

def average_rotation_matrices(R_list):
    """
    多个旋转矩阵的 SO(3) 均值（SVD 正交化投影）。

    直接对矩阵元素取算术均值会得到非法旋转矩阵（行列式不为 1）。
    本函数先对所有矩阵求和，再通过 SVD 投影到 SO(3) 流形上，
    是 Frechet 均值的近似，保证结果是合法旋转矩阵。
    """
    R_sum = np.sum(np.array(R_list), axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt
    return R_mean


def compensate_rotmat(R_pred, R_extrinsic):
    """
    将模型预测的旋转矩阵从相机坐标系补偿到世界坐标系。

    公式：R_world = R_extrinsic.T @ R_pred

    推导：
        R_pred = R_extrinsic @ R_world_head（相机坐标系中的头部旋转）
        R_world_head = R_extrinsic.T @ R_pred（逆变换还原世界坐标系）

        验证：正视前方时 R_pred = R_extrinsic，
              R_world_head = R_extrinsic.T @ R_extrinsic = I（单位矩阵）
              从 I 提取欧拉角 -> pitch=0°, yaw=0°, roll=0° ✓

    参数:
        R_pred      : 模型预测的旋转矩阵（3x3 numpy array）
        R_extrinsic : 外参旋转矩阵（3x3 numpy array，由 calibrate 模式生成）

    返回:
        R_world : 世界坐标系旋转矩阵（3x3 numpy array）
    """
    return R_extrinsic.T @ R_pred


def compensate_euler(pitch, yaw, roll, R_extrinsic):
    """
    将 SemiUHPE 输出的欧拉角（度）补偿到世界坐标系。

    参数:
        pitch, yaw, roll : SemiUHPE 预测的欧拉角（度），已经过 normalize_pitch 处理
        R_extrinsic      : 外参旋转矩阵（3x3 numpy array）

    返回:
        (pitch_world, yaw_world, roll_world) : 世界坐标系欧拉角（度）
    """
    R_pred = euler_to_rotmat(pitch, yaw, roll)
    R_world = compensate_rotmat(R_pred, R_extrinsic)
    return rotmat_to_euler(R_world)


# =============================================================================
# Step 0：下载预训练权重
# =============================================================================

def download_weights(semiuhpe_root, model_name='DAD-COCOHead-EffNetV2-S-best.pth'):
    """
    从 HuggingFace (HoyerChou/SemiUHPE) 下载预训练权重到 semiuhpe_root/weights/ 目录。

    可用模型：
        DAD-COCOHead-EffNetV2-S-best.pth  336 MB  -- 推荐（--network effinetv2）
        DAD-WildHead-EffNetV2-S-best.pth  336 MB  -- 野外场景更强（--network effinetv2）
        DAD-COCOHead-ResNet50-best.pth    395 MB  -- (--network resnet50)
        DAD-COCOHead-RepVGG-best.pth      719 MB  -- (--network repvgg)
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[错误] 请先安装: pip install huggingface_hub")
        sys.exit(1)

    weights_dir = os.path.join(os.path.abspath(semiuhpe_root), 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    save_path = os.path.join(weights_dir, model_name)

    if os.path.exists(save_path):
        print(f"[跳过] 权重文件已存在: {save_path}")
        return save_path

    print(f"[下载] 从 HuggingFace 下载 {model_name} ...")
    downloaded = hf_hub_download(
        repo_id='HoyerChou/SemiUHPE',
        filename=model_name,
        local_dir=weights_dir,
        local_dir_use_symlinks=False
    )
    print(f"[完成] 权重已保存: {downloaded}")
    return downloaded


# =============================================================================
# Step 1：从正视前方图片估计外参
# =============================================================================

def calibrate(image_paths, net, device, output_path='extrinsic.npy'):
    """
    从正视前方图片估计相机外参旋转矩阵。

    参数:
        image_paths : 正视前方图片路径列表（建议 5-20 张，越多越稳定）
        net         : 已加载的 SemiUHPE 模型
        device      : 'cuda' 或 'cpu'
        output_path : 外参矩阵保存路径（.npy）

    返回:
        R_extrinsic : 3x3 外参旋转矩阵

    原理：
        正视前方时 R_world_head = I，因此 R_pred = R_extrinsic @ I = R_extrinsic。
        多张图片取 SO(3) 均值（SVD 正交化）以提高稳定性。
    """
    print(f"[Step 1] 从 {len(image_paths)} 张正视前方图片估计外参...")

    R_list = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  [警告] 无法读取图片: {path}，跳过")
            continue

        R = predict_rotation_matrix(img, net, device)
        pitch, yaw, roll = rotmat_to_euler(R)
        R_list.append(R)
        print(f"  {os.path.basename(path):35s}  "
              f"Pitch={pitch:+7.2f}  Yaw={yaw:+7.2f}  Roll={roll:+7.2f}  (deg)")

    if len(R_list) == 0:
        raise ValueError("没有成功处理任何图片，请检查图片路径")

    if len(R_list) == 1:
        R_extrinsic = R_list[0]
        print("\n[提示] 只有 1 张图片，建议提供更多图片以提高稳定性")
    else:
        R_extrinsic = average_rotation_matrices(R_list)

    pitch_e, yaw_e, roll_e = rotmat_to_euler(R_extrinsic)
    print(f"\n[结果] 相机安装角度估计（SO(3) 均值）:")
    print(f"  Pitch = {pitch_e:+.2f} deg  (相机偏上/下，正视前方时预期接近 0)")
    print(f"  Yaw   = {yaw_e:+.2f} deg  (相机偏左/右，A柱侧视预期约 +-30~45 deg)")
    print(f"  Roll  = {roll_e:+.2f} deg  (相机旋转，预期接近 0)")

    np.save(output_path, R_extrinsic)
    print(f"\n[完成] 外参矩阵已保存: {output_path}")
    return R_extrinsic


# =============================================================================
# Step 2：验证补偿效果
# =============================================================================

def verify(image_paths, net, device, R_extrinsic):
    """
    对转头图片验证外参补偿效果。

    预期结果：
      - 补偿后 Pitch 应接近 0 deg（驾驶员没有低头/抬头）
      - 补偿后 Yaw 应反映驾驶员实际的转头角度

    参数:
        image_paths : 转头图片路径列表
        net         : 已加载的 SemiUHPE 模型
        device      : 'cuda' 或 'cpu'
        R_extrinsic : 外参旋转矩阵（由 calibrate 模式生成）

    输出:
        打印补偿前后欧拉角对比表
        生成 verify_result.png（若安装了 matplotlib）
    """
    print(f"\n[Step 2] 验证外参补偿效果（共 {len(image_paths)} 张图片）...")
    print(f"\n{'图片':<35} {'Pitch(前)':>10} {'Yaw(前)':>9} {'Roll(前)':>9}"
          f" {'Pitch(后)':>10} {'Yaw(后)':>9} {'Roll(后)':>9}")
    print("-" * 100)

    rows = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  [警告] 无法读取: {path}")
            continue

        R_pred = predict_rotation_matrix(img, net, device)
        pitch, yaw, roll = rotmat_to_euler(R_pred)

        R_world = compensate_rotmat(R_pred, R_extrinsic)
        pitch_w, yaw_w, roll_w = rotmat_to_euler(R_world)

        name = os.path.basename(path)
        print(f"{name:<35} {pitch:>+9.2f}  {yaw:>+9.2f}  {roll:>+9.2f}"
              f"  {pitch_w:>+9.2f}  {yaw_w:>+9.2f}  {roll_w:>+9.2f}")
        rows.append((name, pitch, yaw, roll, pitch_w, yaw_w, roll_w))

    if HAS_MATPLOTLIB and len(rows) > 1:
        names = [r[0] for r in rows]
        pitches_before = [r[1] for r in rows]
        pitches_after  = [r[4] for r in rows]

        fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.2), 5))
        x = range(len(rows))
        ax.bar([i - 0.2 for i in x], pitches_before, width=0.4,
               label='Pitch before', color='tomato', alpha=0.8)
        ax.bar([i + 0.2 for i in x], pitches_after, width=0.4,
               label='Pitch after', color='steelblue', alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Pitch (degrees)')
        ax.set_title('Pitch Before / After Extrinsic Compensation (SemiUHPE)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        out = 'verify_result.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n[可视化] 对比图已保存: {out}")

    return rows


# =============================================================================
# Step 3：批量补偿伪标签 CSV
# =============================================================================

def compensate_csv(label_csv, R_extrinsic, output_csv,
                   pitch_col='pitch', yaw_col='yaw', roll_col='roll'):
    """
    批量对伪标签 CSV 文件做外参补偿。

    CSV 格式示例（支持任意其他列，只处理 pitch/yaw/roll 三列）：
        image_path,pitch,yaw,roll
        img001.jpg,-8.5,12.3,1.2
        img002.jpg,-9.0,-5.1,0.8

    注意：CSV 中的欧拉角应为 SemiUHPE 输出的 (pitch, yaw, roll)，
    即已经过 normalize_pitch 处理的直觉俯仰角，单位：度。

    参数:
        label_csv   : 输入 CSV 路径
        R_extrinsic : 外参旋转矩阵（由 calibrate 模式生成）
        output_csv  : 输出 CSV 路径
        pitch_col   : CSV 中 pitch 列名（默认 'pitch'）
        yaw_col     : CSV 中 yaw 列名（默认 'yaw'）
        roll_col    : CSV 中 roll 列名（默认 'roll'）

    输出列说明：
        pitch/yaw/roll         : 补偿后的世界坐标系欧拉角
        pitch_cam/yaw_cam/roll_cam : 原始相机坐标系欧拉角（保留）
    """
    try:
        import pandas as pd
    except ImportError:
        print("[错误] 请先安装 pandas: pip install pandas")
        sys.exit(1)

    print(f"\n[Step 3] 批量补偿伪标签: {label_csv}")
    df = pd.read_csv(label_csv)

    for col in [pitch_col, yaw_col, roll_col]:
        if col not in df.columns:
            raise ValueError(f"CSV 中未找到列 '{col}'，当前列: {list(df.columns)}")

    pitch_w_list, yaw_w_list, roll_w_list = [], [], []

    for _, row in df.iterrows():
        pitch_w, yaw_w, roll_w = compensate_euler(
            row[pitch_col], row[yaw_col], row[roll_col], R_extrinsic
        )
        pitch_w_list.append(pitch_w)
        yaw_w_list.append(yaw_w)
        roll_w_list.append(roll_w)

    df[f'{pitch_col}_cam'] = df[pitch_col]
    df[f'{yaw_col}_cam']   = df[yaw_col]
    df[f'{roll_col}_cam']  = df[roll_col]
    df[pitch_col] = pitch_w_list
    df[yaw_col]   = yaw_w_list
    df[roll_col]  = roll_w_list

    df.to_csv(output_csv, index=False)

    print(f"  处理行数: {len(df)}")
    print(f"  Pitch 补偿前: mean={df[f'{pitch_col}_cam'].mean():.2f}  "
          f"std={df[f'{pitch_col}_cam'].std():.2f}  (deg)")
    print(f"  Pitch 补偿后: mean={df[pitch_col].mean():.2f}  "
          f"std={df[pitch_col].std():.2f}  (deg)")
    print(f"[完成] 补偿后标签已保存: {output_csv}")


# =============================================================================
# 主程序
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='使用 SemiUHPE 从正视前方图片估计相机外参，并对伪标签做坐标系补偿',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 下载预训练权重（首次使用）
  python calibrate_from_frontal.py --mode download_weights \\
      --semiuhpe_root ./SemiUHPE

  # 估计外参（先重新运行 calibrate，再运行 verify）
  python calibrate_from_frontal.py --mode calibrate \\
      --images frontal1.jpg frontal2.jpg frontal3.jpg \\
      --semiuhpe_root ./SemiUHPE \\
      --weights weights/DAD-COCOHead-EffNetV2-S-best.pth \\
      --network effinetv2 \\
      --output extrinsic.npy

  # 验证补偿效果（补偿后 Pitch 应接近 0°）
  python calibrate_from_frontal.py --mode verify \\
      --images look_left.jpg look_right.jpg \\
      --semiuhpe_root ./SemiUHPE \\
      --weights weights/DAD-COCOHead-EffNetV2-S-best.pth \\
      --network effinetv2 \\
      --extrinsic extrinsic.npy

  # 批量补偿伪标签 CSV
  python calibrate_from_frontal.py --mode compensate \\
      --label_csv pseudo_labels.csv \\
      --extrinsic extrinsic.npy \\
      --output_csv pseudo_labels_compensated.csv
        """
    )
    parser.add_argument('--mode',
                        choices=['download_weights', 'calibrate', 'verify', 'compensate'],
                        required=True,
                        help='运行模式')
    parser.add_argument('--semiuhpe_root', default='./SemiUHPE',
                        help='SemiUHPE 仓库根目录路径（默认 ./SemiUHPE）')
    parser.add_argument('--weights',
                        default='weights/DAD-COCOHead-EffNetV2-S-best.pth',
                        help='权重文件路径（相对于 semiuhpe_root 或绝对路径）')
    parser.add_argument('--network', default='effinetv2',
                        choices=['resnet50', 'effinetv2', 'repvgg', 'resnet18', 'mobilenet'],
                        help='骨干网络（需与权重匹配，默认 effinetv2）')
    parser.add_argument('--device', default=None,
                        help='推理设备（cuda/cpu，默认自动选择）')
    parser.add_argument('--images', nargs='+', default=[],
                        help='图片路径列表（calibrate/verify 模式使用）')
    parser.add_argument('--extrinsic', default='extrinsic.npy',
                        help='外参矩阵路径（verify/compensate 模式使用）')
    parser.add_argument('--output', default='extrinsic.npy',
                        help='外参矩阵保存路径（calibrate 模式使用）')
    parser.add_argument('--label_csv', default=None,
                        help='输入伪标签 CSV 路径（compensate 模式使用）')
    parser.add_argument('--output_csv', default='pseudo_labels_compensated.csv',
                        help='输出补偿后 CSV 路径（compensate 模式使用）')
    parser.add_argument('--pitch_col', default='pitch', help='CSV 中 pitch 列名')
    parser.add_argument('--yaw_col',   default='yaw',   help='CSV 中 yaw 列名')
    parser.add_argument('--roll_col',  default='roll',  help='CSV 中 roll 列名')
    parser.add_argument('--hf_model',
                        default='DAD-COCOHead-EffNetV2-S-best.pth',
                        help='从 HuggingFace 下载的权重文件名（download_weights 模式使用）')

    args = parser.parse_args()

    if args.mode == 'download_weights':
        download_weights(args.semiuhpe_root, args.hf_model)
        return

    if args.mode in ('calibrate', 'verify'):
        net, device = load_semiuhpe_model(
            args.semiuhpe_root, args.weights, args.network, args.device
        )

    if args.mode == 'calibrate':
        if not args.images:
            parser.error("calibrate 模式需要提供 --images")
        calibrate(args.images, net, device, args.output)

    elif args.mode == 'verify':
        if not args.images:
            parser.error("verify 模式需要提供 --images")
        if not os.path.exists(args.extrinsic):
            parser.error(f"外参文件不存在: {args.extrinsic}，请先运行 calibrate 模式")
        R_extrinsic = np.load(args.extrinsic)
        verify(args.images, net, device, R_extrinsic)

    elif args.mode == 'compensate':
        if args.label_csv is None:
            parser.error("compensate 模式需要提供 --label_csv")
        if not os.path.exists(args.extrinsic):
            parser.error(f"外参文件不存在: {args.extrinsic}，请先运行 calibrate 模式")
        R_extrinsic = np.load(args.extrinsic)
        compensate_csv(args.label_csv, R_extrinsic, args.output_csv,
                       args.pitch_col, args.yaw_col, args.roll_col)


if __name__ == '__main__':
    main()
