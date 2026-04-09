"""
calibrate_from_frontal.py
=========================
从"驾驶员正视前方"图片估计相机外参旋转矩阵，并验证补偿效果。
推理后端：SemiUHPE（TPAMI 2025）开源模型。

【使用场景】
  只有 A 柱相机视角下驾驶员正视前方的图片，没有视频，没有转头数据。

【核心原理】
  当驾驶员正视前方时，定义此时头部姿态为世界坐标系零点（R_world = I）。
  此时模型预测的 R_camera 就等于相机外参 R_extrinsic。
  估计出 R_extrinsic 后，对任意帧的预测结果做：
      R_world = R_extrinsic @ R_camera
  从 R_world 提取的欧拉角即为世界坐标系下的真实姿态，
  纯 Yaw 转头时 Pitch 将保持稳定。

【SemiUHPE 欧拉角约定】
  - 网络输出 9 维向量，经 SVD 正交化得到旋转矩阵 R（3x3）
  - 从 R 提取欧拉角顺序为 (pitch, yaw, roll)，单位：度（XYZ 旋转顺序）
  - 本脚本内部统一使用旋转矩阵做运算，不依赖欧拉角加减

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
      --weights weights/DAD-COCOHead-ResNet50-best.pth \
      --output extrinsic.npy

  # 4. 验证补偿效果
  python calibrate_from_frontal.py --mode verify \
      --images look_left.jpg look_right.jpg \
      --semiuhpe_root ./SemiUHPE \
      --weights weights/DAD-COCOHead-ResNet50-best.pth \
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
import math
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


def load_semiuhpe_model(semiuhpe_root, weights_path, network='resnet50', device=None):
    """
    加载 SemiUHPE 预训练模型。

    参数:
        semiuhpe_root : SemiUHPE 仓库根目录路径
        weights_path  : 权重文件路径（相对于 semiuhpe_root 或绝对路径）
        network       : 骨干网络，可选 'resnet50'（默认）、'effinetv2'、'repvgg'
        device        : 'cuda' 或 'cpu'，None 时自动选择

    返回:
        (net, device) : 模型对象和设备字符串

    可用预训练权重（来源：https://huggingface.co/HoyerChou/SemiUHPE）:
        DAD-COCOHead-ResNet50-best.pth    395 MB  -- 推荐，精度与速度均衡
        DAD-COCOHead-EffNetV2-S-best.pth  336 MB
        DAD-WildHead-EffNetV2-S-best.pth  336 MB  -- 野外场景更强
        DAD-COCOHead-RepVGG-best.pth      719 MB
    """
    import torch

    setup_semiuhpe_path(semiuhpe_root)

    # 解析权重路径
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(semiuhpe_root, weights_path)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"权重文件不存在: {weights_path}\n"
            "请先运行: python calibrate_from_frontal.py --mode download_weights"
        )

    # 自动选择设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[模型] 使用设备: {device}")

    # 构造最小 config 对象（避免触发 configargparse 的命令行解析）
    class MinimalConfig:
        num_classes = 9

    config = MinimalConfig()
    config.network = network

    from src.networks import get_network
    net = get_network(config)

    # 加载权重
    ckpt = torch.load(weights_path, map_location=device)
    # SemiUHPE checkpoint 结构：{'net': ..., 'ema_net': ..., 'clock': ..., ...}
    if isinstance(ckpt, dict) and 'ema_net' in ckpt:
        state_dict = ckpt['ema_net']   # 优先使用 EMA 权重（更稳定）
    elif isinstance(ckpt, dict) and 'net' in ckpt:
        state_dict = ckpt['net']
    else:
        state_dict = ckpt

    # 处理 DataParallel 前缀
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
    对单张裁剪好的头部图片做推理，返回旋转矩阵 R（3x3 numpy array）。

    参数:
        img_bgr : BGR 格式 numpy 图像（已裁剪为头部区域，任意分辨率）
        net     : 已加载的 SemiUHPE 模型
        device  : 'cuda' 或 'cpu'

    返回:
        R : 3x3 旋转矩阵（numpy array）

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
    img_tensor = img_tensor.unsqueeze(0)   # (1, 3, 224, 224)
    if device == 'cuda':
        img_tensor = img_tensor.cuda()

    with torch.no_grad():
        fisher_out = net(img_tensor)             # (1, 9)
        pd_m = batch_torch_A_to_R(fisher_out)   # (1, 3, 3)

    return pd_m.detach().cpu().numpy()[0]        # (3, 3)


def rotation_matrix_to_euler_semiuhpe(R, full_range=False):
    """
    将旋转矩阵转换为 SemiUHPE 约定的欧拉角（度）。

    输出顺序：(pitch, yaw, roll)
    旋转顺序：XYZ（x=pitch, y=yaw, z=roll）
    来源：src/utils.py::rot_euler_6DRepNet（SemiUHPE 官方实现）

    参数:
        R          : 3x3 旋转矩阵（numpy array）
        full_range : True 时 Yaw 范围扩展到 (-180, 180)，默认 False 即 (-90, 90)

    返回:
        (pitch, yaw, roll) : 欧拉角（度）
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if R[0, 0] < 0 and full_range:
        sy = -sy

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0

    return math.degrees(x), math.degrees(y), math.degrees(z)


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


def compensate_extrinsic_R(R_camera, R_extrinsic):
    """
    将相机坐标系旋转矩阵补偿到世界坐标系。

    公式：R_world = R_extrinsic @ R_camera

    参数:
        R_camera    : 模型预测的旋转矩阵（3x3 numpy array）
        R_extrinsic : 外参旋转矩阵（3x3 numpy array）

    返回:
        R_world : 世界坐标系旋转矩阵（3x3 numpy array）
    """
    return R_extrinsic @ R_camera


def compensate_extrinsic_euler(pitch, yaw, roll, R_extrinsic, full_range=False):
    """
    将 SemiUHPE 输出的欧拉角（pitch, yaw, roll，度）补偿到世界坐标系。

    参数:
        pitch, yaw, roll : SemiUHPE 预测的欧拉角（度），XYZ 顺序
        R_extrinsic      : 外参旋转矩阵（3x3 numpy array）
        full_range       : 是否使用全角度范围

    返回:
        (pitch_world, yaw_world, roll_world) : 世界坐标系欧拉角（度）
    """
    R_camera = Rotation.from_euler('xyz', [pitch, yaw, roll], degrees=True).as_matrix()
    R_world = compensate_extrinsic_R(R_camera, R_extrinsic)
    return rotation_matrix_to_euler_semiuhpe(R_world, full_range)


# =============================================================================
# Step 0：下载预训练权重
# =============================================================================

def download_weights(semiuhpe_root, model_name='DAD-COCOHead-ResNet50-best.pth'):
    """
    从 HuggingFace (HoyerChou/SemiUHPE) 下载预训练权重到 semiuhpe_root/weights/ 目录。

    可用模型：
        DAD-COCOHead-ResNet50-best.pth    395 MB  -- 推荐（--network resnet50）
        DAD-COCOHead-EffNetV2-S-best.pth  336 MB  -- (--network effinetv2)
        DAD-WildHead-EffNetV2-S-best.pth  336 MB  -- 野外场景更强 (--network effinetv2)
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

def calibrate(image_paths, net, device, output_path='extrinsic.npy', full_range=False):
    """
    从正视前方图片估计相机外参旋转矩阵。

    参数:
        image_paths : 正视前方图片路径列表（建议 5-20 张，越多越稳定）
        net         : 已加载的 SemiUHPE 模型
        device      : 'cuda' 或 'cpu'
        output_path : 外参矩阵保存路径（.npy）
        full_range  : 是否使用全角度范围（默认 False，适合驾驶场景）

    返回:
        R_extrinsic : 3x3 外参旋转矩阵

    原理：
        正视前方时 R_world = I，因此 R_extrinsic = R_camera。
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
        pitch, yaw, roll = rotation_matrix_to_euler_semiuhpe(R, full_range)
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

    pitch_e, yaw_e, roll_e = rotation_matrix_to_euler_semiuhpe(R_extrinsic, full_range)
    print(f"\n[结果] 相机安装角度估计（SO(3) 均值）:")
    print(f"  Pitch = {pitch_e:+.2f}  (相机偏上/下)")
    print(f"  Yaw   = {yaw_e:+.2f}  (相机偏左/右，A柱侧视预期约 +-30~45 deg)")
    print(f"  Roll  = {roll_e:+.2f}  (相机旋转)")

    np.save(output_path, R_extrinsic)
    print(f"\n[完成] 外参矩阵已保存: {output_path}")
    return R_extrinsic


# =============================================================================
# Step 2：验证补偿效果
# =============================================================================

def verify(image_paths, net, device, R_extrinsic, full_range=False):
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
        full_range  : 是否使用全角度范围

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

        R_cam = predict_rotation_matrix(img, net, device)
        pitch, yaw, roll = rotation_matrix_to_euler_semiuhpe(R_cam, full_range)

        R_world = compensate_extrinsic_R(R_cam, R_extrinsic)
        pitch_w, yaw_w, roll_w = rotation_matrix_to_euler_semiuhpe(R_world, full_range)

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

def compensate_csv(label_csv, R_extrinsic, output_csv, full_range=False,
                   pitch_col='pitch', yaw_col='yaw', roll_col='roll'):
    """
    批量对伪标签 CSV 文件做外参补偿。

    CSV 格式示例（支持任意其他列，只处理 pitch/yaw/roll 三列）：
        image_path,pitch,yaw,roll
        img001.jpg,-8.5,12.3,1.2
        img002.jpg,-9.0,-5.1,0.8

    注意：CSV 中的欧拉角应为 SemiUHPE 输出的 (pitch, yaw, roll)，单位：度。

    参数:
        label_csv   : 输入 CSV 路径
        R_extrinsic : 外参旋转矩阵（由 calibrate 模式生成）
        output_csv  : 输出 CSV 路径
        full_range  : 是否使用全角度范围
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
        pitch_w, yaw_w, roll_w = compensate_extrinsic_euler(
            row[pitch_col], row[yaw_col], row[roll_col], R_extrinsic, full_range
        )
        pitch_w_list.append(pitch_w)
        yaw_w_list.append(yaw_w)
        roll_w_list.append(roll_w)

    # 原始列保留（加 _cam 后缀），新列为世界坐标系
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

  # 估计外参
  python calibrate_from_frontal.py --mode calibrate \\
      --images frontal1.jpg frontal2.jpg frontal3.jpg \\
      --semiuhpe_root ./SemiUHPE \\
      --weights weights/DAD-COCOHead-ResNet50-best.pth \\
      --output extrinsic.npy

  # 验证补偿效果
  python calibrate_from_frontal.py --mode verify \\
      --images look_left.jpg look_right.jpg \\
      --semiuhpe_root ./SemiUHPE \\
      --weights weights/DAD-COCOHead-ResNet50-best.pth \\
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
                        default='weights/DAD-COCOHead-ResNet50-best.pth',
                        help='权重文件路径（相对于 semiuhpe_root 或绝对路径）')
    parser.add_argument('--network', default='resnet50',
                        choices=['resnet50', 'effinetv2', 'repvgg', 'resnet18', 'mobilenet'],
                        help='骨干网络（需与权重匹配，默认 resnet50）')
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
    parser.add_argument('--full_range', action='store_true',
                        help='使用全角度范围（+-180 deg），默认 False（适合驾驶场景 +-90 deg）')
    parser.add_argument('--pitch_col', default='pitch', help='CSV 中 pitch 列名')
    parser.add_argument('--yaw_col',   default='yaw',   help='CSV 中 yaw 列名')
    parser.add_argument('--roll_col',  default='roll',  help='CSV 中 roll 列名')
    parser.add_argument('--hf_model',
                        default='DAD-COCOHead-ResNet50-best.pth',
                        help='从 HuggingFace 下载的权重文件名（download_weights 模式使用）')

    args = parser.parse_args()

    # -- download_weights 模式 --
    if args.mode == 'download_weights':
        download_weights(args.semiuhpe_root, args.hf_model)
        return

    # -- 需要模型的模式（calibrate / verify）--
    if args.mode in ('calibrate', 'verify'):
        net, device = load_semiuhpe_model(
            args.semiuhpe_root, args.weights, args.network, args.device
        )

    if args.mode == 'calibrate':
        if not args.images:
            parser.error("calibrate 模式需要提供 --images")
        calibrate(args.images, net, device, args.output, args.full_range)

    elif args.mode == 'verify':
        if not args.images:
            parser.error("verify 模式需要提供 --images")
        if not os.path.exists(args.extrinsic):
            parser.error(f"外参文件不存在: {args.extrinsic}，请先运行 calibrate 模式")
        R_extrinsic = np.load(args.extrinsic)
        verify(args.images, net, device, R_extrinsic, args.full_range)

    elif args.mode == 'compensate':
        if args.label_csv is None:
            parser.error("compensate 模式需要提供 --label_csv")
        if not os.path.exists(args.extrinsic):
            parser.error(f"外参文件不存在: {args.extrinsic}，请先运行 calibrate 模式")
        R_extrinsic = np.load(args.extrinsic)
        compensate_csv(args.label_csv, R_extrinsic, args.output_csv,
                       args.full_range,
                       args.pitch_col, args.yaw_col, args.roll_col)


if __name__ == '__main__':
    main()
