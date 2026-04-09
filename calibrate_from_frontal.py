"""
calibrate_from_frontal.py
=========================
从"驾驶员正视前方"图片估计相机外参旋转矩阵，并验证补偿效果。

【使用场景】
  只有 A 柱相机视角下驾驶员正视前方的图片，没有视频，没有转头数据。

【核心原理】
  当驾驶员正视前方时，定义此时头部姿态为世界坐标系零点（R_world = I）。
  此时模型预测的 R_camera 就等于相机外参 R_extrinsic。
  估计出 R_extrinsic 后，对任意帧的预测结果做：
      R_world = R_extrinsic @ R_camera
  从 R_world 提取的欧拉角即为世界坐标系下的真实姿态，
  纯 Yaw 转头时 Pitch 将保持稳定。

【文件结构】
  calibrate_from_frontal.py   本脚本
  ├── Step 1: 从正视前方图片估计外参，保存 extrinsic.npy
  ├── Step 2: 对单张图片验证补偿效果
  └── Step 3: 批量对伪标签做外参补偿

【快速开始】
  # Step 1: 估计外参（支持单张或多张图片）
  python calibrate_from_frontal.py --mode calibrate \
      --images frontal1.jpg frontal2.jpg frontal3.jpg \
      --output extrinsic.npy

  # Step 2: 验证补偿效果（输入一张转头图片，看补偿后 Pitch 是否接近 0）
  python calibrate_from_frontal.py --mode verify \
      --images look_left.jpg look_right.jpg \
      --extrinsic extrinsic.npy

  # Step 3: 批量补偿伪标签 CSV 文件
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
import urllib.request

# ── 可选：matplotlib 用于可视化 ──
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ─────────────────────────────────────────────
# 旋转矩阵工具函数
# ─────────────────────────────────────────────

def euler_to_R(yaw, pitch, roll, convention='ZYX'):
    """
    欧拉角（度）→ 旋转矩阵
    convention: HopeNet 默认使用 'ZYX'（先 Yaw，再 Pitch，再 Roll）
    如果您的模型使用其他约定，请修改此处
    """
    return Rotation.from_euler(convention, [yaw, pitch, roll], degrees=True).as_matrix()


def R_to_euler(R, convention='ZYX'):
    """旋转矩阵 → 欧拉角（度），返回 (yaw, pitch, roll)"""
    euler = Rotation.from_matrix(R).as_euler(convention, degrees=True)
    return float(euler[0]), float(euler[1]), float(euler[2])


def average_rotation_matrices(R_list):
    """
    多个旋转矩阵的 SO(3) 均值（SVD 正交化投影）
    不能直接对矩阵元素取均值，必须用此方法
    """
    R_sum = np.sum(np.array(R_list), axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt
    return R_mean


def compensate_extrinsic(yaw, pitch, roll, R_extrinsic, convention='ZYX'):
    """
    将相机坐标系欧拉角补偿到世界坐标系
    
    参数:
        yaw, pitch, roll: 模型预测的欧拉角（度），相机坐标系
        R_extrinsic: 外参旋转矩阵（3x3 numpy array）
        convention: 欧拉角约定，默认 'ZYX'（HopeNet）
    
    返回:
        (yaw_world, pitch_world, roll_world): 世界坐标系欧拉角（度）
    """
    R_camera = euler_to_R(yaw, pitch, roll, convention)
    R_world = R_extrinsic @ R_camera
    return R_to_euler(R_world, convention)


# ─────────────────────────────────────────────
# 模型推理接口（需替换为您的实际代码）
# ─────────────────────────────────────────────

def load_model():
    """
    加载您的 HopeNet 模型。
    请将此函数替换为您的实际模型加载代码。
    
    返回: model 对象
    """
    # ── 替换示例（HopeNet）──
    # import torch
    # from hopenet import Hopenet
    # import torchvision.transforms as transforms
    #
    # model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    # model.load_state_dict(torch.load('hopenet_robust_alpha1.pkl',
    #                                   map_location='cpu'))
    # model.eval()
    # return model
    raise NotImplementedError(
        "\n[错误] 请在 load_model() 函数中替换为您的实际模型加载代码\n"
        "  参考注释中的 HopeNet 示例"
    )


def predict_euler_from_image(img_bgr, model, convention='ZYX'):
    """
    对单张图片做 Head Pose 推理，返回 (yaw, pitch, roll)（度）。
    请将此函数替换为您的实际推理代码。
    
    参数:
        img_bgr: BGR 格式的 numpy 图像（cv2 读取的格式）
        model: 您的模型对象
        convention: 欧拉角约定
    
    返回:
        (yaw, pitch, roll): 欧拉角（度）
    """
    # ── 替换示例（HopeNet）──
    # import torch
    # import torchvision.transforms as transforms
    # from PIL import Image
    #
    # transformations = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # img_pil = Image.fromarray(img_rgb)
    # img_tensor = transformations(img_pil).unsqueeze(0)
    #
    # with torch.no_grad():
    #     yaw_pred, pitch_pred, roll_pred = model(img_tensor)
    # idx_tensor = torch.FloatTensor(range(66))
    # yaw   = (torch.nn.functional.softmax(yaw_pred,   dim=1) * idx_tensor).sum(1).item() * 3 - 99
    # pitch = (torch.nn.functional.softmax(pitch_pred, dim=1) * idx_tensor).sum(1).item() * 3 - 99
    # roll  = (torch.nn.functional.softmax(roll_pred,  dim=1) * idx_tensor).sum(1).item() * 3 - 99
    # return yaw, pitch, roll
    raise NotImplementedError(
        "\n[错误] 请在 predict_euler_from_image() 函数中替换为您的实际推理代码\n"
        "  参考注释中的 HopeNet 示例"
    )


# ─────────────────────────────────────────────
# Step 1：从正视前方图片估计外参
# ─────────────────────────────────────────────

def calibrate(image_paths, model, convention='ZYX', output_path='extrinsic.npy'):
    """
    从正视前方图片估计相机外参旋转矩阵。
    
    参数:
        image_paths: 正视前方图片路径列表（越多越准，建议 5-20 张）
        model: 已加载的模型
        convention: 欧拉角约定
        output_path: 外参矩阵保存路径
    
    返回:
        R_extrinsic: 3x3 外参旋转矩阵
    """
    print(f"[Step 1] 从 {len(image_paths)} 张正视前方图片估计外参...")
    
    R_list = []
    results = []
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  [警告] 无法读取图片: {path}，跳过")
            continue
        
        yaw, pitch, roll = predict_euler_from_image(img, model, convention)
        R = euler_to_R(yaw, pitch, roll, convention)
        R_list.append(R)
        results.append((os.path.basename(path), yaw, pitch, roll))
        print(f"  {os.path.basename(path):30s}  Yaw={yaw:+7.2f}°  Pitch={pitch:+7.2f}°  Roll={roll:+7.2f}°")
    
    if len(R_list) == 0:
        raise ValueError("没有成功处理任何图片，请检查图片路径和模型")
    
    if len(R_list) == 1:
        R_extrinsic = R_list[0]
        print(f"\n[提示] 只有 1 张图片，建议提供更多图片以提高稳定性")
    else:
        R_extrinsic = average_rotation_matrices(R_list)
    
    # 打印估计结果
    yaw_e, pitch_e, roll_e = R_to_euler(R_extrinsic, convention)
    print(f"\n[结果] 相机安装角度估计（均值）:")
    print(f"  Yaw   = {yaw_e:+.2f}°  （相机偏左/右）")
    print(f"  Pitch = {pitch_e:+.2f}°  （相机偏上/下）")
    print(f"  Roll  = {roll_e:+.2f}°  （相机旋转）")
    
    # 保存
    np.save(output_path, R_extrinsic)
    print(f"\n[完成] 外参矩阵已保存到: {output_path}")
    
    return R_extrinsic


# ─────────────────────────────────────────────
# Step 2：验证补偿效果
# ─────────────────────────────────────────────

def verify(image_paths, model, R_extrinsic, convention='ZYX'):
    """
    对转头图片验证外参补偿效果。
    
    预期结果：
      - 补偿后 Pitch 应接近 0°（驾驶员没有低头/抬头）
      - 补偿后 Yaw 应反映驾驶员实际的转头角度
    """
    print(f"\n[Step 2] 验证外参补偿效果（共 {len(image_paths)} 张图片）...")
    print(f"\n{'图片':<30} {'Yaw(前)':>9} {'Pitch(前)':>10} {'Roll(前)':>9}"
          f" {'Yaw(后)':>9} {'Pitch(后)':>10} {'Roll(后)':>9}")
    print("-" * 90)
    
    rows = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  [警告] 无法读取: {path}")
            continue
        
        yaw, pitch, roll = predict_euler_from_image(img, model, convention)
        yaw_w, pitch_w, roll_w = compensate_extrinsic(yaw, pitch, roll, R_extrinsic, convention)
        
        name = os.path.basename(path)
        print(f"{name:<30} {yaw:>+9.2f}° {pitch:>+9.2f}° {roll:>+9.2f}°"
              f" {yaw_w:>+9.2f}° {pitch_w:>+9.2f}° {roll_w:>+9.2f}°")
        rows.append((name, yaw, pitch, roll, yaw_w, pitch_w, roll_w))
    
    if HAS_MATPLOTLIB and len(rows) > 1:
        names = [r[0] for r in rows]
        pitches_before = [r[2] for r in rows]
        pitches_after  = [r[5] for r in rows]
        
        fig, ax = plt.subplots(figsize=(max(8, len(rows)*1.2), 5))
        x = range(len(rows))
        ax.bar([i - 0.2 for i in x], pitches_before, width=0.4,
               label='Pitch before', color='tomato', alpha=0.8)
        ax.bar([i + 0.2 for i in x], pitches_after,  width=0.4,
               label='Pitch after',  color='steelblue', alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Pitch (degrees)')
        ax.set_title('Pitch Before / After Extrinsic Compensation')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        out = 'verify_result.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n[可视化] 对比图已保存: {out}")
    
    return rows


# ─────────────────────────────────────────────
# Step 3：批量补偿伪标签 CSV
# ─────────────────────────────────────────────

def compensate_csv(label_csv, R_extrinsic, output_csv, convention='ZYX',
                   yaw_col='yaw', pitch_col='pitch', roll_col='roll'):
    """
    批量对伪标签 CSV 文件做外参补偿。
    
    CSV 格式示例（支持任意其他列，只处理 yaw/pitch/roll 三列）：
        image_path,yaw,pitch,roll
        img001.jpg,12.3,-8.5,1.2
        img002.jpg,-5.1,-9.0,0.8
        ...
    
    参数:
        label_csv: 输入 CSV 路径
        R_extrinsic: 外参旋转矩阵
        output_csv: 输出 CSV 路径
        yaw_col, pitch_col, roll_col: CSV 中对应列名
    """
    try:
        import pandas as pd
    except ImportError:
        print("[错误] 请先安装 pandas: pip install pandas")
        sys.exit(1)
    
    print(f"\n[Step 3] 批量补偿伪标签: {label_csv}")
    df = pd.read_csv(label_csv)
    
    required = [yaw_col, pitch_col, roll_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV 中未找到列 '{col}'，当前列: {list(df.columns)}")
    
    yaw_w_list, pitch_w_list, roll_w_list = [], [], []
    
    for _, row in df.iterrows():
        yaw_w, pitch_w, roll_w = compensate_extrinsic(
            row[yaw_col], row[pitch_col], row[roll_col], R_extrinsic, convention
        )
        yaw_w_list.append(yaw_w)
        pitch_w_list.append(pitch_w)
        roll_w_list.append(roll_w)
    
    # 原始列保留（加 _cam 后缀），新列为世界坐标系
    df[f'{yaw_col}_cam']   = df[yaw_col]
    df[f'{pitch_col}_cam'] = df[pitch_col]
    df[f'{roll_col}_cam']  = df[roll_col]
    df[yaw_col]   = yaw_w_list
    df[pitch_col] = pitch_w_list
    df[roll_col]  = roll_w_list
    
    df.to_csv(output_csv, index=False)
    
    print(f"  处理行数: {len(df)}")
    print(f"  Pitch 补偿前: mean={df[f'{pitch_col}_cam'].mean():.2f}°, "
          f"std={df[f'{pitch_col}_cam'].std():.2f}°")
    print(f"  Pitch 补偿后: mean={df[pitch_col].mean():.2f}°, "
          f"std={df[pitch_col].std():.2f}°")
    print(f"[完成] 补偿后标签已保存: {output_csv}")


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='从正视前方图片估计相机外参，并对伪标签做坐标系补偿',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # Step 1: 估计外参
  python calibrate_from_frontal.py --mode calibrate \\
      --images frontal1.jpg frontal2.jpg frontal3.jpg \\
      --output extrinsic.npy

  # Step 2: 验证补偿效果
  python calibrate_from_frontal.py --mode verify \\
      --images look_left.jpg look_right.jpg \\
      --extrinsic extrinsic.npy

  # Step 3: 批量补偿伪标签 CSV
  python calibrate_from_frontal.py --mode compensate \\
      --label_csv pseudo_labels.csv \\
      --extrinsic extrinsic.npy \\
      --output_csv pseudo_labels_compensated.csv
        """
    )
    parser.add_argument('--mode', choices=['calibrate', 'verify', 'compensate'],
                        required=True,
                        help='运行模式: calibrate=估计外参, verify=验证效果, compensate=批量补偿')
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
    parser.add_argument('--convention', default='ZYX',
                        help='欧拉角约定（默认 ZYX，对应 HopeNet）')
    parser.add_argument('--yaw_col',   default='yaw',   help='CSV 中 yaw 列名')
    parser.add_argument('--pitch_col', default='pitch', help='CSV 中 pitch 列名')
    parser.add_argument('--roll_col',  default='roll',  help='CSV 中 roll 列名')
    
    args = parser.parse_args()

    # 加载模型（calibrate/verify 模式需要）
    model = None
    if args.mode in ('calibrate', 'verify'):
        model = load_model()

    if args.mode == 'calibrate':
        if not args.images:
            parser.error("calibrate 模式需要提供 --images")
        calibrate(args.images, model, args.convention, args.output)

    elif args.mode == 'verify':
        if not args.images:
            parser.error("verify 模式需要提供 --images")
        if not os.path.exists(args.extrinsic):
            parser.error(f"外参文件不存在: {args.extrinsic}，请先运行 calibrate 模式")
        R_extrinsic = np.load(args.extrinsic)
        verify(args.images, model, R_extrinsic, args.convention)

    elif args.mode == 'compensate':
        if args.label_csv is None:
            parser.error("compensate 模式需要提供 --label_csv")
        if not os.path.exists(args.extrinsic):
            parser.error(f"外参文件不存在: {args.extrinsic}，请先运行 calibrate 模式")
        R_extrinsic = np.load(args.extrinsic)
        compensate_csv(args.label_csv, R_extrinsic, args.output_csv,
                       args.convention, args.yaw_col, args.pitch_col, args.roll_col)


if __name__ == '__main__':
    main()
