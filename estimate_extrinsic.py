"""
estimate_extrinsic.py
=====================
相机外参旋转矩阵估计工具

【核心思想】
相机外参 R_extrinsic 描述了"相机坐标系相对于世界坐标系的旋转"。
当驾驶员头部处于"世界坐标系零姿态"时，模型预测的 R_camera 就等于 R_extrinsic。

支持三种采集方式：
  方法 A：驾驶员正视前方视频（最推荐，操作最简单）
  方法 B：利用水平尺约束估计 Roll 分量（辅助验证）
  方法 C：最小化 Pitch 方差优化（无需任何先验，自动搜索）

用法示例：
  # 方法 A：从正视前方视频估计外参
  python estimate_extrinsic.py --method A \
      --video frontal.mp4 \
      --model_type hopenet \
      --output extrinsic.npy

  # 方法 C：从任意驾驶视频自动估计（驾驶员做各种 Yaw 运动）
  python estimate_extrinsic.py --method C \
      --video driving.mp4 \
      --output extrinsic.npy

  # 验证外参效果
  python estimate_extrinsic.py --verify \
      --video driving.mp4 \
      --extrinsic extrinsic.npy
"""

import argparse
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import os
import urllib.request


# ─────────────────────────────────────────────
# MediaPipe 人脸关键点检测
# ─────────────────────────────────────────────

def get_face_landmarker(model_path="face_landmarker.task"):
    if not os.path.exists(model_path):
        print(f"[INFO] 下载 MediaPipe FaceLandmarker 模型...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def detect_face_center(frame_rgb, landmarker):
    """返回人脸关键点几何中心 (u, v)，检测失败返回 None"""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    h, w = frame_rgb.shape[:2]
    lms = result.face_landmarks[0]
    # 取全部 478 个关键点的均值作为几何中心
    us = [lm.x * w for lm in lms]
    vs = [lm.y * h for lm in lms]
    return np.array([np.mean(us), np.mean(vs)])


# ─────────────────────────────────────────────
# 模型推理接口（HopeNet / SemiUHPE 占位）
# ─────────────────────────────────────────────

def predict_euler(frame_bgr, model, model_type='hopenet'):
    """
    调用姿态估计模型，返回 (yaw, pitch, roll) 单位：度
    
    注意：这里是占位接口，需要根据您的实际模型替换。
    HopeNet 输出顺序通常是 [yaw, pitch, roll]，欧拉角约定 ZYX。
    """
    # ── 替换为您的实际模型推理代码 ──
    # 示例（HopeNet）：
    #   img_tensor = preprocess(frame_bgr)
    #   yaw, pitch, roll = model(img_tensor)
    #   return yaw.item(), pitch.item(), roll.item()
    raise NotImplementedError("请替换为您的实际模型推理代码")


# ─────────────────────────────────────────────
# 旋转矩阵工具函数
# ─────────────────────────────────────────────

def euler_to_R(yaw, pitch, roll, convention='ZYX'):
    """欧拉角（度）→ 旋转矩阵，约定 ZYX（HopeNet 默认）"""
    return Rotation.from_euler(convention, [yaw, pitch, roll], degrees=True).as_matrix()


def R_to_euler(R, convention='ZYX'):
    """旋转矩阵 → 欧拉角（度），返回 (yaw, pitch, roll)"""
    euler = Rotation.from_matrix(R).as_euler(convention, degrees=True)
    return float(euler[0]), float(euler[1]), float(euler[2])


def average_rotation_matrices(R_list):
    """
    计算多个旋转矩阵的均值（SVD 正交化投影到 SO(3)）
    这是旋转矩阵均值的标准做法，不能直接对矩阵元素取均值后使用
    """
    R_sum = np.sum(R_list, axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    # 确保行列式为 +1（右手坐标系）
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt
    return R_mean


def compensate_extrinsic(yaw, pitch, roll, R_extrinsic, convention='ZYX'):
    """
    将相机坐标系欧拉角补偿到世界坐标系
    R_world = R_extrinsic @ R_camera
    """
    R_camera = euler_to_R(yaw, pitch, roll, convention)
    R_world = R_extrinsic @ R_camera
    return R_to_euler(R_world, convention)


# ─────────────────────────────────────────────
# 方法 A：正视前方视频估计外参
# ─────────────────────────────────────────────

def method_A(video_path, model, model_type, max_frames=200, skip=5):
    """
    【方法 A：正视前方视频】
    
    操作要求：
      - 驾驶员坐在车内，头部保持水平，正视前方（Yaw≈0, Pitch≈0, Roll≈0）
      - 录制 10-30 秒视频，期间可以有轻微晃动（会被均值平滑掉）
      - 不需要完全静止，但不要做明显的转头动作
    
    原理：
      当 R_world = I（正视前方）时，R_camera = R_extrinsic
      对多帧取旋转矩阵均值，得到稳定的 R_extrinsic
    """
    print(f"[方法 A] 从正视前方视频估计外参: {video_path}")
    cap = cv2.VideoCapture(video_path)
    landmarker = get_face_landmarker()
    
    R_list = []
    frame_idx = 0
    
    while len(R_list) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % skip != 0:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_center = detect_face_center(frame_rgb, landmarker)
        if face_center is None:
            continue
        
        try:
            yaw, pitch, roll = predict_euler(frame, model, model_type)
        except NotImplementedError:
            raise
        
        R = euler_to_R(yaw, pitch, roll)
        R_list.append(R)
        
        if len(R_list) % 20 == 0:
            print(f"  已收集 {len(R_list)} 帧...")
    
    cap.release()
    
    if len(R_list) < 10:
        raise ValueError(f"有效帧数不足（{len(R_list)}），请检查视频质量")
    
    print(f"[方法 A] 共收集 {len(R_list)} 帧，计算旋转矩阵均值...")
    R_extrinsic = average_rotation_matrices(np.array(R_list))
    
    # 打印估计出的相机安装角度（便于验证）
    yaw_e, pitch_e, roll_e = R_to_euler(R_extrinsic)
    print(f"[方法 A] 估计的相机安装角度: Yaw={yaw_e:.1f}°, Pitch={pitch_e:.1f}°, Roll={roll_e:.1f}°")
    
    return R_extrinsic


# ─────────────────────────────────────────────
# 方法 C：最小化 Pitch 方差自动优化（无需先验）
# ─────────────────────────────────────────────

def method_C(video_path, model, model_type, max_frames=500, skip=3):
    """
    【方法 C：最小化 Pitch 方差优化】
    
    操作要求：
      - 驾驶员做各种 Yaw 运动（左看右看），Pitch 保持不变
      - 录制 30-60 秒视频
      - 不需要正视前方，不需要任何先验知识
    
    原理：
      正确的 R_extrinsic 补偿后，纯 Yaw 运动时 Pitch 应该保持不变（方差最小）
      通过优化 R_extrinsic 的三个欧拉角参数，最小化补偿后 Pitch 的方差
    
    注意：
      这个方法假设视频中 Pitch 的真实变化很小（驾驶员主要做 Yaw 运动）
      如果视频中有大量真实的 Pitch 变化，估计会不准
    """
    print(f"[方法 C] 从任意驾驶视频自动估计外参: {video_path}")
    cap = cv2.VideoCapture(video_path)
    landmarker = get_face_landmarker()
    
    # 先收集所有帧的预测结果
    predictions = []
    frame_idx = 0
    
    while len(predictions) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % skip != 0:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if detect_face_center(frame_rgb, landmarker) is None:
            continue
        
        try:
            yaw, pitch, roll = predict_euler(frame, model, model_type)
            predictions.append((yaw, pitch, roll))
        except NotImplementedError:
            raise
        
        if len(predictions) % 50 == 0:
            print(f"  已收集 {len(predictions)} 帧...")
    
    cap.release()
    print(f"[方法 C] 共收集 {len(predictions)} 帧，开始优化...")
    
    R_camera_list = [euler_to_R(y, p, r) for y, p, r in predictions]
    
    def pitch_variance(params):
        """目标函数：补偿后 Pitch 的方差（越小越好）"""
        yaw_e, pitch_e, roll_e = params
        R_ext = euler_to_R(yaw_e, pitch_e, roll_e)
        pitches = []
        for R_cam in R_camera_list:
            R_world = R_ext @ R_cam
            _, p, _ = R_to_euler(R_world)
            pitches.append(p)
        return np.var(pitches)
    
    # 初始值：零外参（无补偿）
    x0 = [0.0, 0.0, 0.0]
    result = minimize(pitch_variance, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 0.1, 'fatol': 0.01})
    
    yaw_e, pitch_e, roll_e = result.x
    R_extrinsic = euler_to_R(yaw_e, pitch_e, roll_e)
    
    print(f"[方法 C] 优化完成: Yaw={yaw_e:.1f}°, Pitch={pitch_e:.1f}°, Roll={roll_e:.1f}°")
    print(f"[方法 C] 补偿前 Pitch 方差: {np.var([p for _, p, _ in predictions]):.2f}°²")
    print(f"[方法 C] 补偿后 Pitch 方差: {result.fun:.2f}°²（越小越好）")
    
    return R_extrinsic


# ─────────────────────────────────────────────
# 验证外参效果
# ─────────────────────────────────────────────

def verify_extrinsic(video_path, model, model_type, R_extrinsic, max_frames=300, skip=3):
    """
    可视化验证外参补偿效果：
    绘制补偿前后的 Pitch 曲线，应该能看到补偿后曲线更平坦
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cap = cv2.VideoCapture(video_path)
    landmarker = get_face_landmarker()
    
    pitches_before = []
    pitches_after = []
    yaws_before = []
    frame_idx = 0
    
    while len(pitches_before) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % skip != 0:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if detect_face_center(frame_rgb, landmarker) is None:
            continue
        
        try:
            yaw, pitch, roll = predict_euler(frame, model, model_type)
        except NotImplementedError:
            raise
        
        yaw_w, pitch_w, roll_w = compensate_extrinsic(yaw, pitch, roll, R_extrinsic)
        
        pitches_before.append(pitch)
        pitches_after.append(pitch_w)
        yaws_before.append(yaw)
    
    cap.release()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    frames = range(len(pitches_before))
    
    axes[0].plot(frames, pitches_before, 'r-', alpha=0.7, label='Pitch (before)')
    axes[0].plot(frames, yaws_before, 'b-', alpha=0.5, label='Yaw (before)')
    axes[0].axhline(y=np.mean(pitches_before), color='r', linestyle='--', alpha=0.5,
                    label=f'Pitch mean={np.mean(pitches_before):.1f}°, std={np.std(pitches_before):.1f}°')
    axes[0].set_title('Before Extrinsic Compensation')
    axes[0].set_ylabel('Angle (degrees)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(frames, pitches_after, 'g-', alpha=0.7, label='Pitch (after)')
    axes[1].plot(frames, yaws_before, 'b-', alpha=0.5, label='Yaw (before, reference)')
    axes[1].axhline(y=np.mean(pitches_after), color='g', linestyle='--', alpha=0.5,
                    label=f'Pitch mean={np.mean(pitches_after):.1f}°, std={np.std(pitches_after):.1f}°')
    axes[1].set_title('After Extrinsic Compensation')
    axes[1].set_ylabel('Angle (degrees)')
    axes[1].set_xlabel('Frame')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = 'extrinsic_verify.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n[验证结果]")
    print(f"  补偿前 Pitch: mean={np.mean(pitches_before):.1f}°, std={np.std(pitches_before):.1f}°, "
          f"range=[{np.min(pitches_before):.1f}, {np.max(pitches_before):.1f}]°")
    print(f"  补偿后 Pitch: mean={np.mean(pitches_after):.1f}°, std={np.std(pitches_after):.1f}°, "
          f"range=[{np.min(pitches_after):.1f}, {np.max(pitches_after):.1f}]°")
    print(f"  Pitch 方差降低: {np.std(pitches_before):.1f}° → {np.std(pitches_after):.1f}°")
    print(f"  验证图已保存: {out_path}")
    
    return pitches_before, pitches_after


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='相机外参旋转矩阵估计工具')
    parser.add_argument('--method', choices=['A', 'C'], default='A',
                        help='估计方法: A=正视前方视频, C=最小化Pitch方差优化')
    parser.add_argument('--video', required=True, help='输入视频路径')
    parser.add_argument('--model_type', default='hopenet', help='模型类型: hopenet / semiuhpe')
    parser.add_argument('--output', default='extrinsic.npy', help='外参矩阵保存路径')
    parser.add_argument('--verify', action='store_true', help='验证外参补偿效果')
    parser.add_argument('--extrinsic', default=None, help='已有外参矩阵路径（用于验证）')
    args = parser.parse_args()

    # ── 加载您的模型 ──
    # 示例（HopeNet）：
    #   import torch
    #   model = HopeNet(...)
    #   model.load_state_dict(torch.load('hopenet.pkl'))
    #   model.eval()
    model = None  # 替换为实际模型

    if args.verify:
        if args.extrinsic is None:
            raise ValueError("验证模式需要提供 --extrinsic 路径")
        R_extrinsic = np.load(args.extrinsic)
        verify_extrinsic(args.video, model, args.model_type, R_extrinsic)
        return

    if args.method == 'A':
        R_extrinsic = method_A(args.video, model, args.model_type)
    elif args.method == 'C':
        R_extrinsic = method_C(args.video, model, args.model_type)

    np.save(args.output, R_extrinsic)
    print(f"\n[完成] 外参矩阵已保存到: {args.output}")
    print(f"后续使用方式：")
    print(f"  R_extrinsic = np.load('{args.output}')")
    print(f"  yaw_w, pitch_w, roll_w = compensate_extrinsic(yaw, pitch, roll, R_extrinsic)")


if __name__ == '__main__':
    main()
