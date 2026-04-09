"""
透视矫正可视化 Demo (Perspective Rectification Demo)

使用 MediaPipe FaceLandmarker 检测人脸关键点，
以关键点几何中心作为人脸中心，计算透视矫正单应性矩阵，
并可视化矫正前后的对比效果。

运行方式：
    python rectify_demo.py --image your_image.jpg
    python rectify_demo.py --image your_image.jpg --focal_length 1000
    python rectify_demo.py --image your_image.jpg --focal_length 1000 --output result.png
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import urllib.request
from scipy.spatial.transform import Rotation

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
                                    'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# 用于计算人脸中心的关键点子集（面部轮廓 + 鼻梁 + 眼眶）
FACE_CENTER_LANDMARK_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    1, 2, 5, 4, 19, 94, 125,
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    362, 382, 381, 380, 374, 373, 390, 249, 263,
]


def init_face_landmarker(model_path='face_landmarker.task'):
    """初始化 MediaPipe FaceLandmarker，如模型不存在则自动下载"""
    if not os.path.exists(model_path):
        print(f"[MediaPipe] 正在下载模型到 {model_path} ...")
        urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, model_path)
        print("[MediaPipe] 下载完成")

    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    return landmarker, mp


def detect_face(img_rgb, landmarker, mp_module):
    """
    使用 MediaPipe FaceLandmarker 检测人脸。

    人脸中心计算：
        取面部轮廓 + 鼻梁 + 眼眶关键点子集的 2D 投影几何均值，
        比 BBox 中点更接近面部真实几何中心，对侧脸更鲁棒。

    Returns:
        bbox (tuple): (x1, y1, x2, y2)
        face_center (tuple): (u, v) 关键点几何中心（像素）
        landmarks (np.ndarray): shape=(478, 2)，所有关键点像素坐标
        None, None, None 如果未检测到人脸
    """
    h, w = img_rgb.shape[:2]
    mp_image = mp_module.Image(
        image_format=mp_module.ImageFormat.SRGB,
        data=img_rgb.astype(np.uint8)
    )
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None, None, None

    face_lms = result.face_landmarks[0]
    all_pts = np.array([[lm.x * w, lm.y * h] for lm in face_lms])

    # BBox
    x1 = int(np.clip(all_pts[:, 0].min(), 0, w))
    y1 = int(np.clip(all_pts[:, 1].min(), 0, h))
    x2 = int(np.clip(all_pts[:, 0].max(), 0, w))
    y2 = int(np.clip(all_pts[:, 1].max(), 0, h))

    # 关键点几何中心（使用子集）
    center_indices = [i for i in FACE_CENTER_LANDMARK_INDICES if i < len(face_lms)]
    center_pts = all_pts[center_indices]
    face_u = float(np.mean(center_pts[:, 0]))
    face_v = float(np.mean(center_pts[:, 1]))

    return (x1, y1, x2, y2), (face_u, face_v), all_pts


def build_intrinsics(img, focal_length=None):
    """构建相机内参矩阵 K"""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    if focal_length:
        f = focal_length
        print(f"[内参] 使用提供的焦距: f={f:.1f}px")
    else:
        fov_deg = 60.0
        f = (w / 2.0) / np.tan(np.radians(fov_deg / 2.0))
        print(f"[内参] 启发式猜测 (FOV=60°): f={f:.1f}px")
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)


def compute_rectification(face_u, face_v, K):
    """
    计算透视矫正单应性矩阵。

    原理：
        人脸中心 (face_u, face_v) 偏离图像光轴中心 (cx, cy)，
        构造矫正旋转矩阵 R_rectify，将相机"虚拟旋转"使光轴对准人脸，
        消除透视偏置引起的 Pitch/Yaw 漂移。
        H = K @ R_rectify @ K^{-1}
    """
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    dx, dy = face_u - cx, face_v - cy

    yaw_bias   = np.degrees(np.arctan2(dx, fx))
    pitch_bias = np.degrees(np.arctan2(dy, fy))

    r = Rotation.from_euler('yxz', [-yaw_bias, -pitch_bias, 0], degrees=True)
    R_rectify = r.as_matrix()
    H = K @ R_rectify @ np.linalg.inv(K)

    return H, R_rectify, yaw_bias, pitch_bias


def main():
    parser = argparse.ArgumentParser(
        description="Perspective Rectification Demo using MediaPipe FaceLandmarker"
    )
    parser.add_argument("--image",        type=str, required=True,
                        help="输入图像路径")
    parser.add_argument("--focal_length", type=float, default=None,
                        help="相机焦距（像素），不提供则启发式猜测 FOV=60°")
    parser.add_argument("--model_path",   type=str, default="face_landmarker.task",
                        help="MediaPipe face_landmarker.task 模型路径")
    parser.add_argument("--output",       type=str, default="rectify_result.png",
                        help="输出可视化图像路径")
    args = parser.parse_args()

    # 读取图像
    img = cv2.imread(args.image)
    if img is None:
        print(f"[错误] 无法读取图像: {args.image}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    print(f"图像尺寸: {w}x{h}")

    # 初始化 MediaPipe
    landmarker, mp_module = init_face_landmarker(args.model_path)

    # 检测人脸
    bbox, face_center, landmarks = detect_face(img_rgb, landmarker, mp_module)
    if bbox is None:
        print("[错误] 未检测到人脸，请确认图像中包含正脸或侧脸")
        return

    x1, y1, x2, y2 = bbox
    face_u, face_v = face_center
    print(f"检测到人脸 BBox: {bbox}")
    print(f"关键点几何中心 (face_center): ({face_u:.1f}, {face_v:.1f})")
    print(f"关键点数量: {len(landmarks)}")

    # 构建内参矩阵
    K = build_intrinsics(img_rgb, args.focal_length)
    cx, cy = K[0, 2], K[1, 2]

    # 计算透视矫正
    H, R_rectify, yaw_bias, pitch_bias = compute_rectification(face_u, face_v, K)
    print(f"透视偏置: Yaw={yaw_bias:.2f}°, Pitch={pitch_bias:.2f}°")

    # 执行透视矫正
    rectified = cv2.warpPerspective(img_rgb, H, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)

    # 计算矫正后人脸中心的残余偏移
    pt_orig = np.array([[[face_u, face_v]]], dtype=np.float32)
    pt_rect = cv2.perspectiveTransform(pt_orig, H)[0][0]
    residual = np.sqrt((pt_rect[0] - cx)**2 + (pt_rect[1] - cy)**2)

    # ── 绘制原始图像标注 ──
    vis_orig = img_rgb.copy()
    # 所有关键点（浅蓝色小点）
    for pt in landmarks:
        cv2.circle(vis_orig, (int(pt[0]), int(pt[1])), 1, (180, 180, 255), -1)
    # 用于计算中心的关键点子集（橙色）
    center_indices = [i for i in FACE_CENTER_LANDMARK_INDICES if i < len(landmarks)]
    for idx in center_indices:
        pt = landmarks[idx]
        cv2.circle(vis_orig, (int(pt[0]), int(pt[1])), 2, (255, 140, 0), -1)
    # BBox
    cv2.rectangle(vis_orig, (x1, y1), (x2, y2), (255, 100, 0), 2)
    # 关键点几何中心（绿色）
    cv2.circle(vis_orig, (int(face_u), int(face_v)), 7, (0, 255, 0), -1)
    cv2.circle(vis_orig, (int(face_u), int(face_v)), 7, (255, 255, 255), 2)
    # 光轴中心（蓝色十字）
    cv2.drawMarker(vis_orig, (int(cx), int(cy)), (0, 100, 255),
                   cv2.MARKER_CROSS, 30, 2)
    # 偏置向量
    cv2.arrowedLine(vis_orig, (int(cx), int(cy)), (int(face_u), int(face_v)),
                    (255, 255, 0), 2, tipLength=0.15)

    # ── 绘制矫正后图像标注 ──
    vis_rect = rectified.copy()
    cv2.circle(vis_rect, (int(pt_rect[0]), int(pt_rect[1])), 7, (0, 255, 0), -1)
    cv2.circle(vis_rect, (int(pt_rect[0]), int(pt_rect[1])), 7, (255, 255, 255), 2)
    cv2.drawMarker(vis_rect, (int(cx), int(cy)), (0, 100, 255),
                   cv2.MARKER_CROSS, 30, 2)
    cv2.putText(vis_rect, f"residual={residual:.1f}px",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

    # ── 人脸裁剪对比 ──
    pad = 20
    x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
    x2p = min(w, x2 + pad); y2p = min(h, y2 + pad)
    face_orig_crop = img_rgb[y1p:y2p, x1p:x2p]
    face_rect_crop = rectified[y1p:y2p, x1p:x2p]

    # ── 组合图 ──
    fig = plt.figure(figsize=(18, 7), facecolor='#1a1a2e')

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(vis_orig)
    ax1.set_title(
        '原始图像\n'
        '● 绿点 = 关键点几何中心 (face_center)\n'
        '+ 蓝十字 = 图像光轴中心\n'
        '● 橙点 = 参与计算中心的关键点子集',
        color='white', fontsize=9
    )
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(vis_rect)
    ax2.set_title(
        f'透视矫正后\n'
        f'● 绿点 = 矫正后人脸中心\n'
        f'残余偏移: {residual:.1f}px（理想值 ≈ 0）',
        color='white', fontsize=9
    )
    ax2.axis('off')

    ax3 = fig.add_subplot(1, 3, 3)
    if face_orig_crop.size > 0 and face_rect_crop.size > 0:
        fh = max(face_orig_crop.shape[0], face_rect_crop.shape[0])
        fw = face_orig_crop.shape[1] + face_rect_crop.shape[1] + 8
        combined = np.zeros((fh, fw, 3), dtype=np.uint8)
        combined[:face_orig_crop.shape[0], :face_orig_crop.shape[1]] = face_orig_crop
        combined[:face_rect_crop.shape[0],
                 face_orig_crop.shape[1]+8:face_orig_crop.shape[1]+8+face_rect_crop.shape[1]] = face_rect_crop
        ax3.imshow(combined)
    ax3.set_title(
        f'人脸裁剪对比\n左: 原始  右: 矫正后\n'
        f'偏置 Yaw={yaw_bias:.1f}°  Pitch={pitch_bias:.1f}°',
        color='white', fontsize=9
    )
    ax3.axis('off')

    fig.suptitle(
        f'MediaPipe 关键点中心透视矫正效果\n'
        f'人脸中心偏置: Δu={face_u - cx:.1f}px, Δv={face_v - cy:.1f}px  →  '
        f'Yaw_bias={yaw_bias:.1f}°, Pitch_bias={pitch_bias:.1f}°',
        color='white', fontsize=12, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"\n可视化结果已保存: {args.output}")
    print(f"  人脸中心（关键点几何均值）: ({face_u:.1f}, {face_v:.1f})")
    print(f"  图像光轴中心: ({cx:.1f}, {cy:.1f})")
    print(f"  透视偏置: Yaw={yaw_bias:.2f}°, Pitch={pitch_bias:.2f}°")
    print(f"  矫正后残余偏移: {residual:.2f}px")


if __name__ == "__main__":
    main()
