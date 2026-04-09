"""
透视矫正可视化 Demo (Perspective Rectification Demo)

使用 MediaPipe FaceLandmarker 检测人脸关键点，
以关键点几何中心作为人脸中心，计算透视矫正单应性矩阵，
并可视化矫正前后的对比效果。

【旋转矩阵推导】
  目标：warpPerspective 后，人脸中心从 (face_u, face_v) 移动到 (cx, cy)。
  cv2.warpPerspective(src, H, dsize) 的工作方式：
    dst[p'] = src[H^{-1} @ p']，即 H 是"src → dst"的正向映射。
  因此需要 H @ [face_u, face_v, 1]^T ≈ [cx, cy, 1]^T。
  令 H = K @ R @ K^{-1}，则 R 需满足：
    R @ d = [0, 0, 1]^T，其中 d = K^{-1} @ [face_u, face_v, 1]^T（人脸方向向量）。
  使用 Rodrigues 轴角旋转，将 d 旋转到光轴方向 z=[0,0,1]，无欧拉角耦合误差。

【内参估计方式】
  1. --focal_length 手动指定（最准确，推荐标定后使用）
  2. --intrinsics geocalib：调用 GeoCalib 模型从图像估计焦距
  3. 默认：启发式猜测 FOV=60°

运行方式：
    python rectify_demo.py --image your_image.jpg
    python rectify_demo.py --image your_image.jpg --focal_length 1000
    python rectify_demo.py --image your_image.jpg --intrinsics geocalib
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


# ─────────────────────────────────────────────────────────────
# 1. MediaPipe 人脸检测
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# 2. 相机内参估计
# ─────────────────────────────────────────────────────────────

def estimate_intrinsics_geocalib(img_rgb):
    """
    使用 GeoCalib (ECCV 2024) 从单张图像估计相机内参。

    GeoCalib 通过分析图像中的透视线索（消失点、地平线等）
    预测归一化焦距 f_norm，再乘以 max(w, h) 得到像素焦距。

    安装：
        git clone https://github.com/cvg/GeoCalib.git
        pip install -e ./GeoCalib

    Returns:
        K (np.ndarray): 3x3 相机内参矩阵
        focal_px (float): 估计的像素焦距
    """
    try:
        import torch
        from geocalib import GeoCalib

        h, w = img_rgb.shape[:2]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = GeoCalib().to(device)
        model.eval()

        # GeoCalib 输入：float32 tensor，shape (1, 3, H, W)，值域 [0, 1]
        img_tensor = torch.from_numpy(img_rgb).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            result = model.calibrate(img_tensor)

        # result['camera'] 包含 f (归一化焦距，相对于 max(H,W))
        # 参考：https://github.com/cvg/GeoCalib
        camera = result['camera']
        # 归一化焦距 → 像素焦距
        f_norm = camera.f.item()          # 相对于 max(w, h) 的归一化值
        focal_px = f_norm * max(w, h)
        cx, cy = w / 2.0, h / 2.0

        K = np.array([[focal_px, 0, cx],
                      [0, focal_px, cy],
                      [0, 0, 1]], dtype=np.float64)

        print(f"[GeoCalib] 估计焦距: f_norm={f_norm:.4f}, focal_px={focal_px:.1f}px")
        return K, focal_px

    except ImportError:
        print("[GeoCalib] 未安装，回退到启发式猜测。安装方式：")
        print("  git clone https://github.com/cvg/GeoCalib.git && pip install -e ./GeoCalib")
        return None, None
    except Exception as e:
        print(f"[GeoCalib] 估计失败: {e}，回退到启发式猜测")
        return None, None


def build_intrinsics(img, focal_length=None, use_geocalib=False):
    """
    构建相机内参矩阵 K。

    优先级：
        1. focal_length 手动指定（最准确）
        2. use_geocalib=True → 调用 GeoCalib 模型估计
        3. 启发式猜测 FOV=60°（保底）
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    if focal_length is not None:
        f = focal_length
        print(f"[内参] 使用手动指定焦距: f={f:.1f}px")
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

    if use_geocalib:
        img_rgb = img if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        K, focal_px = estimate_intrinsics_geocalib(img_rgb)
        if K is not None:
            return K
        # GeoCalib 失败，回退

    # 启发式猜测：假设水平 FOV = 60°
    fov_deg = 60.0
    f = (w / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    print(f"[内参] 启发式猜测 (FOV=60°): f={f:.1f}px  "
          f"[建议：使用 --focal_length 或 --intrinsics geocalib 提高精度]")
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)


# ─────────────────────────────────────────────────────────────
# 3. 透视矫正（Rodrigues 旋转，无欧拉角耦合误差）
# ─────────────────────────────────────────────────────────────

def compute_rectification(face_u, face_v, K):
    """
    计算透视矫正单应性矩阵 H 和对应的旋转矩阵 R_rectify。

    推导：
        H = K @ R @ K^{-1}
        要求 H @ [face_u, face_v, 1]^T ≈ [cx, cy, 1]^T
        等价于 R @ d = [0, 0, 1]^T，其中 d = K^{-1} @ [face_u, face_v, 1]^T

        使用 Rodrigues 轴角旋转将 d 旋转到光轴方向 z=[0,0,1]：
            axis = d × z / |d × z|
            angle = arccos(d · z)
            R = Rodrigues(axis, angle)

        此方法无欧拉角分解的耦合误差，对任意偏置角均精确。

    Returns:
        H (np.ndarray): 3x3 单应性矩阵（用于 cv2.warpPerspective）
        R_rectify (np.ndarray): 3x3 旋转矩阵
        R_rectify_inv (np.ndarray): R_rectify 的逆（用于姿态逆变换）
        yaw_bias (float): 水平偏置角（度）
        pitch_bias (float): 垂直偏置角（度）
    """
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    K_inv = np.linalg.inv(K)

    # 人脸方向向量（相机坐标系中的 3D 方向）
    d = K_inv @ np.array([face_u, face_v, 1.0])
    d = d / np.linalg.norm(d)

    # 目标方向：光轴 z
    z = np.array([0.0, 0.0, 1.0])

    # Rodrigues 旋转：将 d 旋转到 z
    axis = np.cross(d, z)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-6:
        # d 已经接近光轴，无需旋转
        R_rectify = np.eye(3)
    else:
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(d, z), -1.0, 1.0))
        # Rodrigues 公式
        K_mat = np.array([
            [0,       -axis[2],  axis[1]],
            [axis[2],  0,       -axis[0]],
            [-axis[1], axis[0],  0      ]
        ])
        R_rectify = (np.eye(3) + np.sin(angle) * K_mat
                     + (1 - np.cos(angle)) * (K_mat @ K_mat))

    H = K @ R_rectify @ K_inv
    R_rectify_inv = R_rectify.T  # 旋转矩阵的逆 = 转置

    # 偏置角（仅用于显示，不参与旋转计算）
    yaw_bias   = np.degrees(np.arctan2(face_u - cx, fx))
    pitch_bias = np.degrees(np.arctan2(face_v - cy, fy))

    return H, R_rectify, R_rectify_inv, yaw_bias, pitch_bias


# ─────────────────────────────────────────────────────────────
# 4. 主函数
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Perspective Rectification Demo using MediaPipe FaceLandmarker"
    )
    parser.add_argument("--image",        type=str, required=True,
                        help="输入图像路径")
    parser.add_argument("--focal_length", type=float, default=None,
                        help="相机焦距（像素）。不提供则由 --intrinsics 决定")
    parser.add_argument("--intrinsics",   type=str, default="heuristic",
                        choices=["heuristic", "geocalib"],
                        help="内参估计方式：heuristic（启发式FOV=60°）或 geocalib（模型估计）")
    parser.add_argument("--model_path",   type=str, default="face_landmarker.task",
                        help="MediaPipe face_landmarker.task 模型路径")
    parser.add_argument("--output",       type=str, default="rectify_result.png",
                        help="输出可视化图像路径")
    args = parser.parse_args()

    # ── 读取图像 ──
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"[错误] 无法读取图像: {args.image}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    print(f"图像尺寸: {w}x{h}")

    # ── 初始化 MediaPipe ──
    landmarker, mp_module = init_face_landmarker(args.model_path)

    # ── 检测人脸 ──
    bbox, face_center, landmarks = detect_face(img_rgb, landmarker, mp_module)
    if bbox is None:
        print("[错误] 未检测到人脸，请确认图像中包含正脸或侧脸")
        return

    x1, y1, x2, y2 = bbox
    face_u, face_v = face_center
    print(f"检测到人脸 BBox: ({x1},{y1})-({x2},{y2})")
    print(f"关键点几何中心 (face_center): ({face_u:.1f}, {face_v:.1f})")

    # ── 相机内参估计 ──
    use_geocalib = (args.intrinsics == "geocalib")
    K = build_intrinsics(img_rgb, args.focal_length, use_geocalib=use_geocalib)
    cx, cy = K[0, 2], K[1, 2]

    # ── 计算透视矫正（Rodrigues 旋转）──
    H, R_rectify, R_rectify_inv, yaw_bias, pitch_bias = compute_rectification(
        face_u, face_v, K
    )
    print(f"透视偏置: Yaw={yaw_bias:.2f}°, Pitch={pitch_bias:.2f}°")

    # 验证：H @ face_center 应接近图像中心
    pt_h = H @ np.array([face_u, face_v, 1.0])
    pt_h = pt_h[:2] / pt_h[2]
    residual_verify = np.sqrt((pt_h[0] - cx)**2 + (pt_h[1] - cy)**2)
    print(f"矫正公式验证: H @ face_center = ({pt_h[0]:.1f},{pt_h[1]:.1f}), "
          f"期望({cx:.0f},{cy:.0f}), 残余={residual_verify:.2f}px")

    # ── 执行透视矫正 ──
    rectified = cv2.warpPerspective(img_rgb, H, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)

    # 矫正后人脸中心的实际位置（通过 perspectiveTransform）
    pt_orig = np.array([[[face_u, face_v]]], dtype=np.float32)
    pt_rect = cv2.perspectiveTransform(pt_orig, H)[0][0]
    residual = np.sqrt((pt_rect[0] - cx)**2 + (pt_rect[1] - cy)**2)

    # ── 绘制原始图像标注 ──
    vis_orig = img_rgb.copy()
    for pt in landmarks:
        cv2.circle(vis_orig, (int(pt[0]), int(pt[1])), 1, (180, 180, 255), -1)
    center_indices = [i for i in FACE_CENTER_LANDMARK_INDICES if i < len(landmarks)]
    for idx in center_indices:
        pt = landmarks[idx]
        cv2.circle(vis_orig, (int(pt[0]), int(pt[1])), 2, (255, 140, 0), -1)
    cv2.rectangle(vis_orig, (x1, y1), (x2, y2), (255, 100, 0), 2)
    cv2.circle(vis_orig, (int(face_u), int(face_v)), 7, (0, 255, 0), -1)
    cv2.circle(vis_orig, (int(face_u), int(face_v)), 7, (255, 255, 255), 2)
    cv2.drawMarker(vis_orig, (int(cx), int(cy)), (0, 100, 255),
                   cv2.MARKER_CROSS, 30, 2)
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
    # 矫正后人脸区域：以矫正后人脸中心为基准裁剪
    rcu, rcv = int(pt_rect[0]), int(pt_rect[1])
    bw, bh = x2p - x1p, y2p - y1p
    rx1 = max(0, rcu - bw // 2); ry1 = max(0, rcv - bh // 2)
    rx2 = min(w, rx1 + bw);      ry2 = min(h, ry1 + bh)
    face_rect_crop = rectified[ry1:ry2, rx1:rx2]

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
        f'透视矫正后（Rodrigues 旋转）\n'
        f'● 绿点 = 矫正后人脸中心\n'
        f'残余偏移: {residual:.1f}px（理想值 ≈ 0）',
        color='white', fontsize=9
    )
    ax2.axis('off')

    ax3 = fig.add_subplot(1, 3, 3)
    if face_orig_crop.size > 0 and face_rect_crop.size > 0:
        fh_c = max(face_orig_crop.shape[0], face_rect_crop.shape[0])
        fw_c = face_orig_crop.shape[1] + face_rect_crop.shape[1] + 8
        combined = np.zeros((fh_c, fw_c, 3), dtype=np.uint8)
        combined[:face_orig_crop.shape[0], :face_orig_crop.shape[1]] = face_orig_crop
        combined[:face_rect_crop.shape[0],
                 face_orig_crop.shape[1]+8:
                 face_orig_crop.shape[1]+8+face_rect_crop.shape[1]] = face_rect_crop
        ax3.imshow(combined)
    ax3.set_title(
        f'人脸裁剪对比\n左: 原始  右: 矫正后\n'
        f'偏置 Yaw={yaw_bias:.1f}°  Pitch={pitch_bias:.1f}°',
        color='white', fontsize=9
    )
    ax3.axis('off')

    intrinsics_method = (f"GeoCalib f={K[0,0]:.0f}px" if use_geocalib
                         else f"f={K[0,0]:.0f}px (FOV=60°)")
    fig.suptitle(
        f'MediaPipe 关键点中心透视矫正效果  [内参: {intrinsics_method}]\n'
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
    print(f"\n  R_rectify_inv（姿态逆变换矩阵，用于 R_final = R_rectify_inv @ R_model）:")
    print(f"  {R_rectify_inv}")


if __name__ == "__main__":
    main()
