"""
伪标签生成管线 (Pseudo-Label Generation Pipeline)

适用场景：A 柱或其他非正视安装的驾驶员监控摄像头（DMS）
核心思路：MediaPipe 关键点检测 → 透视矫正 → SOTA 模型推理 → 姿态逆变换

人脸中心计算方式：
    使用 MediaPipe FaceLandmarker 的 478 个 3D 面部关键点，
    取所有关键点的 2D 投影几何中心作为人脸中心。
    相比纯 BBox 中点，关键点中心更接近面部真实几何中心，
    对于侧脸、遮挡等情况更加鲁棒。

依赖安装：
    pip install mediapipe numpy opencv-python scipy
    # SemiUHPE: git clone https://github.com/hnuzhy/SemiUHPE.git
    # 权重下载: https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq
    # MediaPipe 模型: 运行脚本时会自动下载，或手动下载：
    #   wget -O face_landmarker.task \
    #     https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import os
import json
import argparse
import urllib.request


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def rotation_matrix_to_euler(R, convention='YXZ'):
    """
    旋转矩阵 → 欧拉角（度，YXZ 约定）
    返回: (pitch, yaw, roll) in degrees
    """
    r = Rotation.from_matrix(R)
    yaw, pitch, roll = r.as_euler(convention.lower(), degrees=True)
    return pitch, yaw, roll


def euler_to_rotation_matrix(pitch, yaw, roll=0.0, convention='YXZ'):
    """欧拉角（度，YXZ 约定）→ 旋转矩阵"""
    r = Rotation.from_euler(convention.lower(), [yaw, pitch, roll], degrees=True)
    return r.as_matrix()


def compute_rectification_homography(face_u, face_v, K):
    """
    计算透视矫正单应性矩阵。

    原理：
        人脸中心 (face_u, face_v) 偏离图像光轴中心 (cx, cy)，
        说明相机光轴未对准人脸，存在透视偏置。
        构造矫正旋转矩阵 R_rectify，将相机"虚拟地旋转"使光轴对准人脸，
        消除透视偏置引起的 Pitch/Yaw 漂移。
        H = K @ R_rectify @ K^{-1}

    Args:
        face_u (float): 人脸中心 u 坐标（像素），由关键点几何中心计算
        face_v (float): 人脸中心 v 坐标（像素），由关键点几何中心计算
        K (np.ndarray): 3x3 相机内参矩阵

    Returns:
        H (np.ndarray): 3x3 透视矫正单应性矩阵
        R_rectify (np.ndarray): 3x3 矫正旋转矩阵（用于后续姿态逆变换）
        yaw_bias_deg (float): 偏置 Yaw 角（度）
        pitch_bias_deg (float): 偏置 Pitch 角（度）
    """
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]

    dx = face_u - cx
    dy = face_v - cy

    # 偏置角：人脸中心偏离光轴的角度
    yaw_bias_deg   = np.degrees(np.arctan2(dx, fx))
    pitch_bias_deg = np.degrees(np.arctan2(dy, fy))

    # 矫正旋转矩阵：将相机旋转对准人脸（消除偏置）
    R_rectify = euler_to_rotation_matrix(-pitch_bias_deg, -yaw_bias_deg)

    # 单应性矩阵
    K_inv = np.linalg.inv(K)
    H = K @ R_rectify @ K_inv

    return H, R_rectify, yaw_bias_deg, pitch_bias_deg


def apply_rectification(img, H):
    """
    对图像应用透视矫正变换（warpPerspective）。

    Args:
        img (np.ndarray): 输入图像 (H, W, C)，RGB 格式
        H (np.ndarray): 3x3 单应性矩阵

    Returns:
        rectified_img (np.ndarray): 矫正后的图像
    """
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, H, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def compensate_pose(R_model, R_rectify):
    """
    将模型在矫正图像上预测的姿态，逆变换回真实相机坐标系。

    公式：R_final = R_rectify^{T} @ R_model
    注意：必须在旋转矩阵空间内操作，绝对不能直接对欧拉角加减！

    Args:
        R_model (np.ndarray): 3x3 模型预测的旋转矩阵（相对于矫正后虚拟相机）
        R_rectify (np.ndarray): 3x3 矫正旋转矩阵

    Returns:
        R_final (np.ndarray): 3x3 最终旋转矩阵（相对于真实相机坐标系）
    """
    return R_rectify.T @ R_model  # 正交矩阵的逆 = 转置


def transform_bbox_by_homography(bbox, H, img_w, img_h):
    """
    将 BBox 的四个角点通过单应性矩阵 H 变换到矫正后图像的坐标。

    Args:
        bbox (tuple): (x1, y1, x2, y2)
        H (np.ndarray): 3x3 单应性矩阵
        img_w, img_h (int): 图像宽高（用于裁剪边界）

    Returns:
        (rx1, ry1, rx2, ry2): 矫正后图像中的 BBox 坐标
    """
    x1, y1, x2, y2 = bbox
    corners = np.array([[x1, y1, 1], [x2, y1, 1],
                         [x1, y2, 1], [x2, y2, 1]], dtype=np.float64).T
    corners_rect = H @ corners
    corners_rect = corners_rect[:2] / corners_rect[2]
    rx1 = int(max(0, corners_rect[0].min()))
    ry1 = int(max(0, corners_rect[1].min()))
    rx2 = int(min(img_w, corners_rect[0].max()))
    ry2 = int(min(img_h, corners_rect[1].max()))
    return rx1, ry1, rx2, ry2


# ─────────────────────────────────────────────────────────────────────────────
# 相机内参估计
# ─────────────────────────────────────────────────────────────────────────────

def estimate_intrinsics(img, method='heuristic', focal_length=None):
    """
    估计相机内参矩阵 K。

    方法：
        'provided':  直接使用提供的焦距
        'heuristic': 启发式猜测（假设 FOV=60°），适合快速测试
        'geocalib':  使用 GeoCalib 深度学习模型估计（需要安装 geocalib）

    Args:
        img (np.ndarray): 输入图像（RGB）
        method (str): 估计方法
        focal_length (float): 焦距（仅 'provided' 方法使用）

    Returns:
        K (np.ndarray): 3x3 相机内参矩阵
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    if method == 'provided' and focal_length is not None:
        f = focal_length
        print(f"[内参] 使用提供的焦距: f={f:.1f}px")

    elif method == 'geocalib':
        try:
            import torch
            from geocalib import GeoCalib
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = GeoCalib().to(device)
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_t = img_t.unsqueeze(0).to(device)
            with torch.no_grad():
                result = model(img_t)
            f = result['camera'].f.item() * max(w, h)
            print(f"[内参] GeoCalib 估计: f={f:.1f}px")
        except ImportError:
            print("[警告] GeoCalib 未安装，回退到启发式猜测")
            fov_deg = 60.0
            f = (w / 2.0) / np.tan(np.radians(fov_deg / 2.0))

    else:  # heuristic
        fov_deg = 60.0
        f = (w / 2.0) / np.tan(np.radians(fov_deg / 2.0))
        print(f"[内参] 启发式猜测 (FOV=60°): f={f:.1f}px")

    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe FaceLandmarker 人脸检测器
# ─────────────────────────────────────────────────────────────────────────────

MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# 用于计算人脸中心的关键点子集（面部轮廓 + 鼻梁 + 眼眶）
# 这些点分布均匀，能更准确地代表面部几何中心
FACE_CENTER_LANDMARK_INDICES = [
    # 面部轮廓（oval）
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    # 鼻梁
    1, 2, 5, 4, 19, 94, 125,
    # 左眼眶
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    # 右眼眶
    362, 382, 381, 380, 374, 373, 390, 249, 263,
]


class MediaPipeFaceDetector:
    """
    基于 MediaPipe FaceLandmarker 的人脸检测器。

    人脸中心计算：
        使用 478 个面部关键点中的子集（面部轮廓 + 鼻梁 + 眼眶），
        取其 2D 投影坐标的几何均值作为人脸中心。
        相比 BBox 中点，该中心更接近面部真实几何中心，
        对侧脸和遮挡情况更鲁棒。

    Args:
        model_path (str): face_landmarker.task 模型文件路径
        num_faces (int): 最多检测的人脸数量
        min_detection_confidence (float): 最低检测置信度
    """

    def __init__(self, model_path='face_landmarker.task',
                 num_faces=1, min_detection_confidence=0.3):
        # 自动下载模型文件
        if not os.path.exists(model_path):
            print(f"[MediaPipe] 模型文件不存在，正在下载到 {model_path} ...")
            urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, model_path)
            print(f"[MediaPipe] 下载完成")

        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.mp = mp
        print(f"[MediaPipe] FaceLandmarker 初始化成功，模型: {model_path}")

    def detect(self, img_rgb):
        """
        检测图像中的人脸，返回 BBox 和关键点几何中心。

        Args:
            img_rgb (np.ndarray): RGB 格式图像

        Returns:
            list of dict，每个元素包含：
                'bbox'        : (x1, y1, x2, y2) 像素坐标
                'face_center' : (u, v) 关键点几何中心（像素）
                'landmarks'   : np.ndarray, shape=(N, 2)，所有关键点的像素坐标
                'score'       : 检测置信度
        """
        h, w = img_rgb.shape[:2]
        mp_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=img_rgb.astype(np.uint8)
        )
        result = self.landmarker.detect(mp_image)

        faces = []
        for face_lms in result.face_landmarks:
            # 所有 478 个关键点的像素坐标
            all_pts = np.array([[lm.x * w, lm.y * h] for lm in face_lms])

            # BBox：所有关键点的包围盒
            x1 = int(np.clip(all_pts[:, 0].min(), 0, w))
            y1 = int(np.clip(all_pts[:, 1].min(), 0, h))
            x2 = int(np.clip(all_pts[:, 0].max(), 0, w))
            y2 = int(np.clip(all_pts[:, 1].max(), 0, h))

            # 人脸中心：使用面部轮廓+鼻梁+眼眶关键点子集的几何均值
            # 这比 BBox 中点更接近面部真实几何中心
            center_indices = [i for i in FACE_CENTER_LANDMARK_INDICES
                              if i < len(face_lms)]
            center_pts = all_pts[center_indices]
            face_u = float(np.mean(center_pts[:, 0]))
            face_v = float(np.mean(center_pts[:, 1]))

            # 置信度：取关键点 z 坐标的反（z 越小表示越靠近相机，置信度越高）
            # MediaPipe 未直接提供置信度分数，用 1.0 代替
            faces.append({
                'bbox': (x1, y1, x2, y2),
                'face_center': (face_u, face_v),
                'landmarks': all_pts,
                'score': 1.0,
            })

        return faces


# ─────────────────────────────────────────────────────────────────────────────
# Head Pose 模型接口（SemiUHPE 适配器）
# ─────────────────────────────────────────────────────────────────────────────

class HeadPoseEstimator:
    """
    SemiUHPE Head Pose 估计器适配器。

    使用前请先克隆 SemiUHPE 并下载权重：
        git clone https://github.com/hnuzhy/SemiUHPE.git
        # 权重下载: https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq
        # 推荐: DAD-WildHead-EffNetV2-S-best.pth（野外无约束场景精度最高）

    将 semiuhpe_root 设置为克隆目录的路径。
    """

    def __init__(self, semiuhpe_root, weight_path):
        import sys
        sys.path.insert(0, semiuhpe_root)
        try:
            import torch
            from model import SemiUHPE as SemiUHPEModel
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = SemiUHPEModel()
            checkpoint = torch.load(weight_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.to(self.device)
            self.model.eval()
            self.torch = torch
            print(f"[Head Pose] SemiUHPE 加载成功，设备: {self.device}")
        except Exception as e:
            print(f"[错误] SemiUHPE 加载失败: {e}")
            raise

    def predict(self, face_crop_rgb):
        """
        对裁剪后的人脸图像预测 Head Pose。

        Args:
            face_crop_rgb (np.ndarray): RGB 格式的人脸裁剪图像

        Returns:
            R (np.ndarray): 3x3 旋转矩阵
            (pitch, yaw, roll): 欧拉角（度）
        """
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(face_crop_rgb).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            output = self.model(img_tensor)
        R = self._decode_6d_rotation(output)
        pitch, yaw, roll = rotation_matrix_to_euler(R)
        return R, (pitch, yaw, roll)

    def _decode_6d_rotation(self, output):
        """
        将 SemiUHPE 的 6D 旋转表示解码为旋转矩阵。
        具体实现参考 SemiUHPE 仓库中的 utils.py: compute_rotation_matrix_from_ortho6d()
        """
        # 6D 旋转解码（Zhou et al., CVPR 2019）
        # output shape: (B, 6)
        a1 = output[:, :3]
        a2 = output[:, 3:6]
        b1 = self.torch.nn.functional.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = self.torch.nn.functional.normalize(b2, dim=-1)
        b3 = self.torch.cross(b1, b2, dim=-1)
        R = self.torch.stack([b1, b2, b3], dim=-1)
        return R[0].cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 主管线
# ─────────────────────────────────────────────────────────────────────────────

def process_image(img_path, detector, estimator, K, output_dir, save_vis=False):
    """
    对单张图像执行完整的伪标签生成管线。

    完整步骤：
        1. 读取图像
        2. MediaPipe FaceLandmarker 检测人脸，获取关键点几何中心
        3. 利用关键点中心和内参 K 计算透视矫正单应性矩阵
        4. 对整图执行透视矫正（warpPerspective）
        5. 在矫正后的图像上裁剪人脸 BBox
        6. 送入 SemiUHPE 预测 Head Pose
        7. 姿态逆变换回真实相机坐标系（R_final = R_rectify^T @ R_model）
        8. 保存伪标签 JSON

    Args:
        img_path (str): 输入图像路径
        detector (MediaPipeFaceDetector): 人脸检测器
        estimator (HeadPoseEstimator): Head Pose 估计器（可为 None，仅测试矫正）
        K (np.ndarray): 3x3 相机内参矩阵
        output_dir (str): 伪标签输出目录
        save_vis (bool): 是否保存可视化图像

    Returns:
        list of dict: 每个人脸的伪标签结果
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"[跳过] 无法读取: {img_path}")
        return []
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_rgb.shape[:2]

    # Step 1: MediaPipe 人脸检测
    faces = detector.detect(img_rgb)
    if not faces:
        print(f"  未检测到人脸: {os.path.basename(img_path)}")
        return []

    results = []
    vis_img = img_rgb.copy() if save_vis else None

    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        face_u, face_v = face['face_center']  # 关键点几何中心

        # Step 2: 计算透视矫正单应性矩阵
        H, R_rectify, yaw_bias, pitch_bias = compute_rectification_homography(
            face_u, face_v, K
        )

        # Step 3: 对整图执行透视矫正
        rectified_img = apply_rectification(img_rgb, H)

        # Step 4: 将 BBox 变换到矫正后图像坐标
        rx1, ry1, rx2, ry2 = transform_bbox_by_homography(
            (x1, y1, x2, y2), H, img_w, img_h
        )
        if rx2 <= rx1 or ry2 <= ry1:
            continue

        face_crop = rectified_img[ry1:ry2, rx1:rx2]

        # Step 5: Head Pose 估计（在矫正后图像上）
        if estimator is not None:
            R_model, (pitch_model, yaw_model, roll_model) = estimator.predict(face_crop)

            # Step 6: 姿态逆变换回真实相机坐标系
            R_final = compensate_pose(R_model, R_rectify)
            pitch_final, yaw_final, roll_final = rotation_matrix_to_euler(R_final)
        else:
            pitch_model = yaw_model = roll_model = None
            pitch_final = yaw_final = roll_final = None
            R_final = None

        result = {
            'image': os.path.basename(img_path),
            'bbox': [x1, y1, x2, y2],
            'face_center_landmarks': [round(face_u, 2), round(face_v, 2)],
            'perspective_bias_deg': {
                'yaw': round(yaw_bias, 3),
                'pitch': round(pitch_bias, 3)
            },
            'pose_before_compensation': {
                'pitch': round(pitch_model, 3) if pitch_model is not None else None,
                'yaw': round(yaw_model, 3) if yaw_model is not None else None,
                'roll': round(roll_model, 3) if roll_model is not None else None,
            },
            'pose_final': {
                'pitch': round(pitch_final, 3) if pitch_final is not None else None,
                'yaw': round(yaw_final, 3) if yaw_final is not None else None,
                'roll': round(roll_final, 3) if roll_final is not None else None,
            },
            'detection_score': face['score'],
        }
        results.append(result)

        if estimator is not None:
            print(f"  人脸 BBox={face['bbox']}, "
                  f"关键点中心=({face_u:.1f},{face_v:.1f}), "
                  f"偏置(Yaw={yaw_bias:.1f}°,Pitch={pitch_bias:.1f}°), "
                  f"矫正前Pitch={pitch_model:.1f}°, "
                  f"矫正后Pitch={pitch_final:.1f}°")

        # 可视化
        if save_vis and vis_img is not None:
            # 绘制 BBox
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 100, 0), 2)
            # 绘制关键点中心（绿色圆点）
            cv2.circle(vis_img, (int(face_u), int(face_v)), 5, (0, 255, 0), -1)
            # 绘制图像光轴中心（蓝色十字）
            cx, cy = int(K[0, 2]), int(K[1, 2])
            cv2.drawMarker(vis_img, (cx, cy), (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 2)
            # 标注偏置角
            cv2.putText(vis_img,
                        f"bias: Yaw={yaw_bias:.1f} Pitch={pitch_bias:.1f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1)

    # 保存伪标签 JSON
    if results and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        label_path = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(img_path))[0] + '.json'
        )
        with open(label_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # 保存可视化图像
    if save_vis and vis_img is not None and output_dir:
        vis_dir = os.path.join(output_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, os.path.basename(img_path))
        cv2.imwrite(vis_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

    return results


def run_pipeline(image_dir, output_dir,
                 model_path='face_landmarker.task',
                 focal_length=None,
                 intrinsics_method='heuristic',
                 semiuhpe_root=None,
                 weight_path=None,
                 save_vis=False):
    """
    批量处理图像目录，生成伪标签。

    Args:
        image_dir (str): 输入图像目录
        output_dir (str): 伪标签输出目录
        model_path (str): MediaPipe face_landmarker.task 模型路径
        focal_length (float): 相机焦距（像素），提供则跳过内参估计
        intrinsics_method (str): 内参估计方法 ('heuristic' 或 'geocalib')
        semiuhpe_root (str): SemiUHPE 代码库根目录
        weight_path (str): SemiUHPE 权重文件路径
        save_vis (bool): 是否保存可视化图像
    """
    # 初始化 MediaPipe 人脸检测器
    detector = MediaPipeFaceDetector(model_path=model_path)

    # 初始化 Head Pose 估计器
    estimator = None
    if semiuhpe_root and weight_path:
        estimator = HeadPoseEstimator(semiuhpe_root, weight_path)
    else:
        print("[提示] 未提供 SemiUHPE 路径，将仅执行人脸检测和透视矫正步骤")

    # 获取图像列表
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    img_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(exts)
    ])
    print(f"共找到 {len(img_paths)} 张图像")

    # 内参矩阵（固定相机只需估计一次）
    K = None
    if focal_length:
        sample_img = cv2.imread(img_paths[0])
        h, w = sample_img.shape[:2]
        K = np.array([[focal_length, 0, w / 2.0],
                      [0, focal_length, h / 2.0],
                      [0, 0, 1]], dtype=np.float64)
        print(f"[内参] 使用提供的焦距: f={focal_length}px")

    all_results = []
    for i, img_path in enumerate(img_paths):
        print(f"[{i+1}/{len(img_paths)}] {os.path.basename(img_path)}")

        if K is None:
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            K = estimate_intrinsics(img_rgb,
                                    method=intrinsics_method,
                                    focal_length=focal_length)

        results = process_image(img_path, detector, estimator, K,
                                output_dir, save_vis=save_vis)
        all_results.extend(results)

    print(f"\n完成！共处理 {len(all_results)} 个人脸")
    if output_dir:
        print(f"伪标签保存至: {output_dir}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Head Pose Pseudo-Label Generation Pipeline (MediaPipe + SemiUHPE)"
    )
    parser.add_argument("--image_dir",    type=str, required=True,
                        help="输入图像目录")
    parser.add_argument("--output_dir",   type=str, default="./pseudo_labels",
                        help="伪标签输出目录")
    parser.add_argument("--model_path",   type=str, default="face_landmarker.task",
                        help="MediaPipe face_landmarker.task 模型路径")
    parser.add_argument("--focal_length", type=float, default=None,
                        help="相机焦距（像素），不提供则自动估计")
    parser.add_argument("--intrinsics_method", type=str, default="heuristic",
                        choices=["heuristic", "geocalib"],
                        help="内参估计方法（未提供焦距时使用）")
    parser.add_argument("--semiuhpe_root", type=str, default=None,
                        help="SemiUHPE 代码库根目录")
    parser.add_argument("--weight_path",  type=str, default=None,
                        help="SemiUHPE 权重文件路径")
    parser.add_argument("--save_vis",     action="store_true",
                        help="保存可视化图像（含关键点中心和偏置角标注）")
    args = parser.parse_args()

    run_pipeline(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        focal_length=args.focal_length,
        intrinsics_method=args.intrinsics_method,
        semiuhpe_root=args.semiuhpe_root,
        weight_path=args.weight_path,
        save_vis=args.save_vis,
    )
