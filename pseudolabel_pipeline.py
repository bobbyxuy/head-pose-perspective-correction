"""
伪标签生成管线 (Pseudo-Label Generation Pipeline)

适用场景：A 柱或其他非正视安装的驾驶员监控摄像头（DMS）
核心思路：透视矫正 → SOTA 模型推理 → 姿态逆变换

依赖安装：
    pip install numpy opencv-python scipy
    pip install insightface onnxruntime-gpu
    # SemiUHPE: git clone https://github.com/hnuzhy/SemiUHPE.git
    # 权重下载: https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import os
import json
import argparse


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def rotation_matrix_to_euler(R, convention='YXZ'):
    """
    将旋转矩阵转换为欧拉角（度）。
    convention: 欧拉角约定，默认 YXZ（Yaw-Pitch-Roll）
    返回: (pitch, yaw, roll) in degrees
    """
    r = Rotation.from_matrix(R)
    angles = r.as_euler(convention.lower(), degrees=True)
    # YXZ 约定下: angles = [yaw, pitch, roll]
    yaw, pitch, roll = angles
    return pitch, yaw, roll


def euler_to_rotation_matrix(pitch, yaw, roll, convention='YXZ'):
    """
    将欧拉角（度）转换为旋转矩阵。
    """
    r = Rotation.from_euler(convention.lower(), [yaw, pitch, roll], degrees=True)
    return r.as_matrix()


def compute_rectification_homography(face_u, face_v, K):
    """
    计算透视矫正单应性矩阵。

    原理：
        人脸中心 (face_u, face_v) 偏离图像光轴中心 (cx, cy)，
        这意味着相机光轴并未对准人脸。
        我们构造一个旋转矩阵 R_rectify，将相机"虚拟地旋转"，
        使光轴对准人脸中心，从而消除透视偏置。
        单应性矩阵 H = K @ R_rectify @ K_inv

    Args:
        face_u (float): 人脸中心的 u 坐标（像素）
        face_v (float): 人脸中心的 v 坐标（像素）
        K (np.ndarray): 3x3 相机内参矩阵

    Returns:
        H (np.ndarray): 3x3 透视矫正单应性矩阵
        R_rectify (np.ndarray): 3x3 矫正旋转矩阵（用于后续逆变换）
        yaw_bias (float): 偏置 Yaw 角（度）
        pitch_bias (float): 偏置 Pitch 角（度）
    """
    cx, cy = K[0, 2], K[1, 2]
    f = K[0, 0]  # 假设 fx = fy

    dx = face_u - cx
    dy = face_v - cy

    yaw_bias = np.degrees(np.arctan2(dx, f))
    pitch_bias = np.degrees(np.arctan2(dy, f))

    # 构造矫正旋转矩阵（将相机旋转对准人脸）
    R_rectify = euler_to_rotation_matrix(-pitch_bias, -yaw_bias, 0)

    # 计算单应性矩阵
    K_inv = np.linalg.inv(K)
    H = K @ R_rectify @ K_inv

    return H, R_rectify, yaw_bias, pitch_bias


def apply_rectification(img, H):
    """
    对图像应用透视矫正变换。

    Args:
        img (np.ndarray): 输入图像 (H, W, C)
        H (np.ndarray): 3x3 单应性矩阵

    Returns:
        rectified_img (np.ndarray): 矫正后的图像
    """
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, H, (w, h))


def compensate_pose(R_model, R_rectify):
    """
    将模型在矫正图像上预测的姿态，逆变换回真实相机坐标系。

    公式：R_final = R_rectify^{-1} @ R_model
    注意：必须在旋转矩阵空间内操作，不能直接对欧拉角加减！

    Args:
        R_model (np.ndarray): 3x3 模型预测的旋转矩阵（相对于矫正后的虚拟相机）
        R_rectify (np.ndarray): 3x3 矫正旋转矩阵

    Returns:
        R_final (np.ndarray): 3x3 最终旋转矩阵（相对于真实相机坐标系）
    """
    R_rectify_inv = R_rectify.T  # 正交矩阵的逆等于其转置
    R_final = R_rectify_inv @ R_model
    return R_final


# ─────────────────────────────────────────────────────────────────────────────
# 相机内参估计（可选）
# ─────────────────────────────────────────────────────────────────────────────

def estimate_intrinsics_from_image(img, method='heuristic'):
    """
    估计相机内参矩阵 K。

    方法选择：
        'heuristic': 启发式猜测（假设 FOV=60°），适合快速测试
        'geocalib':  使用 GeoCalib 深度学习模型估计（需要安装 geocalib）

    Args:
        img (np.ndarray): 输入图像
        method (str): 估计方法

    Returns:
        K (np.ndarray): 3x3 相机内参矩阵
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    if method == 'heuristic':
        # 启发式猜测：假设水平 FOV 约 60°
        fov_deg = 60.0
        f = (w / 2.0) / np.tan(np.radians(fov_deg / 2.0))
        print(f"[内参估计] 启发式猜测: f={f:.1f}px, cx={cx:.1f}, cy={cy:.1f}")

    elif method == 'geocalib':
        try:
            import torch
            from geocalib import GeoCalib
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = GeoCalib().to(device)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                result = model(img_tensor)
            f_norm = result['camera'].f.item()
            f = f_norm * max(w, h)
            print(f"[内参估计] GeoCalib: f={f:.1f}px, cx={cx:.1f}, cy={cy:.1f}")
        except ImportError:
            print("[警告] GeoCalib 未安装，回退到启发式猜测")
            fov_deg = 60.0
            f = (w / 2.0) / np.tan(np.radians(fov_deg / 2.0))

    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float64)
    return K


# ─────────────────────────────────────────────────────────────────────────────
# 人脸检测（使用 InsightFace RetinaFace）
# ─────────────────────────────────────────────────────────────────────────────

class FaceDetector:
    """
    使用 InsightFace 的 RetinaFace 进行人脸检测。

    安装：pip install insightface onnxruntime-gpu
    """

    def __init__(self, det_size=(640, 640)):
        try:
            import insightface
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'])
            self.app.prepare(ctx_id=0, det_size=det_size)
            self.available = True
            print("[人脸检测] InsightFace 初始化成功")
        except ImportError:
            print("[警告] InsightFace 未安装，将使用 OpenCV Haar 级联分类器作为备选")
            self.available = False
            self.cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

    def detect(self, img_rgb):
        """
        检测图像中的人脸。

        Args:
            img_rgb (np.ndarray): RGB 格式图像

        Returns:
            list of dict: 每个人脸包含 'bbox' (x1,y1,x2,y2) 和 'score'
        """
        if self.available:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            faces = self.app.get(img_bgr)
            results = []
            for face in faces:
                bbox = face.bbox.astype(int)
                results.append({
                    'bbox': (bbox[0], bbox[1], bbox[2], bbox[3]),
                    'score': float(face.det_score)
                })
            return results
        else:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            faces = self.cascade.detectMultiScale(gray, 1.1, 4)
            results = []
            for (x, y, w, h) in faces:
                results.append({
                    'bbox': (x, y, x + w, y + h),
                    'score': 1.0
                })
            return results


# ─────────────────────────────────────────────────────────────────────────────
# Head Pose 模型接口（SemiUHPE 适配器）
# ─────────────────────────────────────────────────────────────────────────────

class HeadPoseEstimator:
    """
    SemiUHPE Head Pose 估计器适配器。

    使用前请先克隆 SemiUHPE 并下载权重：
        git clone https://github.com/hnuzhy/SemiUHPE.git
        # 权重下载: https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq
        # 推荐: DAD-WildHead-EffNetV2-S-best.pth

    将 semiuhpe_root 设置为克隆目录的路径。
    """

    def __init__(self, semiuhpe_root, weight_path):
        import sys
        sys.path.insert(0, semiuhpe_root)
        try:
            import torch
            # SemiUHPE 的模型加载方式（根据其官方代码调整）
            from model import SemiUHPE as SemiUHPEModel
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = SemiUHPEModel()
            checkpoint = torch.load(weight_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.to(self.device)
            self.model.eval()
            self.torch = torch
            print(f"[Head Pose] SemiUHPE 加载成功，使用设备: {self.device}")
        except Exception as e:
            print(f"[错误] SemiUHPE 加载失败: {e}")
            print("请确认已克隆 SemiUHPE 并下载权重，参考 README.md")
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
        # SemiUHPE 输出 6D 旋转表示，需转换为旋转矩阵
        # 具体转换方式参考 SemiUHPE 官方代码
        R = self._decode_rotation(output)
        pitch, yaw, roll = rotation_matrix_to_euler(R)
        return R, (pitch, yaw, roll)

    def _decode_rotation(self, output):
        """解码 SemiUHPE 的 6D 旋转输出为旋转矩阵（参考官方代码）"""
        # 此处为示意，具体实现参考 SemiUHPE 仓库中的 utils.py
        raise NotImplementedError("请参考 SemiUHPE 官方代码实现此方法")


# ─────────────────────────────────────────────────────────────────────────────
# 主管线
# ─────────────────────────────────────────────────────────────────────────────

def process_image(img_path, detector, estimator, K, output_dir):
    """
    对单张图像执行完整的伪标签生成管线。

    步骤：
        1. 读取图像
        2. 人脸检测
        3. 计算透视矫正单应性矩阵
        4. 对整图执行透视矫正
        5. 在矫正后的图像上裁剪人脸 BBox
        6. 送入 Head Pose 模型预测
        7. 姿态逆变换回真实相机坐标系
        8. 保存伪标签

    Args:
        img_path (str): 输入图像路径
        detector (FaceDetector): 人脸检测器
        estimator (HeadPoseEstimator): Head Pose 估计器
        K (np.ndarray): 3x3 相机内参矩阵
        output_dir (str): 伪标签输出目录

    Returns:
        list of dict: 每个人脸的伪标签结果
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"[跳过] 无法读取: {img_path}")
        return []
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 1: 人脸检测（在原图上检测）
    faces = detector.detect(img_rgb)
    if not faces:
        return []

    results = []
    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        face_u = (x1 + x2) / 2.0
        face_v = (y1 + y2) / 2.0

        # Step 2: 计算透视矫正单应性矩阵
        H, R_rectify, yaw_bias, pitch_bias = compute_rectification_homography(face_u, face_v, K)

        # Step 3: 对整图执行透视矫正
        rectified_img = apply_rectification(img_rgb, H)

        # Step 4: 在矫正后的图像上裁剪人脸 BBox
        # 将原始 BBox 的四个角点通过 H 变换到矫正后图像的坐标
        corners = np.array([[x1, y1, 1], [x2, y1, 1],
                             [x1, y2, 1], [x2, y2, 1]], dtype=np.float64).T
        corners_rect = H @ corners
        corners_rect = corners_rect[:2] / corners_rect[2]
        rx1 = int(max(0, corners_rect[0].min()))
        ry1 = int(max(0, corners_rect[1].min()))
        rx2 = int(min(rectified_img.shape[1], corners_rect[0].max()))
        ry2 = int(min(rectified_img.shape[0], corners_rect[1].max()))

        if rx2 <= rx1 or ry2 <= ry1:
            continue

        face_crop = rectified_img[ry1:ry2, rx1:rx2]

        # Step 5: Head Pose 估计
        R_model, (pitch_model, yaw_model, roll_model) = estimator.predict(face_crop)

        # Step 6: 姿态逆变换回真实相机坐标系
        # 注意：必须在旋转矩阵空间内操作！
        R_final = compensate_pose(R_model, R_rectify)
        pitch_final, yaw_final, roll_final = rotation_matrix_to_euler(R_final)

        result = {
            'image': os.path.basename(img_path),
            'bbox': [x1, y1, x2, y2],
            'face_center': [face_u, face_v],
            'perspective_bias': {'yaw': yaw_bias, 'pitch': pitch_bias},
            'pose_before_compensation': {
                'pitch': pitch_model, 'yaw': yaw_model, 'roll': roll_model
            },
            'pose_final': {
                'pitch': pitch_final, 'yaw': yaw_final, 'roll': roll_final
            },
            'detection_score': face['score']
        }
        results.append(result)
        print(f"  人脸: bbox={face['bbox']}, "
              f"矫正前 Pitch={pitch_model:.1f}°, "
              f"矫正后 Pitch={pitch_final:.1f}°")

    # 保存伪标签
    if results and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        label_path = os.path.join(output_dir,
                                  os.path.splitext(os.path.basename(img_path))[0] + '.json')
        with open(label_path, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def run_pipeline(image_dir, output_dir, focal_length=None, intrinsics_method='heuristic',
                 semiuhpe_root=None, weight_path=None):
    """
    批量处理图像目录，生成伪标签。

    Args:
        image_dir (str): 输入图像目录
        output_dir (str): 伪标签输出目录
        focal_length (float): 相机焦距（可选，如果提供则跳过内参估计）
        intrinsics_method (str): 内参估计方法 ('heuristic' 或 'geocalib')
        semiuhpe_root (str): SemiUHPE 代码库根目录
        weight_path (str): SemiUHPE 权重文件路径
    """
    # 初始化检测器
    detector = FaceDetector()

    # 初始化 Head Pose 估计器
    if semiuhpe_root and weight_path:
        estimator = HeadPoseEstimator(semiuhpe_root, weight_path)
    else:
        print("[警告] 未提供 SemiUHPE 路径，管线将跳过 Head Pose 估计步骤")
        estimator = None

    # 获取图像列表
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    img_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                 if f.lower().endswith(exts)]
    print(f"共找到 {len(img_paths)} 张图像")

    # 估计内参（如果未提供焦距，对第一张图像估计一次即可）
    K = None
    if focal_length:
        # 使用提供的焦距，主点默认为图像中心
        sample_img = cv2.imread(img_paths[0])
        h, w = sample_img.shape[:2]
        K = np.array([[focal_length, 0, w / 2],
                      [0, focal_length, h / 2],
                      [0, 0, 1]], dtype=np.float64)
        print(f"[内参] 使用提供的焦距: f={focal_length}px")

    all_results = []
    for i, img_path in enumerate(img_paths):
        print(f"[{i+1}/{len(img_paths)}] 处理: {os.path.basename(img_path)}")

        # 如果未提供焦距，对每张图估计内参（或只估计一次）
        if K is None:
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            K = estimate_intrinsics_from_image(img_rgb, method=intrinsics_method)

        if estimator:
            results = process_image(img_path, detector, estimator, K, output_dir)
            all_results.extend(results)

    print(f"\n完成！共处理 {len(all_results)} 个人脸，伪标签保存至: {output_dir}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Head Pose Pseudo-Label Generation Pipeline")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="输入图像目录")
    parser.add_argument("--output_dir", type=str, default="./pseudo_labels",
                        help="伪标签输出目录")
    parser.add_argument("--focal_length", type=float, default=None,
                        help="相机焦距（像素），如不提供则自动估计")
    parser.add_argument("--intrinsics_method", type=str, default="heuristic",
                        choices=["heuristic", "geocalib"],
                        help="内参估计方法")
    parser.add_argument("--semiuhpe_root", type=str, default=None,
                        help="SemiUHPE 代码库根目录")
    parser.add_argument("--weight_path", type=str, default=None,
                        help="SemiUHPE 权重文件路径")
    args = parser.parse_args()

    run_pipeline(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        focal_length=args.focal_length,
        intrinsics_method=args.intrinsics_method,
        semiuhpe_root=args.semiuhpe_root,
        weight_path=args.weight_path
    )
