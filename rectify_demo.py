import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_rotation_matrix_from_euler(pitch, yaw, roll):
    """根据欧拉角(度)计算旋转矩阵 (YXZ 约定)"""
    p = np.radians(pitch)
    y = np.radians(yaw)
    r = np.radians(roll)
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(p), -np.sin(p)],
                   [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0],
                   [np.sin(r), np.cos(r), 0],
                   [0, 0, 1]])
    return Ry @ Rx @ Rz

def rectify_image(img_path, focal_length=None):
    # 1. 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # 2. 假设相机内参 (如果未提供，默认焦距为图像宽度)
    f = focal_length if focal_length else w
    cx, cy = w / 2, h / 2
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])

    # 3. 假设人脸中心位置 (简单示例，实际应由检测器提供)
    # 这里我们假设人脸在图像左侧 (A柱视角特征)
    face_u, face_v = w * 0.25, h * 0.5 
    
    # 4. 计算透视偏置角
    dx = face_u - cx
    dy = face_v - cy
    yaw_bias = np.degrees(np.arctan2(dx, f))
    pitch_bias = np.degrees(np.arctan2(dy, f))
    print(f"人脸偏离中心: dx={dx:.1f}px, dy={dy:.1f}px")
    print(f"对应透视偏置角: Yaw={yaw_bias:.1f}°, Pitch={pitch_bias:.1f}°")

    # 5. 计算透视矫正单应性矩阵
    # 构造一个旋转矩阵，将相机"旋转"对准人脸
    R_rectify = get_rotation_matrix_from_euler(-pitch_bias, -yaw_bias, 0)
    # 单应性矩阵 H = K * R * K^-1
    H = K @ R_rectify @ np.linalg.inv(K)

    # 6. 执行透视变换
    rectified_img = cv2.warpPerspective(img, H, (w, h))

    # 7. 可视化对比
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.plot(cx, cy, 'b+', markersize=15, markeredgewidth=2, label='光轴中心')
    plt.plot(face_u, face_v, 'ro', markersize=8, label='人脸中心')
    plt.title(f'原始图像 (A柱视角)\n偏置: Yaw={yaw_bias:.1f}°, Pitch={pitch_bias:.1f}°')
    plt.legend()
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(rectified_img)
    # 矫正后的人脸中心应该移动到光轴中心附近
    rectified_face_pt = H @ np.array([face_u, face_v, 1])
    rectified_face_pt = rectified_face_pt[:2] / rectified_face_pt[2]
    plt.plot(cx, cy, 'b+', markersize=15, markeredgewidth=2)
    plt.plot(rectified_face_pt[0], rectified_face_pt[1], 'go', markersize=8, label='矫正后人脸中心')
    plt.title('透视矫正后 (模拟正视)')
    plt.legend()
    plt.axis('off')
    
    # 裁剪局部对比形变
    crop_size = int(min(w, h) * 0.4)
    x1, y1 = int(max(0, face_u - crop_size/2)), int(max(0, face_v - crop_size/2))
    x2, y2 = int(min(w, x1 + crop_size)), int(min(h, y1 + crop_size))
    
    rx1, ry1 = int(max(0, rectified_face_pt[0] - crop_size/2)), int(max(0, rectified_face_pt[1] - crop_size/2))
    rx2, ry2 = int(min(w, rx1 + crop_size)), int(min(h, ry1 + crop_size))
    
    crop_orig = img[y1:y2, x1:x2]
    crop_rect = rectified_img[ry1:ry2, rx1:rx2]
    
    if crop_orig.shape == crop_rect.shape and crop_orig.size > 0:
        plt.subplot(1, 3, 3)
        # 将两张图拼在一起对比
        combined = np.hstack((crop_orig, crop_rect))
        plt.imshow(combined)
        plt.title('局部形变对比 (左:原始 右:矫正)')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('rectify_result.png')
    print("对比图已保存为 rectify_result.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Head Pose Perspective Correction Demo")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--focal_length", type=float, default=None, help="Camera focal length in pixels (optional)")
    args = parser.parse_args()
    
    rectify_image(args.image, args.focal_length)
