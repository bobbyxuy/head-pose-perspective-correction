# Head Pose Perspective Correction (A-Pillar Camera)

本项目旨在解决在非正视视角（如 A 柱驾驶员监控摄像头）下，由于透视畸变导致 Head Pose 预测出现 Pitch 随 Yaw 漂移的系统性误差问题。

## 现象描述

当相机安装在 A 柱（驾驶员左前侧，偏置角约 30°-45°）时，即使驾驶员的真实 Pitch 保持不变，仅仅改变 Yaw 角（左右转头），模型预测的 Pitch 也会出现系统性漂移。例如，在 Yaw 处于 `[-45°, 45°]` 区间时，预测出的 Pitch 可能会在 `[-3°, 13°]` 之间变化。

这是因为大多数 Head Pose 模型在推理前会裁剪人脸 BBox，丢失了人脸在原图中的绝对位置信息和相机内参。当驾驶员转头时，人脸在图像中的位置发生偏移，透视投影会使图像产生形变（如左看时鼻尖相对向下，右看时鼻尖相对向上），模型将这种形变错误地解释为了 Pitch 的变化。

## 解决方案：透视矫正管线

为了解决这一问题，我们在将图像送入 Head Pose 模型之前，引入显式的几何透视矫正步骤：

1. **相机内参估计**：获取相机的内参矩阵 $K$。
2. **计算透视偏置**：根据人脸在原图中的位置和内参，计算偏离光轴的角度。
3. **图像透视矫正 (WarpPerspective)**：将原图“矫正”为正视视角，消除透视畸变。
4. **姿态估计**：将矫正后的图像送入高精度模型（如 SemiUHPE）预测姿态。
5. **姿态逆变换**：将预测出的姿态通过逆矩阵补偿回真实的相机坐标系。

## 依赖安装

推荐使用 Python 3.8+ 环境。

```bash
# 基础依赖
pip install numpy opencv-python scipy matplotlib

# 人脸检测 + 关键点（MediaPipe FaceLandmarker）
pip install mediapipe

# 可选：单图相机内参估计（GeoCalib ECCV 2024，比启发式猜测更准确）
pip install -e "git+https://github.com/cvg/GeoCalib#egg=geocalib"

# 可选：PyTorch（运行 SemiUHPE 伪标签生成管线）
pip install torch torchvision
```

## 预训练模型下载

本项目推荐使用 **SemiUHPE (TPAMI 2025)** 作为高精度 Head Pose 预测模型。

1. **克隆代码库**：
   ```bash
   git clone https://github.com/hnuzhy/SemiUHPE.git
   ```
2. **下载权重**：
   - 官方权重托管在 Google Drive：[下载链接](https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq)
   - 推荐下载 `DAD-WildHead-EffNetV2-S-best.pth`（在野外无约束场景精度更高）。
   - 将下载的权重文件放置在 `SemiUHPE/weights/` 目录下。

## 测试脚本说明

本项目提供两个测试脚本，帮助您验证透视矫正的效果：

1. `rectify_demo.py`：使用 **MediaPipe FaceLandmarker** 检测人脸关键点，以关键点几何中心作为人脸中心，对单张图像进行透视矫正，可视化矫正前后的人脸形变差异。
2. `pitch_drift_sim.py`：使用精确的 3D 几何模型，仿真在 A 柱视角下，Pitch 随 Yaw 漂移的现象，并验证透视矫正能否彻底消除该漂移。

## 使用方法

### 1. 图像透视矫正测试

运行 `rectify_demo.py`，脚本将读取示例图像，应用透视矫正，并生成对比图。

```bash
python rectify_demo.py --image path/to/your/image.jpg --focal_length 1000
```

*参数说明：*
- `--image`: 输入图像路径。
- `--focal_length`: 相机焦距（像素，可选）。如果未提供，脚本将启发式猜测 FOV=60°。在实际应用中，强烈建议使用标定得到的准确焦距或通过 GeoCalib 估计。
- `--model_path`: MediaPipe `face_landmarker.task` 模型路径（默认 `face_landmarker.task`，首次运行时会自动下载）。
- `--output`: 输出可视化图像路径（默认 `rectify_result.png`）。

**关于人脸中心的计算方式：**

本脚本使用 MediaPipe FaceLandmarker 的 478 个 3D 面部关键点，取面部轮廓、鼻梁、眼眶等关键点子集的 2D 投影坐标几何均值作为人脸中心。相比纯 BBox 中点，该中心更接近面部真实几何中心，对侧脸和遮挡情况更鲁棒。

### 2. Pitch 漂移仿真测试

运行 `pitch_drift_sim.py`，脚本将生成 Pitch 漂移曲线对比图。

```bash
python pitch_drift_sim.py
```

该脚本模拟相机安装在偏置角 30°（Yaw）和 10°（Pitch）的位置，驾驶员真实 Pitch 保持 5° 不变，Yaw 在 `[-45°, 45°]` 之间变化。输出图表将直观展示未矫正时的严重漂移以及矫正后的稳定效果。

### 3. 完整伪标签生成管线

```bash
# 克隆 SemiUHPE 并下载权重后运行
python pseudolabel_pipeline.py \
    --image_dir ./your_images \
    --output_dir ./pseudo_labels \
    --semiuhpe_root ./SemiUHPE \
    --weight_path ./SemiUHPE/weights/DAD-WildHead-EffNetV2-S-best.pth \
    --save_vis
```

*参数说明：*
- `--image_dir`: 输入图像目录。
- `--output_dir`: 伪标签 JSON 输出目录。
- `--focal_length`: 相机焦距（像素，可选，不提供则自动估计）。
- `--intrinsics_method`: 内参估计方法，`heuristic`（默认）或 `geocalib`。
- `--semiuhpe_root`: SemiUHPE 代码库根目录（不提供则仅执行检测和矫正步骤）。
- `--weight_path`: SemiUHPE 权重文件路径。
- `--save_vis`: 保存可视化图像（含关键点中心和偏置角标注）。

## 注意事项

- 所有姿态补偿必须在**旋转矩阵**空间内完成（$R_{final} = R_{rectify}^{-1} \cdot R_{model}$），绝对不能直接对欧拉角做加减。
- 如果您的相机是固定安装的（如某款车型的 DMS 摄像头），建议离线标定相机内参 $K$，并在推理时作为常量硬编码，避免逐帧调用内参估计模型，以提高运行效率。
