# calibrate_from_frontal.py 使用说明

## 一句话概述

从驾驶员**正视前方**的图片中估计 A 柱相机的外参旋转矩阵，并将 HopeNet 预测的欧拉角从相机坐标系补偿到世界坐标系，消除 Pitch 随 Yaw 变化的系统性漂移。

---

## 背景与原理

### 问题来源

A 柱相机并非正对驾驶员安装，而是侧视 30–45°。HopeNet 等模型输出的欧拉角是**相机坐标系**下的姿态，当驾驶员纯粹左右转头（Yaw 变化）时，由于相机坐标系与世界坐标系不对齐，模型预测的 Pitch 会出现 [-3, 13]° 的系统性漂移。

### 核心数学

设相机外参旋转矩阵为 `R_extrinsic`，模型预测的旋转矩阵为 `R_camera`，则世界坐标系下的真实旋转为：

```
R_world = R_extrinsic @ R_camera
```

从 `R_world` 提取的欧拉角即为无漂移的真实姿态。

> **关键约束**：必须在旋转矩阵空间做矩阵乘法，**不能**直接对欧拉角做加减。

### 外参估计原理

当驾驶员**正视前方**时，定义此时头部姿态为世界坐标系零点（`R_world = I`）。  
因此：`R_extrinsic = R_camera`（即模型在正视前方图片上的预测值）。

多张图片时，使用 **SVD 正交化均值**（不能直接对矩阵元素取均值）：

```python
R_sum = sum(R_list)
U, _, Vt = np.linalg.svd(R_sum)
R_mean = U @ Vt
```

---

## 安装依赖

```bash
pip install numpy scipy opencv-python matplotlib pandas
```

---

## 准备工作

### 1. 准备正视前方图片

| 要求 | 说明 |
|------|------|
| 数量 | 建议 **5–20 张**，越多越稳定 |
| 姿态 | 驾驶员头部**水平**，视线朝向正前方（以车身为参考） |
| 多样性 | 不同驾驶员、不同光照均可，但头部姿态必须是正视前方 |
| 格式 | JPG / PNG，任意分辨率 |
| 注意 | 图片中头部应已裁剪或模型能正确检测到人脸 |

> **"正视前方"的定义**：驾驶员头部与车身平行，Yaw ≈ 0°，Pitch ≈ 0°，Roll ≈ 0°。如果驾驶员习惯性低头或侧头，该图片不应纳入标定集。

### 2. 替换推理函数

脚本中预留了两个接口函数，**必须**替换为您的实际代码：

#### `load_model()`

```python
def load_model():
    import torch
    from hopenet import Hopenet
    import torchvision
    model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    model.load_state_dict(torch.load('hopenet_robust_alpha1.pkl', map_location='cpu'))
    model.eval()
    return model
```

#### `predict_euler_from_image(img_bgr, model, convention='ZYX')`

```python
def predict_euler_from_image(img_bgr, model, convention='ZYX'):
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    transformations = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transformations(img_pil).unsqueeze(0)
    with torch.no_grad():
        yaw_pred, pitch_pred, roll_pred = model(img_tensor)
    idx_tensor = torch.FloatTensor(range(66))
    yaw   = (torch.nn.functional.softmax(yaw_pred,   dim=1) * idx_tensor).sum(1).item() * 3 - 99
    pitch = (torch.nn.functional.softmax(pitch_pred, dim=1) * idx_tensor).sum(1).item() * 3 - 99
    roll  = (torch.nn.functional.softmax(roll_pred,  dim=1) * idx_tensor).sum(1).item() * 3 - 99
    return yaw, pitch, roll
```

> **欧拉角约定**：HopeNet 使用 ZYX 约定（先 Yaw，再 Pitch，再 Roll）。如果您的模型使用其他约定，修改 `--convention` 参数和上述函数中的约定字符串。

---

## 三步使用流程

### Step 1：估计外参

```bash
python calibrate_from_frontal.py --mode calibrate \
    --images frontal1.jpg frontal2.jpg frontal3.jpg frontal4.jpg frontal5.jpg \
    --output extrinsic.npy
```

**输出示例：**

```
[Step 1] 从 5 张正视前方图片估计外参...
  frontal1.jpg                    Yaw= -32.15°  Pitch=  +2.31°  Roll=  +1.05°
  frontal2.jpg                    Yaw= -31.87°  Pitch=  +2.18°  Roll=  +0.98°
  ...

[结果] 相机安装角度估计（均值）:
  Yaw   = -31.97°  （相机偏左/右）
  Pitch =  +2.24°  （相机偏上/下）
  Roll  =  +1.01°  （相机旋转）

[完成] 外参矩阵已保存到: extrinsic.npy
```

> Yaw ≈ -32° 符合 A 柱相机侧视 30–45° 的预期。

### Step 2：验证补偿效果

准备几张驾驶员**转头**的图片（向左看、向右看），验证补偿后 Pitch 是否接近 0°：

```bash
python calibrate_from_frontal.py --mode verify \
    --images look_left.jpg look_right.jpg look_center.jpg \
    --extrinsic extrinsic.npy
```

**输出示例：**

```
图片                           Yaw(前)   Pitch(前)   Roll(前)   Yaw(后)   Pitch(后)   Roll(后)
------------------------------------------------------------------------------------------
look_left.jpg                  -62.3°     +11.2°      +1.5°    -30.1°      +0.3°      +0.8°
look_right.jpg                  +2.1°      -2.8°      +0.9°    +34.2°      +0.1°      +0.5°
look_center.jpg                -31.8°      +2.2°      +1.0°     +0.2°      +0.0°      +0.2°
```

**判断标准：**

| 指标 | 良好 | 需要检查 |
|------|------|----------|
| 补偿后 Pitch 绝对值 | < 2° | > 5° |
| 补偿后 Pitch 方差 | < 1° | > 3° |
| 补偿后正视前方 Yaw | ≈ 0° | 偏差 > 5° |

同时会生成 `verify_result.png` 可视化对比图。

### Step 3：批量补偿伪标签

```bash
python calibrate_from_frontal.py --mode compensate \
    --label_csv pseudo_labels.csv \
    --extrinsic extrinsic.npy \
    --output_csv pseudo_labels_compensated.csv
```

**输入 CSV 格式：**

```csv
image_path,yaw,pitch,roll
img001.jpg,12.3,-8.5,1.2
img002.jpg,-5.1,-9.0,0.8
```

**输出 CSV 格式**（原始列保留为 `*_cam`，新列为世界坐标系）：

```csv
image_path,yaw,pitch,roll,yaw_cam,pitch_cam,roll_cam
img001.jpg,44.5,0.3,0.9,12.3,-8.5,1.2
img002.jpg,27.1,0.1,0.5,-5.1,-9.0,0.8
```

---

## 在伪标签生成管线中集成

### 方案 A：离线批量补偿（推荐）

```python
import numpy as np
from calibrate_from_frontal import compensate_extrinsic

# 加载外参（一次性）
R_extrinsic = np.load('extrinsic.npy')

# 对每条伪标签做补偿
for label in pseudo_labels:
    yaw_w, pitch_w, roll_w = compensate_extrinsic(
        label['yaw'], label['pitch'], label['roll'], R_extrinsic
    )
    label['yaw']   = yaw_w
    label['pitch'] = pitch_w
    label['roll']  = roll_w
```

### 方案 B：在 Teacher 模型推理后直接补偿

```python
# Teacher 模型（SemiUHPE）推理后
yaw_teacher, pitch_teacher, roll_teacher = teacher_model.predict(img)

# 立即做外参补偿
yaw_w, pitch_w, roll_w = compensate_extrinsic(
    yaw_teacher, pitch_teacher, roll_teacher, R_extrinsic
)

# 将补偿后的结果写入伪标签
```

---

## 常见问题

### Q1：补偿后 Pitch 仍然有漂移，怎么办？

可能原因及解决方案：

1. **正视前方图片不够准确**：检查标定图片，确保驾驶员头部真正水平，剔除低头/侧头的图片。
2. **图片数量不足**：增加到 10 张以上。
3. **模型本身存在系统误差**：尝试用 SemiUHPE 等高精度模型替换 HopeNet 做 Teacher。
4. **欧拉角约定不匹配**：确认 `--convention` 参数与您的模型一致。

### Q2：外参 Yaw 估计值与实际安装角度差异较大？

模型预测的欧拉角包含模型自身的偏差，估计出的外参是"模型感知到的相机安装角度"，不一定等于物理安装角度。这是正常的，因为我们的目标是消除漂移，而非精确测量物理角度。

### Q3：如何处理多个相机（不同安装位置）？

对每个相机分别运行 `calibrate` 模式，得到各自的 `extrinsic.npy`，在补偿时使用对应的外参矩阵。

### Q4：Student 模型训练后还需要做外参补偿吗？

**不需要**。外参补偿只在**离线伪标签生成**阶段使用。Student 模型用补偿后的无漂移标签训练，训练完成后线上推理时直接输出世界坐标系下的姿态，无需任何后处理。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `calibrate_from_frontal.py` | 主脚本（外参估计、验证、批量补偿） |
| `extrinsic.npy` | 估计出的外参旋转矩阵（3×3 numpy array） |
| `verify_result.png` | 验证模式生成的可视化对比图 |
| `pseudo_labels_compensated.csv` | 补偿后的伪标签 CSV |

---

## 参考资料

- HopeNet: [Hopenet: Towards Multi-View Head Pose Estimation](https://arxiv.org/abs/1710.00925)
- SemiUHPE: [SemiUHPE: Semi-Supervised Unconstrained Head Pose Estimation](https://arxiv.org/abs/2307.01566) (TPAMI 2025)
- 旋转矩阵 SO(3) 均值: [Averaging Rotations](https://www.cs.cmu.edu/~quake/robust.html)
