# XFeat 模型 ONNX 转换完成

## 概述

已成功将 XFeat 模型转换为 ONNX 格式，输入尺寸为 `(1, 3, 480, 640)`。

## 导出的文件

| 文件 | 路径 | 说明 |
|------|------|------|
| ONNX 模型 | `third_party/xfeatnetworkplus/weights/xfeat_rgb.onnx` | 接受 RGB 输入的完整模型 (2.7MB) |
| 导出脚本 | `third_party/xfeatnetworkplus/export_to_onnx.py` | ONNX 导出脚本 |
| 推理示例 | `third_party/xfeatnetworkplus/inference_example.py` | Python 推理示例代码 |

## 模型信息

### 输入
- **名称**: `input_image`
- **形状**: `(1, 3, 480, 640)` - 批次大小=1, RGB通道, 高度=480, 宽度=640
- **格式**: Float32, 值范围 [0, 1]
- **说明**: 输入 RGB 图像会被内部转换为灰度图并归一化

### 输出

| 输出 | 形状 | 说明 |
|------|------|------|
| `features` | `(1, 64, 60, 80)` | 密集局部特征描述符 (L2归一化) |
| `keypoints` | `(1, 65, 60, 80)` | 关键点 logits (65类: 64方向+1背景) |
| `heatmap` | `(1, 1, 60, 80)` | 可靠性热力图 (0-1范围, sigmoid输出) |

注意: 输出的高度和宽度是输入的 1/8 (480/8=60, 640/8=80)

## 验证结果

✓ ONNX 模型已验证有效
✓ ONNX 与 PyTorch 输出对比:
  - Features 最大差异: 3.19e-05
  - Keypoints 最大差异: 1.74e-04
  - Heatmap 最大差异: 4.23e-07

## 快速使用

### Python 推理示例

```python
import numpy as np
import onnxruntime as ort
from PIL import Image

# 加载模型
session = ort.InferenceSession('xfeat_rgb.onnx')

# 加载并预处理图像
img = Image.open('image.jpg').convert('RGB')
img = img.resize((640, 480))
img_array = np.array(img).astype(np.float32) / 255.0
img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
img_array = np.expand_dims(img_array, axis=0)   # 添加批次维度

# 运行推理
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: img_array})

features, keypoints, heatmap = outputs

print(f"Features:  {features.shape}")   # (1, 64, 60, 80)
print(f"Keypoints: {keypoints.shape}")  # (1, 65, 60, 80)
print(f"Heatmap:   {heatmap.shape}")    # (1, 1, 60, 80)
```

### C++ 推理示例

```cpp
#include <onnxruntime_cxx_api.h>

int main() {
    // 初始化 ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    Ort::Session session(env, L"xfeat_rgb.onnx", session_options);

    // 获取输入输出名称
    auto input_name = session.GetInputName(0, allocator);
    auto output_name1 = session.GetOutputName(0, allocator);
    auto output_name2 = session.GetOutputName(1, allocator);
    auto output_name3 = session.GetOutputName(2, allocator);

    // 准备输入: (1, 3, 480, 640)
    std::vector<int64_t> input_shape = {1, 3, 480, 640};
    std::vector<float> input_data(1 * 3 * 480 * 640);
    // ... 填充图像数据 ...

    // 创建输入张量
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());

    // 运行推理
    const char* input_names[] = {input_name};
    const char* output_names[] = {output_name1, output_name2, output_name3};
    auto outputs = session.Run(Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 3);

    return 0;
}
```

## 安装依赖

```bash
pip install numpy onnxruntime onnx pillow
```

对于 GPU 加速:
```bash
pip install onnxruntime-gpu
```

## 测试推理

```bash
cd third_party/xfeatnetworkplus
python inference_example.py
```

## 重新导出模型

如果需要修改配置后重新导出:

```python
from export_to_onnx import export_xfeat_to_onnx

export_xfeat_to_onnx(
    weights_path='weights/best.pth',
    output_path='weights/xfeat_rgb.onnx',
    input_shape=(1, 3, 480, 640),
    opset_version=17
)
```

## 技术细节

### ONNX 兼容性修改

原始模型使用了 `torch.unfold()` 操作,该操作在 ONNX 中不支持。导出时使用了以下替代方案:

```python
def _unfold_onnx(self, x, ws):
    """ONNX-compatible unfold operation"""
    B, C, H, W = x.shape
    x_reshaped = x.view(B, C, H // ws, ws, W // ws, ws)
    x_transposed = x_reshaped.permute(0, 1, 3, 5, 2, 4).contiguous()
    unfolded = x_transposed.view(B, C * ws * ws, H // ws, W // ws)
    return unfolded
```

### 模型结构

1. **预处理**: RGB → 灰度图 → InstanceNorm
2. **骨干网络**: 5个卷积块 (block1-block5)
3. **特征融合**: NovaOp 多尺度融合
4. **输出头**:
   - heatmap_head: 可靠性热力图
   - keypoint_head: 关键点检测

## 性能

- 模型大小: 2.7 MB
- ONNX opset 版本: 17
- 支持动态批次和输入尺寸

## 故障排除

### 常见问题

1. **输入尺寸必须能被8整除**
   - 因为模型有8倍下采样
   - 推荐: 480x640, 640x480, 720x1280 等

2. **输入值范围应该是 [0, 1]**
   - 确保图像归一化到 [0, 1]
   - 不要使用 [-1, 1] 或未归一化的 [0, 255]

3. **GPU 推理**
   - 安装 `onnxruntime-gpu`
   - 确保 CUDA 版本兼容

## 参考

- XFeat 论文: "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024"
- ONNX Runtime 文档: https://onnxruntime.ai/docs/
