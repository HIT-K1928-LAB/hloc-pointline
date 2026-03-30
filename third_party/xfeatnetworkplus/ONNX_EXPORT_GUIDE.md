# XFeat 模型 ONNX 导出指南

本文档说明如何将 XFeat 模型导出为 ONNX 格式。

## 模型结构

XFeatModel 是一个特征提取网络，包含以下组件：

- **输入**: RGB 图像 (B, 3, H, W) 或 灰度图 (B, 1, H, W)
- **输出**:
  - `features`: 密集局部特征 (B, 64, H/8, W/8)
  - `keypoints`: 关键点logit图 (B, 65, H/8, W/8)
  - `heatmap`: 可靠性热力图 (B, 1, H/8, W/8)

## 环境要求

```bash
pip install torch onnx onnxruntime numpy pillow
```

## 导出步骤

### 方法 1: 使用导出脚本 (推荐)

运行导出脚本:

```bash
cd /home/hxy/doctor/feature\ dectect/hloc/Hierarchical-Localization/third_party/xfeatnetworkplus/
python export_to_onnx.py
```

脚本会提示选择导出模式:
- **选项 1**: 导出原始 XFeatModel (需要灰度图输入)
- **选项 2**: 导出带预处理的包装器 (接受 RGB 输入) - 推荐

### 方法 2: 在 Python 代码中导出

```python
import torch
from export_to_onnx import export_simple_wrapper

# 导出为接受 RGB 输入的 ONNX 模型
export_simple_wrapper(
    weights_path='weights/best.pth',
    output_path='weights/xfeat_rgb.onnx',
    input_shape=(1, 3, 480, 640),
    opset_version=17
)
```

## 测试导出的模型

### 运行测试脚本

```bash
python test_onnx.py --batch 1 --height 480 --width 640
```

测试脚本会:
1. 加载 PyTorch 模型
2. 加载 ONNX 模型
3. 比较两者的输出
4. 运行性能基准测试

### 使用测试图像

```bash
python test_onnx.py --image /path/to/test_image.jpg
```

## 使用 ONNX 模型进行推理

### Python 示例

```python
import numpy as np
import onnxruntime as ort
from PIL import Image

# 加载 ONNX 模型
session = ort.InferenceSession('xfeat_rgb.onnx')

# 获取输入输出信息
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

# 加载并预处理图像
img = Image.open('image.jpg').convert('RGB')
img = img.resize((640, 480))
img_array = np.array(img).astype(np.float32) / 255.0
img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
img_array = np.expand_dims(img_array, axis=0)   # Add batch dim

# 运行推理
outputs = session.run(output_names, {input_name: img_array})

features, keypoints, heatmap = outputs

print(f"Features shape:  {features.shape}")   # (1, 64, 60, 80)
print(f"Keypoints shape: {keypoints.shape}")  # (1, 65, 60, 80)
print(f"Heatmap shape:   {heatmap.shape}")    # (1, 1, 60, 80)
```

### C++ 示例 (使用 ONNX Runtime C++ API)

```cpp
#include <onnxruntime_cxx_api.h>

int main() {
    // 创建环境
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
    Ort::SessionOptions session_options;
    Ort::Session session{env, L"xfeat_rgb.onnx", session_options};

    // 获取输入输出名称
    char* input_name = session.GetInputName(0, allocator);
    char* output_names[] = {
        session.GetOutputName(0, allocator),
        session.GetOutputName(1, allocator),
        session.GetOutputName(2, allocator)
    };

    // 准备输入数据
    std::vector<int64_t> input_shape = {1, 3, 480, 640};
    std::vector<float> input_data(1 * 3 * 480 * 640);
    // ... 填充输入数据 ...

    // 创建输入张量
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());

    // 运行推理
    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        &input_name, &input_tensor, 1,
        output_names, 3
    );

    // 获取输出
    float* features = outputs[0].GetTensorMutableData<float>();
    float* keypoints = outputs[1].GetTensorMutableData<float>();
    float* heatmap = outputs[2].GetTensorMutableData<float>();

    return 0;
}
```

## 模型输出说明

### Features (特征描述符)
- 形状: `(B, 64, H/8, W/8)`
- 范围: L2 归一化后的向量
- 用途: 局部特征描述符，可用于特征匹配

### Keypoints (关键点logits)
- 形状: `(B, 65, H/8, W/8)`
- 说明: 包含64个方向类别 + 1个背景类
- 用途: 通过 softmax 和 argmax 提取关键点位置

### Heatmap (可靠性热力图)
- 形状: `(B, 1, H/8, W/8)`
- 范围: [0, 1] (经过 sigmoid)
- 用途: 表示特征检测的可靠性分数

## 动态输入尺寸

导出的模型支持动态输入尺寸。可以在推理时使用不同的高度和宽度:

```python
# 例如使用 (1, 3, 720, 1280) 的输入
img = img.resize((1280, 720))
# ... 其余代码相同 ...
```

## 故障排除

### 常见问题

1. **Opset 版本错误**
   - 确保使用 opset_version >= 17
   - 更新 ONNX Runtime: `pip install --upgrade onnxruntime`

2. **GPU 支持**
   - 对于 GPU 推理，安装: `pip install onnxruntime-gpu`
   - 确保 CUDA 版本匹配

3. **输出差异较大**
   - 检查是否正确加载了权重
   - 确认输入预处理一致 (RGB转灰度, 归一化)

## 性能优化建议

1. **使用固定输入尺寸** 可以获得更好的性能优化
2. **启用 ONNX Runtime 提供的优化选项**:
   ```python
   sess_options = ort.SessionOptions()
   sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
   sess = ort.InferenceSession('model.onnx', sess_options)
   ```

3. **对于生产环境**, 考虑使用 ONNX Runtime 的执行提供程序:
   - CUDA GPU: `ORTExecutionProviderCUDA`
   - TensorRT: `ORTExecutionProviderTensorRT`
   - OpenVINO: `ORTExecutionProviderOpenVINO`

## 文件结构

```
third_party/xfeatnetworkplus/
├── model.py              # XFeatModel 定义
├── xfeat.py              # XFeat 推理类
├── Nova.py               # NovaOp 模块
├── PartialConv.py        # PartialConv3WithSpatialAttention 模块
├── interpolator.py       # 插值工具
├── export_to_onnx.py     # ONNX 导出脚本
├── test_onnx.py          # ONNX 测试脚本
├── ONNX_EXPORT_GUIDE.md  # 本文档
└── weights/
    ├── best.pth          # PyTorch 权重
    ├── xfeat.pt          # 备用权重
    ├── xfeat_rgb.onnx    # 导出的 ONNX 模型 (RGB 输入)
    └── xfeat_raw.onnx    # 导出的 ONNX 模型 (灰度输入)
```
