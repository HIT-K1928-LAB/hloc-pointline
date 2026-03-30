import torch
import torch.onnx
from xfeat_model_v1 import XFeatModel

weight_path = '/home/hxy/doctor/feature dectect/hloc/Hierarchical-Localization/third_party/xfeatnetworkplus/weights/best.pth'
onnx_file_path = 'xfeatplusv2.onnx'

device = torch.device('cpu')

model = XFeatModel()
model.load_state_dict(torch.load(weight_path, map_location=device))
model = model.to(device)
print('==> Load pre-trained network Successful.')
model.eval()

dummy_input = torch.randn(1, 1, 480, 640).to(device)


# # 定义动态输入尺寸
# dynamic_axes = {
#     'input': {2: 'height', 3: 'width'},  # 2 和 3 是图像的高度和宽度
#     'output_feats': {2: 'height/8', 3: 'width/8'},  # 输出张量的高度和宽度也设置为动态
#     'output_scores': {2: 'height', 3: 'width'},
#     'output_reliability': {2: 'height/8', 3: 'width/8'}
# }
torch.onnx.export(
    model,                    # PyTorch 模型
    dummy_input,              # 示例输入
    onnx_file_path,           # 输出的 ONNX 文件路径
    verbose=True,             # 输出详细信息
    input_names=['input'],    # 输入名称
    output_names=['output_feats', 'output_scores', 'output_reliability'],  # 输出名称
    #dynamic_axes=dynamic_axes,  # 设置动态轴
    opset_version=12,         # ONNX 操作集版本，通常选择 12 或更高
)



