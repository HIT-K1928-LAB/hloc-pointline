"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Minimal example of how to use XFeat.
"""

import numpy as np
import os
import torch
import tqdm

from xfeat import XFeat
from model import *
import onnx
import onnxruntime as ort
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #Force CPU, comment for GPU

xfeat = XFeatModel()
onnx_path = "xfeatplus1.onnx"




#Random input
x = torch.randn(1,1,480,640, device='cuda' if torch.cuda.is_available() else 'cpu')

torch.onnx.export(xfeat, x, onnx_path, input_names=['input'], output_names=['output'], opset_version=11)
print("Model exported to ", onnx_path)

onnx_model = onnx.load(onnx_path)

# Check the model
onnx.checker.check_model(onnx_model)
print("ONNX model is valid.")