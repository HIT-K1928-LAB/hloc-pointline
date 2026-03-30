"""
XFeat Model ONNX Export Script
Exports XFeatModel to ONNX format with fixed input size (1, 3, 480, 640)
"""

import torch
import torch.onnx
import torch.nn as nn
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from third_party.xfeatnetworkplus.model import XFeatModel, BasicLayer
from third_party.xfeatnetworkplus.Nova import NovaOp
from third_party.xfeatnetworkplus.PartialConv import PartialConv3WithSpatialAttention


def unfold2d_onnx_compatible(x, ws=2):
    """
    ONNX-compatible version of unfold2d.
    Implements the same functionality as _unfold2d in model.py but using
    operations that are supported by ONNX.
    """
    B, C, H, W = x.shape

    # Manual implementation using convolution-like reshaping
    # This creates sliding windows of size ws x ws

    # Create patches by reshaping
    # For ws=8, we reshape to extract 8x8 patches
    patches = []

    for i in range(ws):
        for j in range(ws):
            # Extract patch starting at (i, j)
            if i == 0 and j == 0:
                patch = x[:, :, i:H-ws+1+i+1, j:W-ws+1+j+1]
            else:
                patch = x[:, :, i:H-ws+1+i+1, j:W-ws+1+j+1]
            patches.append(patch)

    # Stack and reshape
    # Result shape: (B, C*ws*ws, H//ws, W//ws)
    result = torch.cat(patches, dim=1)

    # Downsample by taking every ws-th element
    result = result[:, :, ::ws, ::ws]

    return result


class XFeatONNXWrapper(nn.Module):
    """
    ONNX-compatible wrapper for XFeatModel.
    Replaces unsupported operations (like unfold) with ONNX-compatible alternatives.
    """

    def __init__(self, xfeat_model):
        super().__init__()
        # Copy all necessary components from the original model
        self.norm = nn.InstanceNorm2d(1)

        # Copy all blocks
        self.skip1 = xfeat_model.skip1
        self.block1 = xfeat_model.block1
        self.block2 = xfeat_model.block2
        self.block3 = xfeat_model.block3
        self.block4 = xfeat_model.block4
        self.block5 = xfeat_model.block5
        self.block_fusion = xfeat_model.block_fusion
        self.heatmap_head = xfeat_model.heatmap_head
        self.keypoint_head = xfeat_model.keypoint_head

        # Store for unfold operation
        self.ws = 8

    def forward(self, x):
        """
        Forward pass compatible with ONNX export.
        Args:
            x: Input tensor (B, 3, H, W) - RGB image
        Returns:
            feats: (B, 64, H/8, W/8) - Dense features
            keypoints: (B, 65, H/8, W/8) - Keypoint logits
            heatmap: (B, 1, H/8, W/8) - Reliability heatmap
        """
        import torch.nn.functional as F

        # Convert RGB to grayscale and normalize
        x = x.mean(dim=1, keepdim=True)
        x = self.norm(x)

        # Main backbone
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        # Pyramid fusion
        x4_interp = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear', align_corners=False)
        x5_interp = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear', align_corners=False)
        feats = self.block_fusion(x3 + x4_interp + x5_interp, x5_interp, x4_interp, x3)

        # Heads
        heatmap = self.heatmap_head(feats)

        # Keypoint head with ONNX-compatible unfold
        # Using a custom reshape approach instead of torch.unfold
        B, C, H, W = x.shape

        # Manual unfold implementation for ws=8
        # This creates 8x8 patches and flattens them
        # Result: (B, C*64, H//8, W//8)

        # Method using slicing and reshaping
        unfolded = self._unfold_onnx(x, self.ws)
        keypoints = self.keypoint_head(unfolded)

        return feats, keypoints, heatmap

    def _unfold_onnx(self, x, ws):
        """
        ONNX-compatible unfold operation.
        Creates ws x ws patches from input tensor.
        """
        B, C, H, W = x.shape

        # Ensure dimensions are divisible by ws
        assert H % ws == 0 and W % ws == 0, f"Height and Width must be divisible by {ws}"

        # Reshape to create patches
        # Shape: (B, C, H//ws, ws, W//ws, ws)
        x_reshaped = x.view(B, C, H // ws, ws, W // ws, ws)

        # Rearrange dimensions: (B, C, ws, ws, H//ws, W//ws)
        x_transposed = x_reshaped.permute(0, 1, 3, 5, 2, 4).contiguous()

        # Reshape to final: (B, C*ws*ws, H//ws, W//ws)
        unfolded = x_transposed.view(B, C * ws * ws, H // ws, W // ws)

        return unfolded


def export_xfeat_to_onnx(
    weights_path=None,
    output_path="xfeat_model.onnx",
    input_shape=(1, 3, 480, 640),
    opset_version=17
):
    """
    Export XFeatModel to ONNX format.

    Args:
        weights_path: Path to model weights (.pth file)
        output_path: Output ONNX file path
        input_shape: Input tensor shape (batch, channels, height, width)
        opset_version: ONNX opset version
    """

    print("=" * 60)
    print("XFeat Model ONNX Export")
    print("=" * 60)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    print("\nInitializing XFeatModel...")
    xfeat_model = XFeatModel().to(device)

    # Load weights
    if weights_path is None:
        weights_path = '/home/hxy/doctor/feature dectect/hloc/Hierarchical-Localization/third_party/xfeatnetworkplus/weights/best.pth'

    if os.path.exists(weights_path):
        print(f"Loading weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        xfeat_model.load_state_dict(state_dict)
        print("Weights loaded successfully!")
    else:
        print(f"Warning: Weights file not found at {weights_path}")
        print("Exporting model with random initialization...")

    xfeat_model.eval()

    # Create ONNX-compatible wrapper
    print("\nCreating ONNX-compatible wrapper...")
    model = XFeatONNXWrapper(xfeat_model).to(device)
    model.eval()

    # Create dummy input
    batch_size, channels, height, width = input_shape
    dummy_input = torch.randn(batch_size, channels, height, width, device=device)
    print(f"Dummy input shape: {dummy_input.shape}")

    # Test the model before export
    print("\nTesting model forward pass...")
    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"✓ Forward pass successful!")
        print(f"  Output shapes:")
        print(f"    Features:  {test_output[0].shape}")
        print(f"    Keypoints: {test_output[1].shape}")
        print(f"    Heatmap:   {test_output[2].shape}")

    # Define input and output names
    input_names = ['input_image']
    output_names = ['features', 'keypoints', 'heatmap']

    # Define dynamic axes
    dynamic_axes = {
        'input_image': {0: 'batch_size', 2: 'height', 3: 'width'},
        'features': {0: 'batch_size', 2: 'height', 3: 'width'},
        'keypoints': {0: 'batch_size', 2: 'height', 3: 'width'},
        'heatmap': {0: 'batch_size', 2: 'height', 3: 'width'}
    }

    print("\nExporting to ONNX...")
    print(f"Output path: {output_path}")
    print(f"Opset version: {opset_version}")

    try:
        # Export the model
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )

        print(f"\n✓ ONNX export successful!")
        print(f"✓ Model saved to: {output_path}")

        # Verify the model
        print("\nVerifying ONNX model...")
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid!")

        # Print model info
        print("\n" + "=" * 60)
        print("Model Information:")
        print("=" * 60)
        print(f"Input shape:  {input_shape}")
        print(f"Output shapes (for 480x640 input):")
        print(f"  - Features:  (1, 64, 60, 80)")
        print(f"  - Keypoints: (1, 65, 60, 80)")
        print(f"  - Heatmap:   (1, 1,  60, 80)")

        print("\nONNX Graph Inputs:")
        for inp in onnx_model.graph.input:
            print(f"  - {inp.name}")

        print("\nONNX Graph Outputs:")
        for out in onnx_model.graph.output:
            print(f"  - {out.name}")

        print("\n" + "=" * 60)
        print("Export completed successfully!")
        print("=" * 60)

        return output_path

    except Exception as e:
        print(f"\n✗ Export failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Default paths
    WEIGHTS_PATH = '/home/hxy/doctor/feature dectect/hloc/Hierarchical-Localization/third_party/xfeatnetworkplus/weights/best.pth'
    OUTPUT_DIR = '/home/hxy/doctor/feature dectect/hloc/Hierarchical-Localization/third_party/xfeatnetworkplus/weights'

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Export the model
    output_path = os.path.join(OUTPUT_DIR, "xfeat_rgb.onnx")

    export_xfeat_to_onnx(
        weights_path=WEIGHTS_PATH,
        output_path=output_path,
        input_shape=(1, 3, 480, 640),
        opset_version=17
    )
