"""
Test script for exported ONNX model
Compares PyTorch and ONNX outputs
"""

import torch
import onnxruntime as ort
import numpy as np
import sys
import os
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from third_party.xfeatnetworkplus.model import XFeatModel


def test_onnx_model(
    onnx_path=None,
    weights_path=None,
    test_image_path=None,
    input_shape=(1, 3, 480, 640)
):
    """
    Test ONNX model by comparing outputs with PyTorch model
    """

    print("=" * 60)
    print("Testing ONNX Model")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    if onnx_path is None:
        onnx_path = '/home/hxy/doctor/feature dectect/hloc/Hierarchical-Localization/third_party/xfeatnetworkplus/weights/xfeat_rgb.onnx'

    if weights_path is None:
        weights_path = '/home/hxy/doctor/feature dectect/hloc/Hierarchical-Localization/third_party/xfeatnetworkplus/weights/best.pth'

    # Load PyTorch model
    print("\nLoading PyTorch model...")
    pt_model = XFeatModel().to(device)
    if os.path.exists(weights_path):
        pt_model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"✓ Loaded weights from: {weights_path}")
    pt_model.eval()

    # Create dummy input
    batch_size, channels, height, width = input_shape
    dummy_input_torch = torch.randn(batch_size, channels, height, width, device=device)
    print(f"Input shape: {dummy_input_torch.shape}")

    # Get PyTorch outputs
    print("\nRunning PyTorch inference...")
    with torch.no_grad():
        # Preprocess (convert to grayscale and normalize)
        x = dummy_input_torch.mean(dim=1, keepdim=True)
        x = torch.nn.InstanceNorm2d(1)(x)

        pt_feats, pt_kpts, pt_heatmap = pt_model(x)

    print(f"✓ PyTorch outputs:")
    print(f"  - Features:  {pt_feats.shape}")
    print(f"  - Keypoints: {pt_kpts.shape}")
    print(f"  - Heatmap:   {pt_heatmap.shape}")

    # Load ONNX model
    print(f"\nLoading ONNX model from: {onnx_path}")
    ort_session = ort.InferenceSession(onnx_path)

    # Get input/output info
    input_name = ort_session.get_inputs()[0].name
    output_names = [o.name for o in ort_session.get_outputs()]

    print("✓ ONNX model loaded")
    print(f"  Input:  {input_name}")
    print(f"  Outputs: {output_names}")

    # Prepare input for ONNX
    dummy_input_onnx = dummy_input_torch.cpu().numpy().astype(np.float32)

    # Run ONNX inference
    print("\nRunning ONNX inference...")
    onnx_outputs = ort_session.run(output_names, {input_name: dummy_input_onnx})

    onnx_feats, onnx_kpts, onnx_heatmap = onnx_outputs

    print(f"✓ ONNX outputs:")
    print(f"  - Features:  {onnx_feats.shape}")
    print(f"  - Keypoints: {onnx_kpts.shape}")
    print(f"  - Heatmap:   {onnx_heatmap.shape}")

    # Compare outputs
    print("\n" + "=" * 60)
    print("Comparing PyTorch vs ONNX outputs")
    print("=" * 60)

    def compare_outputs(pt_output, onnx_output, name):
        pt_np = pt_output.cpu().numpy()
        diff = np.abs(pt_np - onnx_output)
        max_diff = diff.max()
        mean_diff = diff.mean()
        relative_diff = diff / (np.abs(pt_np) + 1e-8)

        print(f"\n{name}:")
        print(f"  Max absolute difference:  {max_diff:.6e}")
        print(f"  Mean absolute difference: {mean_diff:.6e}")
        print(f"  Max relative difference:  {relative_diff.max():.6e}")

        if max_diff < 1e-4:
            print(f"  ✓ PASS - Outputs match closely!")
        elif max_diff < 1e-2:
            print(f"  ⚠ WARNING - Small differences detected")
        else:
            print(f"  ✗ FAIL - Large differences detected")

        return max_diff

    max_diff = 0
    max_diff = max(max_diff, compare_outputs(pt_feats, onnx_feats, "Features"))
    max_diff = max(max_diff, compare_outputs(pt_kpts, onnx_kpts, "Keypoints"))
    max_diff = max(max_diff, compare_outputs(pt_heatmap, onnx_heatmap, "Heatmap"))

    print("\n" + "=" * 60)
    if max_diff < 1e-2:
        print("✓ Overall: ONNX model is working correctly!")
    else:
        print("⚠ Warning: Check differences above")
    print("=" * 60)

    # Benchmark inference speed
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    # PyTorch benchmark
    print("\nPyTorch inference (100 runs)...")
    import time

    pt_model.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            x = dummy_input_torch.mean(dim=1, keepdim=True)
            x = torch.nn.InstanceNorm2d(1)(x)
            _ = pt_model(x)
        pt_time = (time.time() - start) / 100

    print(f"  Average time: {pt_time*1000:.2f} ms")

    # ONNX benchmark
    print("\nONNX inference (100 runs)...")
    start = time.time()
    for _ in range(100):
        _ = ort_session.run(output_names, {input_name: dummy_input_onnx})
    onnx_time = (time.time() - start) / 100

    print(f"  Average time: {onnx_time*1000:.2f} ms")

    speedup = pt_time / onnx_time
    print(f"\n  Speedup: {speedup:.2f}x")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


def test_onnx_with_image(
    onnx_path=None,
    image_path=None
):
    """
    Test ONNX model with a real image
    """

    if image_path is None:
        print("No image path provided. Skipping image test.")
        return

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    print("=" * 60)
    print("Testing ONNX Model with Real Image")
    print("=" * 60)

    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((640, 480))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension

    print(f"\nImage shape: {img_array.shape}")
    print(f"Image range: [{img_array.min():.3f}, {img_array.max():.3f}]")

    # Load ONNX model
    if onnx_path is None:
        onnx_path = '/home/hxy/doctor/feature dectect/hloc/Hierarchical-Localization/third_party/xfeatnetworkplus/weights/xfeat_rgb.onnx'

    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    output_names = [o.name for o in ort_session.get_outputs()]

    # Run inference
    print("\nRunning inference...")
    outputs = ort_session.run(output_names, {input_name: img_array})

    feats, kpts, heatmap = outputs

    print("✓ Inference complete!")
    print(f"  Features:  {feats.shape} - range: [{feats.min():.3f}, {feats.max():.3f}]")
    print(f"  Keypoints: {kpts.shape} - range: [{kpts.min():.3f}, {kpts.max():.3f}]")
    print(f"  Heatmap:   {heatmap.shape} - range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

    print("\n✓ Image test completed successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test ONNX XFeat model')
    parser.add_argument('--onnx', type=str, default=None,
                        help='Path to ONNX model')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to PyTorch weights')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to test image')
    parser.add_argument('--batch', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--height', type=int, default=480,
                        help='Input height')
    parser.add_argument('--width', type=int, default=640,
                        help='Input width')

    args = parser.parse_args()

    input_shape = (args.batch, 3, args.height, args.width)

    # Run comparison test
    test_onnx_model(
        onnx_path=args.onnx,
        weights_path=args.weights,
        input_shape=input_shape
    )

    # Run image test if image provided
    if args.image:
        test_onnx_with_image(
            onnx_path=args.onnx,
            image_path=args.image
        )
