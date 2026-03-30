"""
Simple inference example for exported XFeat ONNX model
"""

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch


def load_and_preprocess_image(image_path, target_size=(640, 480)):
    """
    Load and preprocess an image for ONNX model inference.

    Args:
        image_path: Path to input image
        target_size: Target size (width, height)

    Returns:
        Preprocessed numpy array with shape (1, 3, H, W)
    """
    # Load image
    img = Image.open(image_path).convert('RGB')

    # Store original size
    original_size = img.size

    # Resize to target size
    img = img.resize(target_size)

    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0

    # Convert from HWC to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, original_size


def run_inference(onnx_model_path, image_path):
    """
    Run inference on an image using the ONNX model.

    Args:
        onnx_model_path: Path to ONNX model
        image_path: Path to input image
    """
    print("=" * 60)
    print("XFeat ONNX Model Inference")
    print("=" * 60)

    # Load ONNX model
    print(f"\nLoading ONNX model from: {onnx_model_path}")
    session = ort.InferenceSession(onnx_model_path)

    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    print(f"Input name:  {input_name}")
    print(f"Output names: {output_names}")

    # Get input info
    input_shape = session.get_inputs()[0].shape
    print(f"Expected input shape: {input_shape}")

    # Load and preprocess image
    print(f"\nLoading image from: {image_path}")
    img_array, original_size = load_and_preprocess_image(image_path)

    print(f"Original image size: {original_size}")
    print(f"Preprocessed shape: {img_array.shape}")
    print(f"Value range: [{img_array.min():.3f}, {img_array.max():.3f}]")

    # Run inference
    print("\nRunning inference...")
    outputs = session.run(output_names, {input_name: img_array})

    features, keypoints, heatmap = outputs

    print("✓ Inference complete!")

    # Print output info
    print("\n" + "-" * 60)
    print("Output Information:")
    print("-" * 60)

    print(f"\n1. Features (Dense local descriptors):")
    print(f"   Shape: {features.shape}")
    print(f"   Range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"   Mean:  {features.mean():.3f}")
    print(f"   Std:   {features.std():.3f}")

    print(f"\n2. Keypoints (Keypoint logits):")
    print(f"   Shape: {keypoints.shape}")
    print(f"   Range: [{keypoints.min():.3f}, {keypoints.max():.3f}]")
    print(f"   Mean:  {keypoints.mean():.3f}")

    print(f"\n3. Heatmap (Reliability map):")
    print(f"   Shape: {heatmap.shape}")
    print(f"   Range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    print(f"   Mean:  {heatmap.mean():.3f}")

    # Optional: Extract top keypoints from heatmap
    print("\n" + "-" * 60)
    print("Keypoint Detection (Top 10):")
    print("-" * 60)

    # Get heatmap scores
    scores = heatmap[0, 0]  # Remove batch and channel dimensions

    # Get top 10 scores
    flat_scores = scores.flatten()
    top_k = 10

    if len(flat_scores) > top_k:
        top_indices = np.argpartition(flat_scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-flat_scores[top_indices])]

        # Convert indices to (y, x) coordinates
        h, w = scores.shape
        top_coords = [(idx // w, idx % w) for idx in top_indices]
        top_scores = flat_scores[top_indices]

        print(f"\nTop {top_k} keypoints:")
        for i, ((y, x), score) in enumerate(zip(top_coords, top_scores), 1):
            # Scale coordinates to original input resolution
            x_scaled = x * 8  # 8x downsampling
            y_scaled = y * 8
            print(f"  {i}. Position: ({x_scaled:4d}, {y_scaled:4d}), Score: {score:.4f}")

    print("\n" + "=" * 60)
    print("Inference completed successfully!")
    print("=" * 60)

    return features, keypoints, heatmap


def compare_with_pytorch(onnx_model_path, pytorch_weights_path, image_path):
    """
    Compare ONNX model output with PyTorch model output.
    """
    print("=" * 60)
    print("Comparing ONNX vs PyTorch Outputs")
    print("=" * 60)

    # Load image
    img_array, _ = load_and_preprocess_image(image_path)

    # 1. Run ONNX inference
    print("\n1. Running ONNX inference...")
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    onnx_outputs = session.run(output_names, {input_name: img_array})
    onnx_feats, onnx_kpts, onnx_heatmap = onnx_outputs

    print("✓ ONNX inference complete")

    # 2. Run PyTorch inference
    print("\n2. Running PyTorch inference...")
    import sys
    sys.path.insert(0, '/home/hxy/doctor/feature dectect/hloc/Hierarchical-Localization')
    from third_party.xfeatnetworkplus.export_to_onnx import XFeatONNXWrapper
    from third_party.xfeatnetworkplus.model import XFeatModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load PyTorch model
    pt_model = XFeatModel().to(device)
    state_dict = torch.load(pytorch_weights_path, map_location=device, weights_only=False)
    pt_model.load_state_dict(state_dict)
    pt_model.eval()

    # Create wrapper
    wrapper = XFeatONNXWrapper(pt_model).to(device)
    wrapper.eval()

    # Convert numpy to torch tensor
    img_tensor = torch.from_numpy(img_array).to(device)

    with torch.no_grad():
        pt_outputs = wrapper(img_tensor)

    pt_feats, pt_kpts, pt_heatmap = pt_outputs

    # Convert to numpy
    pt_feats = pt_feats.cpu().numpy()
    pt_kpts = pt_kpts.cpu().numpy()
    pt_heatmap = pt_heatmap.cpu().numpy()

    print("✓ PyTorch inference complete")

    # 3. Compare outputs
    print("\n3. Comparing outputs:")
    print("-" * 60)

    def compare_arrays(name, onnx_arr, pt_arr, tolerance=1e-4):
        diff = np.abs(onnx_arr - pt_arr)
        max_diff = diff.max()
        mean_diff = diff.mean()

        # Calculate relative difference (avoid division by zero)
        rel_diff = diff / (np.abs(pt_arr) + 1e-8)
        max_rel_diff = rel_diff.max()

        status = "✓ PASS" if max_diff < tolerance else "✗ FAIL"

        print(f"\n{name}:")
        print(f"  Max absolute difference:  {max_diff:.6e}")
        print(f"  Mean absolute difference: {mean_diff:.6e}")
        print(f"  Max relative difference:  {max_rel_diff:.6e}")
        print(f"  Status: {status}")

        return max_diff < tolerance

    all_passed = True
    all_passed &= compare_arrays("Features", onnx_feats, pt_feats, tolerance=1e-3)
    all_passed &= compare_arrays("Keypoints", onnx_kpts, pt_kpts, tolerance=1e-3)
    all_passed &= compare_arrays("Heatmap", onnx_heatmap, pt_heatmap, tolerance=1e-4)

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All outputs match! ONNX model is correct.")
    else:
        print("⚠ Some outputs differ. Please check the differences above.")
    print("=" * 60)


if __name__ == "__main__":
    import os

    # Paths
    onnx_model_path = '/home/hxy/doctor/feature dectect/hloc/Hierarchical-Localization/third_party/xfeatnetworkplus/weights/xfeat_rgb.onnx'
    pytorch_weights_path = '/home/hxy/doctor/feature dectect/hloc/Hierarchical-Localization/third_party/xfeatnetworkplus/weights/best.pth'

    # Use a test image (you can replace this with your own image path)
    # For demonstration, create a random test image
    print("Creating random test image...")
    test_image = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    test_image_path = '/tmp/test_image.jpg'
    test_image.save(test_image_path)

    # Run inference
    run_inference(onnx_model_path, test_image_path)

    # Compare with PyTorch
    print("\n\n")
    compare_with_pytorch(onnx_model_path, pytorch_weights_path, test_image_path)

    # Clean up
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
