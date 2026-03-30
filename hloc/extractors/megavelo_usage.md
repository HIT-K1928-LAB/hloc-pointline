# MEgaVelo: Enhanced Visual Localization with Multi-Scale Feature Fusion

## Overview

This implementation integrates the innovations from the research paper "基于虚拟视点的位姿优化算法研究" into the hloc framework. The model provides enhanced visual localization capabilities through multi-scale feature extraction, adaptive fusion, and improved aggregation.

## Key Innovations

### 1. Multi-Scale Feature Extraction
- Extracts features from multiple resolutions (1/4, 1/8, 1/16, 1/32)
- Supports VGG16, ResNet18, and EfficientNet-B3 backbones
- L2 normalization at each scale for better fusion

### 2. Adaptive Weighted Feature Fusion
- **Cross-scale attention**: Models relationships between different scales
- **Channel attention**: SE-block style attention for each scale
- **Spatial attention**: Focuses on important spatial regions
- **Learnable scale weights**: Adaptive weighting of different scales

### 3. Optimized Keypoint Selection
- Multi-scale attention score generation
- Non-maximum suppression (NMS) for spatial distribution
- Spatial coherence filtering
- Adaptive thresholding based on feature quality

### 4. Enhanced VLAD Aggregation
- **Ghost clusters**: Additional clusters for better representation
- **Attention mechanism**: Dynamic cluster weighting
- **Improved normalization**: Better convergence and stability
- **Soft assignment**: Robust feature-to-cluster assignment

## Installation

Ensure you have the required dependencies:

```bash
pip install torch torchvision numpy scipy
```

## Usage

### Basic Usage with hloc

```python
from hloc import extract_features, match_features, pairs_from_retrieval
from hloc.utils import read_image

# Configure MEgaVelo
conf = {
    'backbone_name': 'efficientnet_b3',  # or 'vgg16', 'resnet18'
    'num_clusters': 64,
    'num_keypoints': 64,
    'use_attention': True,
    'use_nms': True,
    'num_ghost_clusters': 8,
    'pretrained': True
}

# Extract global features for image retrieval
outputs = extract_features(
    main_conf,
    images_dir,
    image_list,
    feature_dir,
    extractor_name='megavelo',
    conf=conf
)
```

### Standalone Usage

```python
import torch
from PIL import Image
from torchvision import transforms
from hloc.extractors.megavelo import MEgaVelo

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MEgaVelo(
    backbone_name='efficientnet_b3',
    num_clusters=64,
    num_keypoints=64,
    use_attention=True,
    pretrained=True
).to(device)
model.eval()

# Load and preprocess image
image = Image.open('query.jpg')
preprocess = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor()
])
image_tensor = preprocess(image).unsqueeze(0).to(device)  # [1, 3, H, W]

# Extract global descriptor
with torch.no_grad():
    descriptor = model(image_tensor)  # [1, num_clusters * total_channels]

print(f"Descriptor shape: {descriptor.shape}")
print(f"Descriptor norm: {torch.norm(descriptor, dim=1).item()}")
```

### With Attention Visualization

```python
# Extract descriptor with attention maps
with torch.no_grad():
    descriptor, aux_info = model(image_tensor, return_attention=True)

# Access attention information
fusion_attention = aux_info['fusion_attention']
keypoint_attention = aux_info['keypoint_attention']
keypoint_info = aux_info['keypoint_info']

print(f"Scale weights: {fusion_attention['scale_weights']}")
print(f"Channel weights: {fusion_attention['channel_weights']}")
print(f"Top keypoint scores: {keypoint_info['scores'][:, :5]}")
```

### Using with hloc Pipeline

```python
from hloc import extract_features, pairs_from_retrieval, match_features
from pathlib import Path

# Configuration
images = Path('datasets/aachen/images/')
outputs = Path('outputs/aachen/')

# 1. Extract global features with MEgaVelo
global_features = extract_features(
    main_conf,
    images,
    image_list,
    outputs / 'global_feats',
    extractor_name='megavelo',
    conf={
        'backbone_name': 'efficientnet_b3',
        'num_clusters': 64,
        'num_keypoints': 64
    }
)

# 2. Perform image retrieval for pose estimation
pairs = pairs_from_retrieval(
    main_conf,
    global_features,
    num_matched=5
)

# 3. Extract local features and match
local_features = extract_features(
    main_conf,
    images,
    image_list,
    outputs / 'local_feats',
    extractor_name='superpoint'
)

matches = match_features(
    main_conf,
    pairs,
    local_features,
    matcher_name='superglue'
)
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backbone_name` | str | `'efficientnet_b3'` | Backbone network (vgg16, resnet18, efficientnet_b3) |
| `num_clusters` | int | `64` | Number of VLAD clusters |
| `num_keypoints` | int | `64` | Number of keypoints to select |
| `use_attention` | bool | `True` | Enable attention mechanisms |
| `use_nms` | bool | `True` | Enable NMS for keypoint selection |
| `num_ghost_clusters` | int | `8` | Number of ghost clusters in VLAD |
| `pretrained` | bool | `True` | Use pretrained backbone |
| `checkpoint_path` | str | `None` | Path to pretrained weights |

## Model Architecture

```
MEgaVelo
├── MultiScaleBackbone
│   ├── Scale 1 (1/4 resolution)
│   ├── Scale 2 (1/8 resolution)
│   ├── Scale 3 (1/16 resolution)
│   └── Scale 4 (1/32 resolution)
├── AdaptiveFusionModule
│   ├── CrossScaleAttention (for each scale)
│   ├── ChannelAttention (SE-block)
│   ├── SpatialAttention
│   └── FusionConv
├── AdaptiveKeypointSelection
│   ├── AttentionConv
│   ├── CoherenceConv
│   └── NMS
└── EnhancedVLAD
    ├── Centroids (main + ghost)
    ├── SoftAssignment
    ├── ClusterAttention
    └── GhostClusterWeights
```

## Performance

Based on the paper's experiments on Aachen Day-Night dataset:

| Method | 0.25m, 2° | 0.5m, 5° | 5m, 10° | Time (ms) |
|--------|-----------|-----------|-----------|-----------|
| Hloc | 88.6% | 95.4% | 98.3% | 288 |
| MEgaVelo (fast) | 89.0% | 95.6% | 98.4% | 347 |
| MEgaVelo (full) | 89.0% | 96.0% | 98.4% | 509 |

## Training

To train MEgaVelo on your own dataset:

```python
import torch
import torch.nn as nn
from hloc.extractors.megavelo import MEgaVelo

# Initialize model
model = MEgaVelo(
    backbone_name='efficientnet_b3',
    num_clusters=64,
    pretrained=True
)

# Define loss function (e.g., contrastive loss for retrieval)
criterion = nn.TripletMarginLoss(margin=0.3)

# Optimizer
optimizer = torch.optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.fusion.parameters(), 'lr': 1e-4},
    {'params': model.vlad.parameters(), 'lr': 1e-4}
])

# Training loop
for epoch in range(num_epochs):
    for anchor, positive, negative in dataloader:
        # Extract descriptors
        anchor_desc = model(anchor)
        positive_desc = model(positive)
        negative_desc = model(negative)

        # Compute loss
        loss = criterion(anchor_desc, positive_desc, negative_desc)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Tips for Best Performance

1. **Backbone Selection**: Use `efficientnet_b3` for best accuracy, `vgg16` for faster inference
2. **Number of Clusters**: Increase `num_clusters` (e.g., 128) for larger datasets
3. **Ghost Clusters**: Keep `num_ghost_clusters` around 10-15% of `num_clusters`
4. **Key Points**: Adjust `num_keypoints` based on image resolution (64 for 640x480)
5. **Attention**: Enable `use_attention` for better accuracy, disable for faster inference

## Troubleshooting

### Out of Memory
- Reduce `num_keypoints` or `num_clusters`
- Use a smaller backbone (e.g., vgg16 instead of efficientnet_b3)
- Process images in smaller batches

### Poor Retrieval Performance
- Ensure images are properly preprocessed (normalized to [0,1])
- Try different backbone networks
- Increase `num_clusters` for more discriminative descriptors
- Fine-tune on your specific dataset

### Slow Inference
- Set `use_attention=False` for faster but slightly less accurate results
- Use `vgg16` backbone instead of `efficientnet_b3`
- Reduce `num_keypoints`

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{megavelo2024,
  title={基于虚拟视点的位姿优化算法研究},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024}
}
```

## License

This implementation follows the same license as the hloc framework.

## Contact

For questions or issues, please open an issue on the hloc GitHub repository.
