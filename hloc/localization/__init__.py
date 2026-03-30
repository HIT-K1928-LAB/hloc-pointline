"""
Localization module with enhanced 3D point recall.

This module implements the innovations from Section 4.2-4.3 of the paper:
"基于虚拟视点的位姿优化算法研究"

Key features:
- Co-visibility based 3D point recall
- Spatial consistency based recall
- Virtual viewpoint generation
- Iterative pose optimization
"""

from .enhanced_recall import (
    CoVisibilityRecall,
    SpatialConsistencyRecall,
    Enhanced3DPointRecall,
    VirtualViewpointGenerator,
    IterativePoseOptimizer,
    enhanced_point_recall_pipeline,
)

__all__ = [
    'CoVisibilityRecall',
    'SpatialConsistencyRecall',
    'Enhanced3DPointRecall',
    'VirtualViewpointGenerator',
    'IterativePoseOptimizer',
    'enhanced_point_recall_pipeline',
]
