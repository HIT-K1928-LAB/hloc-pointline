import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_coord_scale(coord_scale):
    if isinstance(coord_scale, (tuple, list)):
        if len(coord_scale) != 2:
            raise ValueError('coord_scale tuple/list must have length 2, got {}'.format(len(coord_scale)))
        scale_x = max(float(coord_scale[0]), 1.0)
        scale_y = max(float(coord_scale[1]), 1.0)
        return scale_x, scale_y

    scale = max(float(coord_scale), 1.0)
    return scale, scale


def canonicalize_lines(lines):
    """Canonically order endpoints so line descriptors stay direction-invariant."""
    if lines is None or lines.numel() == 0:
        return lines

    lines = lines.clone()
    dx = torch.abs(lines[:, 0] - lines[:, 2])
    dy = torch.abs(lines[:, 1] - lines[:, 3])
    use_x = dx > dy
    swap_mask = torch.where(use_x, lines[:, 0] > lines[:, 2], lines[:, 1] > lines[:, 3])
    if torch.any(swap_mask):
        lines[swap_mask] = lines[swap_mask][:, [2, 3, 0, 1]]
    return lines


class LineDescriptorHead(nn.Module):
    def __init__(self, in_channels=16, descriptor_dim=64, num_samples=5):
        super(LineDescriptorHead, self).__init__()
        hidden_dim = max(64, descriptor_dim)
        self.descriptor_dim = descriptor_dim
        self.num_samples = num_samples

        self.dense_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, descriptor_dim, kernel_size=1, bias=True),
        )

        self.geometry_embed = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, descriptor_dim),
            nn.ReLU(inplace=True),
        )

        self.aggregator = nn.Sequential(
            nn.Linear(descriptor_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, descriptor_dim),
        )

    def forward_map(self, line_features):
        descriptor_map = self.dense_head(line_features)
        return F.normalize(descriptor_map, p=2, dim=1)

    def extract_descriptors(self, descriptor_map, lines, coord_scale=512.0, num_samples=None):
        if descriptor_map.dim() == 3:
            descriptor_map = descriptor_map.unsqueeze(0)

        if lines is None or lines.numel() == 0:
            return descriptor_map.new_zeros((0, self.descriptor_dim))

        lines = canonicalize_lines(lines.to(descriptor_map.device).float())
        sample_points = self._build_sample_points(lines, num_samples=num_samples)
        sampled_features = self._sample_points(descriptor_map, sample_points, coord_scale=coord_scale)

        mean_features = sampled_features.mean(dim=1)
        max_features = sampled_features.max(dim=1).values
        geometry_features = self.geometry_embed(self._geometry_features(lines, coord_scale=coord_scale))

        fused = torch.cat((mean_features, max_features, geometry_features), dim=-1)
        descriptors = self.aggregator(fused)
        return F.normalize(descriptors, p=2, dim=-1)

    def _build_sample_points(self, lines, num_samples=None):
        num_samples = self.num_samples if num_samples is None else num_samples
        start = lines[:, :2]
        end = lines[:, 2:]
        center = (start + end) * 0.5

        if num_samples > 0:
            t = torch.linspace(0.0, 1.0, steps=num_samples + 2, device=lines.device, dtype=lines.dtype)[1:-1]
            line_points = start.unsqueeze(1) * (1.0 - t.view(1, -1, 1)) + end.unsqueeze(1) * t.view(1, -1, 1)
            return torch.cat((start.unsqueeze(1), center.unsqueeze(1), end.unsqueeze(1), line_points), dim=1)

        return torch.cat((start.unsqueeze(1), center.unsqueeze(1), end.unsqueeze(1)), dim=1)

    def _sample_points(self, descriptor_map, points, coord_scale=512.0):
        _, channels, height, width = descriptor_map.shape
        num_lines, num_points, _ = points.shape

        if num_lines == 0:
            return descriptor_map.new_zeros((0, 0, channels))

        coord_scale_x, coord_scale_y = _parse_coord_scale(coord_scale)
        max_coord_x = max(coord_scale_x - 1.0, 1.0)
        max_coord_y = max(coord_scale_y - 1.0, 1.0)

        x = points[..., 0].clamp(0.0, max_coord_x)
        y = points[..., 1].clamp(0.0, max_coord_y)

        x = x * (width - 1) / max_coord_x if width > 1 else torch.zeros_like(x)
        y = y * (height - 1) / max_coord_y if height > 1 else torch.zeros_like(y)

        x = x / max(width - 1, 1) * 2.0 - 1.0
        y = y / max(height - 1, 1) * 2.0 - 1.0
        grid = torch.stack((x, y), dim=-1).view(1, num_lines * num_points, 1, 2)

        sampled = F.grid_sample(
            descriptor_map,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
        sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)
        return sampled.reshape(num_lines, num_points, channels)

    @staticmethod
    def _geometry_features(lines, coord_scale=512.0):
        start = lines[:, :2]
        end = lines[:, 2:]
        center = (start + end) * 0.5
        delta = end - start
        line_length = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
        direction = delta / line_length
        coord_scale_x, coord_scale_y = _parse_coord_scale(coord_scale)
        center_scale = center.new_tensor([coord_scale_x, coord_scale_y])
        center = center / center_scale.unsqueeze(0)
        diagonal = max((coord_scale_x ** 2 + coord_scale_y ** 2) ** 0.5, 1.0)
        length = line_length / diagonal
        return torch.cat((center, direction, length), dim=-1)
