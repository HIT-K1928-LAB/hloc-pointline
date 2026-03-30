# `hloc.visualize_point_line_map` 使用说明

## 1. 功能简介

`hloc.visualize_point_line_map` 用于可视化 `pipeline_joint_point_line_v2.py` 生成的：

- 3D 点云
- 3D line
- 相机位姿

它支持两类输出：

1. 交互式 `HTML`
   适合在浏览器中旋转、缩放、查看点云、线段和相机 frustum。

2. `PLY`
   适合在 MeshLab、CloudCompare、Open3D 等软件中查看。

脚本位置：

- [visualize_point_line_map.py](/home/hxy/doctor/feature%20dectect/hloc/Hierarchical-Localization/hloc/visualize_point_line_map.py)

---

## 2. 适用输入

该脚本默认面向 `pipeline_joint_point_line_v2.py` 的输出目录，至少需要以下两个文件：

- `sfm_joint-xfeat-mlsd+NN/`
- `line_map.h5`

典型输出目录例如：

```bash
/media/hxy/PortableSSD/Hierarchical-Localizationv2/outputs/aachen_joint_point_line_v2_netvlad
```

在这个目录下，脚本默认会自动寻找：

- `sfm_joint-xfeat-mlsd+NN`
- `line_map.h5`

如果你的模型目录名或线地图文件名不同，也可以通过命令行参数手动指定。

---

## 3. 运行环境

建议在 `hl` conda 环境中运行。

脚本依赖：

- `pycolmap`
- `plotly`
- `numpy`
- `h5py`

这些依赖在当前环境中已经验证可用。

---

## 4. 基本用法

先进入仓库根目录：

```bash
cd "/home/hxy/doctor/feature dectect/hloc/Hierarchical-Localization"
```

最简单的运行方式：

```bash
python -m hloc.visualize_point_line_map \
  --outputs /media/hxy/PortableSSD/Hierarchical-Localizationv2/outputs/aachen_joint_point_line_v2_netvlad
```

这会生成默认的交互式可视化文件：

```bash
/media/hxy/PortableSSD/Hierarchical-Localizationv2/outputs/aachen_joint_point_line_v2_netvlad/point_line_map_visualization.html
```

---

## 5. 导出 PLY

如果你希望在 MeshLab 或 Open3D 中查看，请加上：

```bash
--export_ply
```

示例：

```bash
python -m hloc.visualize_point_line_map \
  --outputs /media/hxy/PortableSSD/Hierarchical-Localizationv2/outputs/aachen_joint_point_line_v2_netvlad \
  --export_ply
```

导出后会额外生成两个文件：

- `point_cloud_only.ply`
- `point_line_camera_overlay.ply`

说明：

- `point_cloud_only.ply` 只包含 3D 点云。
- `point_line_camera_overlay.ply` 包含 3D 点云、3D line 和相机位姿。

注意：

- 为了兼容 MeshLab / Open3D，脚本不是把 3D line 和相机 frustum 以“线元 primitive”导出，而是把它们采样成彩色点云后写入 `PLY`。
- 因此在 `PLY` 中，绿色点表示 3D line，蓝色点表示相机 frustum 边，橙色点表示相机中心。

---

## 6. 常用命令示例

## 6.1 仅导出 HTML

```bash
python -m hloc.visualize_point_line_map \
  --outputs /media/hxy/PortableSSD/Hierarchical-Localizationv2/outputs/aachen_joint_point_line_v2_netvlad
```

## 6.2 同时导出 HTML 和 PLY

```bash
python -m hloc.visualize_point_line_map \
  --outputs /media/hxy/PortableSSD/Hierarchical-Localizationv2/outputs/aachen_joint_point_line_v2_netvlad \
  --export_ply
```

## 6.3 轻量级预览

如果点云太大、浏览器太卡，建议先降低显示规模：

```bash
python -m hloc.visualize_point_line_map \
  --outputs /media/hxy/PortableSSD/Hierarchical-Localizationv2/outputs/aachen_joint_point_line_v2_netvlad \
  --max_points 50000 \
  --max_lines 5000 \
  --max_cameras 500
```

## 6.4 轻量级预览并导出 PLY

```bash
python -m hloc.visualize_point_line_map \
  --outputs /media/hxy/PortableSSD/Hierarchical-Localizationv2/outputs/aachen_joint_point_line_v2_netvlad \
  --max_points 50000 \
  --max_lines 5000 \
  --max_cameras 500 \
  --export_ply
```

## 6.5 手动指定模型目录和输出 HTML

```bash
python -m hloc.visualize_point_line_map \
  --outputs /media/hxy/PortableSSD/Hierarchical-Localizationv2/outputs/aachen_joint_point_line_v2_netvlad \
  --reference_sfm /media/hxy/PortableSSD/Hierarchical-Localizationv2/outputs/aachen_joint_point_line_v2_netvlad/sfm_joint-xfeat-mlsd+NN \
  --line_map /media/hxy/PortableSSD/Hierarchical-Localizationv2/outputs/aachen_joint_point_line_v2_netvlad/line_map.h5 \
  --output_html /tmp/aachen_point_line_map_visualization.html
```

---

## 7. 输出内容说明

## 7.1 HTML 中包含什么

在 `HTML` 交互式窗口中会显示：

- 3D points
- 3D lines
- Cameras
- Camera centers

其中：

- 3D 点使用 SfM 点云的原始颜色。
- 3D line 使用绿色折线显示。
- 相机位姿使用蓝色 frustum 显示。
- 相机中心使用蓝色点显示，并可悬停查看图像名。

## 7.2 PLY 中包含什么

- `point_cloud_only.ply`
  - 只保留过滤和采样后的 3D 点云。

- `point_line_camera_overlay.ply`
  - 包含点云
  - 包含采样后的 3D line
  - 包含采样后的相机 frustum
  - 包含相机中心

---

## 8. 关键参数说明

以下是最常用的参数：

- `--outputs`
  - 必填。
  - 指向 pipeline 输出目录。

- `--reference_sfm`
  - 可选。
  - 默认使用 `<outputs>/sfm_joint-xfeat-mlsd+NN`。

- `--line_map`
  - 可选。
  - 默认使用 `<outputs>/line_map.h5`。

- `--output_html`
  - 可选。
  - 默认输出到 `<outputs>/point_line_map_visualization.html`。

- `--max_points`
  - 最多显示多少 3D 点。
  - 默认 `200000`。

- `--max_lines`
  - 最多显示多少 3D line。
  - 默认 `20000`。

- `--max_cameras`
  - 最多显示多少相机。
  - 默认 `0`，表示全部显示。

- `--max_reproj_error`
  - 点云过滤阈值，只有重投影误差不超过该值的 3D 点才会被显示。
  - 默认 `6.0`。

- `--min_track_length`
  - 点云过滤阈值，只有 track length 不小于该值的 3D 点才会被显示。
  - 默认 `2`。

- `--min_line_support_count`
  - 线地图过滤阈值，只有支持视图数不小于该值的 3D line 才会被显示。
  - 默认 `3`。

- `--max_line_reproj_error`
  - 线地图过滤阈值，只有平均线重投影误差不超过该值的 3D line 才会被显示。
  - 默认 `20.0`。

- `--camera_size`
  - 相机 frustum 大小。
  - 默认 `1.0`。

- `--camera_line_width`
  - 相机 frustum 线宽。
  - 默认 `2.0`。

- `--camera_center_size`
  - 相机中心点大小。
  - 默认 `2.5`。

- `--export_ply`
  - 是否导出 `PLY`。

- `--line_samples_per_segment`
  - 导出 `PLY` 时，每条 3D line 采样多少个点。
  - 默认 `24`。

- `--camera_edge_samples`
  - 导出 `PLY` 时，每条相机 frustum 边采样多少个点。
  - 默认 `8`。

- `--seed`
  - 随机采样种子。
  - 默认 `7`。

---

## 9. 过滤与采样策略

为了避免 Aachen 这种大场景直接可视化时过慢，脚本做了几层过滤与采样：

- 对 3D 点：
  - 按 bounding box 过滤明显离群点
  - 按 `max_reproj_error` 过滤高误差点
  - 按 `min_track_length` 过滤弱点轨迹
  - 若数量仍太大，再按 `max_points` 随机采样

- 对 3D line：
  - 按 `min_line_support_count` 过滤支持视图太少的线
  - 按 `max_line_reproj_error` 过滤质量较差的线
  - 若数量仍太大，再按 `max_lines` 随机采样

- 对相机：
  - 若图像过多，则按 `max_cameras` 随机采样

---

## 10. 常见问题

## 10.1 浏览器打开 HTML 很卡怎么办

先减小：

- `--max_points`
- `--max_lines`
- `--max_cameras`

推荐起步值：

```bash
--max_points 50000 --max_lines 5000 --max_cameras 500
```

## 10.2 为什么 PLY 里看起来线不是“真正的线”

这是兼容性设计决定的。

- 很多通用点云工具对 `PLY` 的 line primitive 支持并不稳定。
- 为了保证 MeshLab / Open3D 都能看，脚本把 3D line 和相机 frustum 都采样成了点云。

因此在 `PLY` 中看到的是：

- 点云本体
- 由绿色采样点构成的 3D line
- 由蓝色采样点构成的相机 frustum
- 由橙色点构成的相机中心

## 10.3 为什么看不到某些线或相机

可能原因：

- 线被 `min_line_support_count` 或 `max_line_reproj_error` 过滤掉了
- 相机被 `max_cameras` 采样限制掉了
- 点云/线/相机数量太大，当前参数下只显示了一个子集

---

## 11. 一个推荐命令

如果你的目标是“先快速看一眼结果，同时还能导出给 MeshLab”，推荐先运行：

```bash
python -m hloc.visualize_point_line_map \
  --outputs /media/hxy/PortableSSD/Hierarchical-Localizationv2/outputs/aachen_joint_point_line_v2_netvlad \
  --max_points 50000 \
  --max_lines 5000 \
  --max_cameras 500 \
  --export_ply
```

这样通常能在可视化质量和交互速度之间取得比较平衡的效果。
