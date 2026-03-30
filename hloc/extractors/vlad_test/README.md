# 全局特征提取和相似度分析

本脚本用于对vlad_test文件夹中的图像使用不同的全局特征提取器提取特征，并计算图像之间的相似度。

## 支持的模型

- **gaussvladplus**: 使用EfficientNet-B3作为骨干网络的高斯VLAD模型
- **gaussvladplusvgg**: 使用VGG16作为骨干网络的高斯VLAD模型
- **gaussvladsimple**: 简化版高斯VLAD模型
- **netvladtorch**: PyTorch实现的NetVLAD模型

## 使用方法

### 基本使用

```bash
cd /home/hxy/doctor/feature\ dectect/hloc/Hierarchical-Localization
python -m hloc.extractors.test_global_match
```

### 自定义参数

可以修改脚本中的以下参数：

```python
# 图像目录
image_dir = "path/to/your/images"

# 输出目录
output_dir = "path/to/output"

# 选择使用的模型
model_names = [
    'gaussvladplus',
    'gaussvladplusvgg',
    'gaussvladsimple',
    'netvladtorch'
]
```

## 输出结果

脚本会在输出目录中生成以下文件：

### 1. 相似度矩阵文件 (CSV)
- `similarity_gaussvladplus.csv`
- `similarity_gaussvladplusvgg.csv`
- `similarity_gaussvladsimple.csv`
- `similarity_netvladtorch.csv`

每个CSV文件包含所有图像之间的余弦相似度矩阵。

### 2. Top-3相似图像 (TXT)
- `similarity_gaussvladplus_top3.txt`
- `similarity_gaussvladplusvgg_top3.txt`
- `similarity_gaussvladsimple_top3.txt`
- `similarity_netvladtorch_top3.txt`

每个文件列出了每张图像最相似的前3张图像及其相似度分数。

### 3. 汇总报告 (TXT)
- `summary.txt`

包含所有模型的统计信息和对比分析。

## 结果解读

### 相似度分数范围
- **1.0**: 完全相同（图像与自身）
- **0.9-1.0**: 非常相似
- **0.7-0.9**: 相似
- **0.5-0.7**: 中等相似
- **<0.5**: 不太相似

### 模型性能对比

根据测试结果：

| 模型 | 平均相似度 | 最高相似度 | 最低相似度 | 特征维度 |
|------|-----------|-----------|-----------|---------|
| gaussvladplusvgg | 0.900 | 0.969 | 0.838 | 65536 |
| gaussvladplus | 0.878 | 0.964 | 0.817 | 49152 |
| gaussvladsimple | 0.877 | 0.970 | 0.819 | 49152 |
| netvladtorch | 0.688 | 0.926 | 0.515 | 32768 |

**观察：**
- `gaussvladplusvgg` 获得最高的平均相似度，说明VGG16骨干网络在这个任务上表现最好
- `netvladtorch` 的相似度分布更广（标准差更大），可能更适合区分性较强的任务
- 所有模型都能正确识别出相似的图像组（ori20-23为一组，ori1292-1304为另一组）

## 代码结构

```python
GlobalFeatureExtractor
├── __init__()           # 初始化模型
├── extract_features()   # 提取单张图像特征
└── compute_similarity() # 计算相似度矩阵

主函数流程：
1. 加载图像列表
2. 对每个模型：
   a. 初始化特征提取器
   b. 提取所有图像特征
   c. 计算相似度矩阵
   d. 保存结果
3. 生成汇总报告
```

## 依赖项

- PyTorch
- NumPy
- Pandas
- Pillow (PIL)
- tqdm

## 注意事项

1. 确保所有模型权重文件存在于 `hloc/extractors/weights/` 目录
2. 图像会被统一resize到 480x640 分辨率
3. 使用CUDA（如果可用）加速计算
4. 相似度计算使用余弦相似度

## 示例输出

```
相似度分析结果 - gaussvladplus
================================================================================

图像: loop_image_ori20_image.png
--------------------------------------------------------------------------------
  1. loop_image_ori21_image.png: 0.960280
  2. loop_image_ori22_image.png: 0.948767
  3. loop_image_ori23_image.png: 0.939960
```

这表明loop_image_ori20_image.png与loop_image_ori21_image.png最相似（0.960）。
