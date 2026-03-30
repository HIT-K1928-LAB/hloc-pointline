"""
全局特征提取和相似度计算脚本
支持: gaussvladplus, gaussvladplusvgg, gaussvladsimple, netvladtorch
"""

import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple

# 导入特征提取器
from .gaussvladplus import GaussVLADplushloc
from .gaussvladplusvgg import gaussvladplusvgghloc
from .gaussvladsimple import gaussvladsimplehloc
from .netvladtorch import gaussvladplusvgghloc as netvladtorchhloc


class GlobalFeatureExtractor:
    """全局特征提取器"""

    def __init__(self, model_name: str = 'gaussvladplus', device: str = 'cuda'):
        """
        Args:
            model_name: 模型名称,可选 'gaussvladplus', 'gaussvladplusvgg', 'gaussvladsimple', 'netvladtorch'
            device: 设备 ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name

        # 创建模型
        if model_name == 'gaussvladplus':
            model_class = GaussVLADplushloc
        elif model_name == 'gaussvladplusvgg':
            model_class = gaussvladplusvgghloc
        elif model_name == 'gaussvladsimple':
            model_class = gaussvladsimplehloc
        elif model_name == 'netvladtorch':
            model_class = netvladtorchhloc
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # 初始化模型
        self.model = model_class({})
        self.model.eval()
        self.model.to(self.device)

        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize((480, 640)),  # 统一尺寸
            transforms.ToTensor(),
        ])

    def extract_features(self, image_path: str) -> np.ndarray:
        """
        提取单张图像的全局特征

        Args:
            image_path: 图像路径

        Returns:
            全局特征向量 (numpy array)
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # 提取特征
        with torch.no_grad():
            output = self.model({'image': image_tensor})
            feature = output['global_descriptor'].cpu().numpy()

        return feature.squeeze()


def compute_similarity_matrix(features: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    计算特征之间的余弦相似度矩阵

    Args:
        features: 字典 {图像名: 特征向量}

    Returns:
        相似度矩阵 (DataFrame)
    """
    image_names = list(features.keys())
    n = len(image_names)
    similarity_matrix = np.zeros((n, n))

    # 计算相似度矩阵
    for i in range(n):
        for j in range(n):
            feat_i = features[image_names[i]]
            feat_j = features[image_names[j]]

            # 余弦相似度
            similarity = np.dot(feat_i, feat_j) / (
                np.linalg.norm(feat_i) * np.linalg.norm(feat_j) + 1e-8
            )
            similarity_matrix[i, j] = similarity

    # 创建DataFrame
    df = pd.DataFrame(similarity_matrix, index=image_names, columns=image_names)
    return df


def save_similarity_results(similarity_df: pd.DataFrame, output_path: str):
    """
    保存相似度结果到CSV文件

    Args:
        similarity_df: 相似度矩阵DataFrame
        output_path: 输出路径
    """
    # 保存完整矩阵
    similarity_df.to_csv(output_path)
    print(f"相似度矩阵已保存到: {output_path}")

    # 保存每张图像最相似的前3张图像
    output_dir = Path(output_path).parent
    top3_path = output_dir / f"{Path(output_path).stem}_top3.txt"

    with open(top3_path, 'w') as f:
        f.write(f"相似度分析结果 - {Path(output_path).stem}\n")
        f.write("=" * 80 + "\n\n")

        for img_name in similarity_df.index:
            similarities = similarity_df[img_name].sort_values(ascending=False)

            # 排除自身
            similarities = similarities[similarities.index != img_name]

            f.write(f"图像: {img_name}\n")
            f.write("-" * 80 + "\n")
            for rank, (similar_img, sim_score) in enumerate(similarities.head(3).items(), 1):
                f.write(f"  {rank}. {similar_img}: {sim_score:.6f}\n")
            f.write("\n")

    print(f"Top-3相似图像已保存到: {top3_path}")


def process_images_with_model(
    image_dir: str,
    model_name: str,
    output_dir: str
):
    """
    使用指定模型处理图像并计算相似度

    Args:
        image_dir: 图像目录路径
        model_name: 模型名称
        output_dir: 输出目录
    """
    print(f"\n{'='*80}")
    print(f"使用模型: {model_name}")
    print(f"{'='*80}\n")

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 初始化特征提取器
    extractor = GlobalFeatureExtractor(model_name=model_name)

    # 获取所有图像
    image_dir = Path(image_dir)
    image_files = sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.jpg'))

    if len(image_files) == 0:
        print(f"警告: 在 {image_dir} 中未找到图像文件")
        return

    print(f"找到 {len(image_files)} 张图像")

    # 提取所有图像的特征
    print("\n提取全局特征...")
    features = {}
    for img_path in tqdm(image_files, desc=f"提取特征 ({model_name})"):
        feature = extractor.extract_features(str(img_path))
        features[img_path.name] = feature

    print(f"特征维度: {list(features.values())[0].shape}")

    # 计算相似度矩阵
    print("\n计算相似度矩阵...")
    similarity_df = compute_similarity_matrix(features)

    # 保存结果
    output_path = output_dir / f"similarity_{model_name}.csv"
    save_similarity_results(similarity_df, str(output_path))

    # 打印统计信息
    print(f"\n统计信息 ({model_name}):")
    print(f"  - 平均相似度: {similarity_df.values[np.triu_indices_from(similarity_df.values, k=1)].mean():.6f}")
    print(f"  - 最高相似度: {similarity_df.values[np.triu_indices_from(similarity_df.values, k=1)].max():.6f}")
    print(f"  - 最低相似度: {similarity_df.values[np.triu_indices_from(similarity_df.values, k=1)].min():.6f}")

    return similarity_df


def main():
    """主函数"""
    # 配置参数
    image_dir = "/home/hxy/doctor/feature dectect/hloc/Hierarchical-Localization/hloc/extractors/vlad_test"
    output_dir = "/home/hxy/doctor/feature dectect/hloc/Hierarchical-Localization/hloc/extractors/vlad_test/results"

    # 模型列表
    model_names = [
        'gaussvladplus',
        'gaussvladplusvgg',
        'gaussvladsimple',
        'netvladtorch'
    ]

    # 处理每个模型
    all_results = {}
    for model_name in model_names:
        try:
            similarity_df = process_images_with_model(
                image_dir=image_dir,
                model_name=model_name,
                output_dir=output_dir
            )
            all_results[model_name] = similarity_df
        except Exception as e:
            print(f"\n错误: 处理模型 {model_name} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 生成汇总报告
    print("\n" + "="*80)
    print("汇总报告")
    print("="*80)

    summary_path = Path(output_dir) / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("全局特征相似度分析汇总报告\n")
        f.write("="*80 + "\n\n")

        for model_name, sim_df in all_results.items():
            f.write(f"模型: {model_name}\n")
            f.write("-"*80 + "\n")

            triu_values = sim_df.values[np.triu_indices_from(sim_df.values, k=1)]
            f.write(f"  平均相似度: {triu_values.mean():.6f}\n")
            f.write(f"  最高相似度: {triu_values.max():.6f}\n")
            f.write(f"  最低相似度: {triu_values.min():.6f}\n")
            f.write(f"  标准差: {triu_values.std():.6f}\n\n")

        # 比较不同模型
        f.write("\n模型对比\n")
        f.write("="*80 + "\n")
        f.write(f"{'模型名称':<20} {'平均相似度':<15} {'最高相似度':<15} {'最低相似度':<15}\n")
        f.write("-"*80 + "\n")

        for model_name, sim_df in all_results.items():
            triu_values = sim_df.values[np.triu_indices_from(sim_df.values, k=1)]
            f.write(f"{model_name:<20} {triu_values.mean():<15.6f} {triu_values.max():<15.6f} {triu_values.min():<15.6f}\n")

    print(f"\n汇总报告已保存到: {summary_path}")

    print("\n所有处理完成！")


if __name__ == "__main__":
    main()
