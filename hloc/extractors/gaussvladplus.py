from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms as T
import numpy as np
import logging
from scipy.io import loadmat
# from sklearn.mixture import GaussianMixture
import math
from torchvision.models.efficientnet import EfficientNet_B3_Weights

from ..utils.base_model import BaseModel
EPS = 1e-6

netvlad_path = Path(__file__).parent / '../weights/netvlad/'
logger = logging.getLogger(__name__)

def attention(query, key, value, alpha, eps=1e-6):
    score = torch.einsum('bdk,bdn->bkn', query, key)
    prob = F.softmax(alpha * score, dim=1)
    denom = prob.sum(dim=-1, keepdim=True).clamp_min_(eps).expand_as(prob)
    prob = torch.div(prob, denom)
    return torch.einsum('bkn,bdn->bdk', prob, value)




class AttnVLADLayer(nn.Module):
    def __init__(self, input_dim=512, K=64):
        super().__init__()
        centers = nn.parameter.Parameter(torch.empty([1, input_dim, K])) # D×K
        nn.init.xavier_uniform_(centers)
        self.register_parameter('centers', centers)
        self.alpha = nn.Parameter(torch.tensor(300.0))
        
        self.output_dim = input_dim * K
        self.cluster_weights = nn.Parameter(torch.ones([1, 1, K]))
        print('Loaded VLAD layer.')

    def init_params(self, clsts, traindescs):
        self.centers.data = torch.from_numpy(clsts).T[None]# 1KD
        
        clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)#K ×D
        dots = np.dot(clstsAssign, traindescs.T)# dots：k×N   clstsAssign：K ×D   traindescs.T：D × N
        dots.sort(0)
        dots = dots[::-1, :]# 从大到小 k×N
        alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()# 最相似与次相似的差 k×N
        self.alpha.data = torch.tensor(alpha)
        

    def forward(self, x):
        b = x.size(0)
        # 'bdk,bdn->bkn'
        # centers：D * K ； x: D * N ; x: D * N
        desc = attention(F.normalize(self.centers), x, x, self.alpha) # D × K
        desc = desc - self.centers  #centers: D×K   desc:D × K
        desc = F.normalize(desc, dim=1)
        desc = self.cluster_weights * desc
        desc = desc.reshape(b, -1)
        desc = F.normalize(desc, dim=1)
        return desc



class NetVLADLayer(nn.Module):
    def __init__(self, input_dim=512, K=64, score_bias=False, intranorm=True):
        super().__init__()
        self.score_proj = nn.Conv1d(
            input_dim, K, kernel_size=1, bias=score_bias)
        centers = nn.parameter.Parameter(torch.empty([input_dim, K]))
        nn.init.xavier_uniform_(centers)
        self.register_parameter('centers', centers)
        self.intranorm = intranorm
        self.output_dim = input_dim * K

    def forward(self, x):
        b = x.size(0)
        scores = self.score_proj(x)
        scores = F.softmax(scores, dim=1)
        diff = (x.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(-1))
        desc = (scores.unsqueeze(1) * diff).sum(dim=-1)
        if self.intranorm:
            # From the official MATLAB implementation.
            desc = F.normalize(desc, dim=1)
        desc = desc.view(b, -1)
        desc = F.normalize(desc, dim=1)
        return desc






class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        input = input - input.mean(dim=[-2, -1], keepdim=True)
        return F.normalize(input, p=2, dim=self.dim)

def get_backbone(name, l2_norm=True, from_matconvmat=True):
    if name == 'vgg':
        # l1 [:3] l2 [:8] l3 [:15] l4[:22] l5[:29]
        backbone = list(models.vgg16(pretrained=True).children())[0]   # Remove classification head.
        layers = list(backbone.children())[:29] #  - ReLu - MaxPool
        for l in layers[:22]: 
            for p in l.parameters():
                p.requires_grad = False
        output_dim = 512
    elif name == 'efficient':
        # BN 的参数会在forward时更新，应将其冻结
        backbone = list(models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT).children())[0]  # Remove classification head.
        layers = list(backbone.children())[:-1]
        for l in layers[:-1]: 
            for p in l.parameters():
                p.requires_grad = False
        output_dim = 384
    else:
        raise ValueError(f'{name} not find.')
    
    if l2_norm:
        layers.append(L2Norm())
    backbone = nn.Sequential(*layers)
    # if name == 'vgg' and from_matconvmat:
    #     backbone.load_state_dict(torch.load('./weights/vd16_offtheshelf_conv5_3_max.pth'))
    backbone.output_dim = output_dim 
    return backbone

def get_pool(name, **kwargs):
    if name == 'attnvlad':
        pool = AttnVLADLayer(**kwargs)
    elif name == 'netvlad':
        pool = NetVLADLayer(**kwargs)
    # elif name =="WassersteinFeatureExtractor":
    #     pool = WassersteinFeatureExtractor(**kwargs)
    # elif name == "WassVLADLayer":
    #     pool = WassVLADLayer(**kwargs)
    elif name == "WassersteinFeatureCase2":
        pool = WassersteinFeatureCase2(**kwargs)
    else:
        raise ValueError(f'{name} not find.')
    return pool

EPS = 1e-6

class DiagonalGMMPosterior(nn.Module):
    def __init__(self, input_dim, num_clusters):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(num_clusters, input_dim))  # K × D
        self.log_sigma = nn.Parameter(torch.zeros(num_clusters, input_dim))  # K × D
        self.log_alpha = nn.Parameter(torch.zeros(num_clusters))  # K
        # self.alpha = nn.Parameter(torch.tensor(300.0))
    def forward(self, x):  # x: B × D × N
        # B, D, N = x.shape
        # score = torch.einsum('bdk,bdn->bkn', F.normalize(self.mu.T[None]), x)
        # prob = F.softmax(self.alpha * score, dim=1)
        # return prob
        x = x.permute(0, 2, 1)  # B × N × D
        x_expand = x.unsqueeze(2)  # B × N × 1 × D
        mu_expand = F.normalize(self.mu).unsqueeze(0).unsqueeze(0)  # 1 × 1 × K × D
        sigma_inv = torch.exp(-self.log_sigma).unsqueeze(0).unsqueeze(0)  # 1 × 1 × K × D
        log_sigma_sum = self.log_sigma.sum(dim=1).unsqueeze(0).unsqueeze(0)  # 1 × 1 × K

        diff = x_expand - mu_expand  # B × N × K × D
        dist_term = (diff ** 2 * sigma_inv).sum(dim=-1)  # B × N × K

        logits = -dist_term + self.log_alpha.unsqueeze(0).unsqueeze(0) - 0.5 * log_sigma_sum  # B × N × K
        # logits = -dist_term  - 0.5 * log_sigma_sum  # B × N × K
        posterior = F.softmax(logits, dim=-1)  # B × N × K
        return posterior.permute(0, 2, 1)  # B × K × N

    def init_params(self, clsts, traindescs):
        clsts = torch.from_numpy(clsts).float()  # K × D
        # print(clsts.shape)
        traindescs = torch.from_numpy(traindescs).float()  # N × D

        clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
        dots = np.dot(clstsAssign, traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :]
        alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        # self.alpha.data = torch.tensor(alpha)
        
        
        K, D = clsts.shape
        N = traindescs.shape[0]

        self.mu.data = clsts.clone()

        # Calculate distance and posterior probabilities (assignment)
        descs = traindescs.unsqueeze(1)  # N × 1 × D
        centers = clsts.unsqueeze(0)  # 1 × K × D
        dist = ((descs - centers) ** 2).sum(dim=2)  # N × K
        assignment = torch.softmax(-dist, dim=1).T  # K × N

        weighted_var = []
        for i in range(K):
            weights = assignment[i]
            weights = weights / (weights.sum() + EPS)
            mean_i = clsts[i]
            diff = traindescs - mean_i
            var = (weights[:, None] * diff ** 2).sum(dim=0)
            weighted_var.append(var)

        sigma = torch.stack(weighted_var, dim=0)  # K × D
        sigma = torch.sqrt(sigma + EPS)
        self.log_sigma.data = sigma.log()


# Wasserstein特征提取模型
class WassersteinFeatureCase2(nn.Module):
    def __init__(self, input_dim, K=64):
        super().__init__()
        self.posterior_estimator = DiagonalGMMPosterior(input_dim, K)
        self.cluster_weights = nn.Parameter(torch.ones([1, 1, K]))
        self.output_dim = 2 * input_dim * K

    def forward(self, x):  # x: B × D × N
        B, D, N = x.shape
        posterior = self.posterior_estimator(x)  # B × K × N

        # 计算每个高斯成分对每个特征的加权系数 γᵢⱼ
        post_sum = posterior.sum(dim=-1, keepdim=True).clamp_min_(EPS)  # B × K × 1
        gamma = torch.div(posterior, post_sum)  # B × K × N (归一化后的后验概率)
        # print("gamma",gamma.shape)

        # 加权均值计算
        mu_hat = torch.einsum('bkn,bdn->bkd', gamma, x)  # B × K × D
     
        # 计算每个高斯成分的均值和方差
        mu_i = self.posterior_estimator.mu.unsqueeze(0).expand(B, -1, -1)  # B × K × D
        sigma_i = torch.exp(self.posterior_estimator.log_sigma).unsqueeze(0).expand(B, -1, -1)  # B × K × D
        # print("x",x.shape,"mu_hat",mu_hat.shape)
        # 计算特征差异
        # print("x.unsqueeze(1)",x.unsqueeze(1).shape,"mu_hat.unsqueeze(2)",mu_hat.unsqueeze(2).shape)
        diff = x.unsqueeze(1).permute(0,1,3,2) - mu_hat.unsqueeze(2)  # B × K × N × D
        
        # print("diff",diff.shape)
        weight = gamma.unsqueeze(-1)  # B × K × N × 1 (后验概率)
        
        # 加权方差计算
        var_hat = (diff ** 2) * weight
        var_hat = var_hat.sum(dim=2) 
        # print("var_hat",var_hat.shape)
        
        sigma_hat = torch.sqrt(var_hat + EPS)
        # print("sigma_hat",sigma_hat.shape)

        # 特征 g_i 计算
        feat_mu = mu_hat - mu_i  # B × K × D
        # print("mu_hat",mu_hat.shape,"mu_i",mu_i.shape)
        # print("sigma_hat",sigma_hat.shape,"sigma_i",sigma_i.shape)
        feat_sigma = sigma_hat - sigma_i  # B × K × D
        feat = torch.cat([feat_mu, feat_sigma], dim=-1).permute(0,2,1) # B * 2D * K
        # print("feat",feat.shape)

        # 归一化
        feat = F.normalize(feat,dim=1)  #  B * 2D * K
        feat = self.cluster_weights * feat  #  B * 2D * K
        feat = feat.reshape(B, -1)  #  B * 2D * K
        # feat = F.normalize(feat, dim=1)

        return feat


class AttnVLAD(nn.Module):
    """
    完整的Wasserstein模型
    输入: (B, 3, H, W) 图像
    输出: (B, 2*K*D) Wasserstein全局描述符
    """
    default_conf = {
        'backbone_name': 'efficient',
        'pool_name': 'WassersteinFeatureCase2',
        'checkpoint_dir': netvlad_path,
        'whiten': False,
    }

    def __init__(self,  K=64, **conf):
        super().__init__()
        self.conf =  {**self.default_conf, **conf}
        
        # 构建骨干网络
        self.backbone = get_backbone(self.conf['backbone_name'], l2_norm=True)
        
        # Wasserstein特征提取模块
        self.pool = get_pool(self.conf['pool_name'], input_dim=self.backbone.output_dim,K=K)
        
        self.model_name = self.conf['backbone_name'].lower() + '-wasserstein'
        
        # 输出维度
        self.output_dim = self.pool.output_dim
        
        # 图像预处理
        self.preprocess = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def init_params(self, clsts, traindescs):
        """初始化参数"""
        self.pool.init_params(clsts, traindescs)
    
    def forward(self, image):
         # 验证输入范围
        assert image.min() >= -1e-3 and image.max() <= 1 + 1e-3
        assert image.shape[1] == 3
        # 图像预处理
        image = self.preprocess(image)
        
        # 提取局部特征
        features = self.backbone(image)  # (B, D, H, W)
        
        # 调整维度为 (B, D, N) 其中 N = H * W
        B, D, H, W = features.size()
        features = features.reshape(B, D, -1)  # (B, D, N)
        
        # Wasserstein特征提取
        global_desc = self.pool(features)
        
        return global_desc



class NetVLAD(nn.Module):
    default_conf = {
        'model_name': 'VGG16-NetVLAD-Pitts30K',
        'backbone_name': 'vgg',
        'pool_name': 'netvlad',
        'checkpoint_dir': netvlad_path,
        'whiten': False,
    }


    def __init__(self, **conf):
        super().__init__()
        self.conf = conf = {**self.default_conf, **conf}
        
        checkpoint = conf['checkpoint_dir'] / str(conf['model_name'] + '.mat')
        if not checkpoint.exists():
            logger.warning('No checkpoint find in {checkpoint}. Using default params.')
        # Remove classification head.
        backbone = list(models.vgg16().children())[0]
        # Remove last ReLU + MaxPool2d.
        self.backbone = nn.Sequential(*list(backbone.children())[: -2])

        self.pool = get_pool(conf['pool_name'], input_dim=self.backbone.output_dim)

        if conf['whiten']:
            self.whiten = nn.Linear(self.pool.output_dim, 4096)

        # Parse MATLAB weights using https://github.com/uzh-rpg/netvlad_tf_open
        mat = loadmat(checkpoint, struct_as_record=False, squeeze_me=True)

        # CNN weights.
        for layer, mat_layer in zip(self.backbone.children(),
                                    mat['net'].layers):
            if isinstance(layer, nn.Conv2d):
                w = mat_layer.weights[0]  # Shape: S x S x IN x OUT
                b = mat_layer.weights[1]  # Shape: OUT
                # Prepare for PyTorch - enforce float32 and right shape.
                # w should have shape: OUT x IN x S x S
                # b should have shape: OUT
                w = torch.tensor(w).float().permute([3, 2, 0, 1])
                b = torch.tensor(b).float()
                # Update layer weights.
                layer.weight = nn.Parameter(w)
                layer.bias = nn.Parameter(b)

        # NetVLAD weights.
        score_w = mat['net'].layers[30].weights[0]  # D x K
        # centers are stored as opposite in official MATLAB code
        center_w = -mat['net'].layers[30].weights[1]  # D x K
        # Prepare for PyTorch - make sure it is float32 and has right shape.
        # score_w should have shape K x D x 1
        # center_w should have shape D x K
        score_w = torch.tensor(score_w).float().permute([1, 0]).unsqueeze(-1)
        center_w = torch.tensor(center_w).float()
        # Update layer weights.
        self.pool.score_proj.weight = nn.Parameter(score_w)
        self.pool.centers = nn.Parameter(center_w[None])

        # Whitening weights.
        if conf['whiten']:
            w = mat['net'].layers[33].weights[0]  # Shape: 1 x 1 x IN x OUT
            b = mat['net'].layers[33].weights[1]  # Shape: OUT
            # Prepare for PyTorch - make sure it is float32 and has right shape
            w = torch.tensor(w).float().squeeze().permute([1, 0])  # OUT x IN
            b = torch.tensor(b.squeeze()).float()  # Shape: OUT
            # Update layer weights.
            self.whiten.weight = nn.Parameter(w)
            self.whiten.bias = nn.Parameter(b)

        # Preprocessing parameters.
        self.preprocess = {
            'mean': mat['net'].meta.normalization.averageImage[0, 0],
            'std': np.array([1, 1, 1], dtype=np.float32)
        }

    def forward(self, image):
        assert image.shape[1] == 3
        assert image.min() >= -EPS and image.max() <= 1 + EPS
        image = torch.clamp(image * 255, 0.0, 255.0)  # Input should be 0-255.
        mean = self.preprocess['mean']
        std = self.preprocess['std']
        image = image - image.new_tensor(mean).view(1, -1, 1, 1)
        image = image / image.new_tensor(std).view(1, -1, 1, 1)

        # Feature extraction.
        descriptors = self.backbone(image)
        b, c, _, _ = descriptors.size()
        descriptors = descriptors.view(b, c, -1)

        # NetVLAD layer.
        descriptors = F.normalize(descriptors, dim=1)  # Pre-normalization.
        desc = self.pool(descriptors)

        # Whiten if needed.
        if hasattr(self, 'whiten'):
            desc = self.whiten(desc)
            desc = F.normalize(desc, dim=1)  # Final L2 normalization.

        return desc
class GaussVLADplushloc(BaseModel):
    required_inputs = ["image"]
    def _init(self,conf):
        self.model =  AttnVLAD(whiten=False)
        device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
    # default_conf = {"model_name": "VGG16-NetVLAD-Pitts30K", "whiten": True}
        weight = "/home/hxy/doctor/feature dectect/hloc/Hierarchical-Localization/hloc/extractors/weights/gaussvladplus.pth.tar"
        print('Load model from', weight)
        state_dict = torch.load(weight, map_location=device,weights_only=False)['state_dict']
        print(state_dict['pool.cluster_weights'])
        self.model.load_state_dict(state_dict)

    
    def _forward(self, data):
        image = data["image"]
        desc = self.model(image)
        return {"global_descriptor": desc}

