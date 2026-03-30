from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms as T
import numpy as np
import logging
from scipy.io import loadmat
# from torchvision import transforms as T
from ..utils.base_model import BaseModel

EPS = 1e-6

netvlad_path = Path(__file__).parent / '../weights/netvlad/'
logger = logging.getLogger(__name__)



def attention(query, key, value, alpha, eps=1e-6):
    score = torch.einsum('bdk,bdn->bkn', query, key)# query：K列d维的特征；key：N列d维的特征
    prob = F.softmax(alpha * score, dim=1) # K ×N  按照行相加
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
        self.centers.data = torch.from_numpy(clsts).T[None]# D × K
        
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
        centers = nn.parameter.Parameter(torch.empty([1, input_dim, K]))
        nn.init.xavier_uniform_(centers)
        self.register_parameter('centers', centers)
        self.intranorm = intranorm
        self.output_dim = input_dim * K
        self.alpha = nn.Parameter(torch.tensor(300.0))

    def forward(self, x):
        b = x.size(0)
        desc = attention(F.normalize(self.centers), x, x, self.alpha)
        desc = desc - self.centers
        if self.intranorm:
            desc = F.normalize(desc, dim=1)
        desc = desc.reshape(b, -1)
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
        backbone = list(models.efficientnet_b3(pretrained=True).children())[0]  # Remove classification head.
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
    if name == 'vgg' and from_matconvmat:
        backbone.load_state_dict(torch.load('./weights/vd16_offtheshelf_conv5_3_max.pth'))
    backbone.output_dim = output_dim 
    return backbone

def get_pool(name, **kwargs):
    if name == 'attnvlad':
        pool = AttnVLADLayer(**kwargs)
    elif name == 'netvlad':
        pool = NetVLADLayer(**kwargs)
    else:
        raise ValueError(f'{name} not find.')
    return pool

class AttnVLAD(nn.Module):
    default_conf = {
        'backbone_name': 'efficient',
        'pool_name': 'attnvlad',
        'checkpoint_dir': netvlad_path,
        'whiten': False,
    }

    def __init__(self, **conf):
        super().__init__()
        self.conf = conf = {**self.default_conf, **conf}
        
        # build model
        self.backbone = get_backbone(conf['backbone_name'])
        self.pool = get_pool(conf['pool_name'], input_dim=self.backbone.output_dim)
        
        self.model_name = conf['backbone_name'].lower() + '-' +conf['pool_name'].lower()

        # Preprocessing parameters.        
        # self.preprocess = T.Normalize(mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
        #                                 std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
        self.preprocess = T.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])

    def forward(self, image):
        assert image.shape[1] == 3
        assert image.min() >= -EPS and image.max() <= 1 + EPS
        
        image = self.preprocess(image)

        # Feature extraction.
        descriptors = self.backbone(image)
        b, c, _, _ = descriptors.size()
        descriptors = descriptors.view(b, c, -1)
        desc = self.pool(descriptors)
        
        return desc
    
    
class AttenVLADhloc(BaseModel):
    required_inputs = ["image"]
    def _init(self,conf):
        self.model =  AttnVLAD(whiten=False)
    # default_conf = {"model_name": "VGG16-NetVLAD-Pitts30K", "whiten": True}
        weight = "/home/hxy/doctor/feature dectect/hloc/Hierarchical-Localization/hloc/extractors/weights/efficient-attnvlad.pth.tar"
        print('Load model from', weight)
        state_dict = torch.load(weight, weights_only=False)['state_dict']
        print(state_dict['pool.cluster_weights'])
        self.model.load_state_dict(state_dict)

    
    def _forward(self, data):
        image = data["image"]
        desc = self.model(image)
        return {"global_descriptor": desc}