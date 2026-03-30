from .xfeat_mlsd import XFeat_MLSD


def build_model(cfg):
    model_name = cfg.model.model_name
    if model_name != "xfeat_mlsd":
        raise NotImplementedError(f"{model_name} no such model!")
    return XFeat_MLSD(cfg)
