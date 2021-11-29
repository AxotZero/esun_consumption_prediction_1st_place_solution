import yaml
from easydict import EasyDict


def get_cat_emb_dim(num_classes):
    return int(min(600,round(1.6*num_classes**0.56)))


def read_yml(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))


def parse_cols_config(path):
    config = read_yml(path)
    input_dim = len(config)
    cat_dims = []
    cat_idxs = []
    cat_emb_dim = []
    num_idxs = [] 

    for i, (k, v) in enumerate(config.items()):
        if v.type == 'numerical': 
            num_idxs.append(i)
        else:
            cat_idxs.append(i)
            cat_dims.append(len(v.ori_index))
            cat_emb_dim.append(get_cat_emb_dim(len(v.ori_index)))
    return input_dim, num_idxs, cat_dims, cat_idxs, cat_emb_dim



