from pathlib import Path
from yacs.config import CfgNode

opt = CfgNode()

# dataset settings
opt.data = CfgNode()
opt.data.dataset = 'Dataset'
opt.data.dataloader = 'DataLoader'
opt.data.train_ratio = 0.8
opt.data.batch_size = 32

# model settings
opt.model = CfgNode()
opt.model.name = ''

# training settings
opt.train = CfgNode()
opt.train.loss = ''
opt.train.metric = ''
opt.train.optimizer = ''
opt.train.lr = 0.001
opt.train.epoch = 10


def get_cloned_default_config():
    return opt.clone()


def load_config(path):
    opt_copy =  get_cloned_default_config()
    opt_copy.merge_from_file(Path(path) / 'config.yaml')
    
    return opt_copy