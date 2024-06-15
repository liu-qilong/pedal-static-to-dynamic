import argparse

import torch
from torch.utils.data import random_split

from src.tool import config, train
from src import dataset, dataloader, model, loss, metric, optimizer
from src.tool.registry import DATASET_REGISTRY, DATALOADER_REGISTRY, MODEL_REGISTRY, LOSS_REGISTRY, METRIC_REGISTRY, OPTIMIZER_REGISTRY


if __name__ == '__main__':
    # torch setup
    torch.manual_seed(0)

    if torch.cuda.is_available():
        device = 'cuda'
    if torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # load options
    parser = argparse.ArgumentParser('training script')
    parser.add_argument('--path', '-p', help="The path to the experiment folder where the configuration sheet, network weights, and other results are stored.", type=str, required=True)
    args = parser.parse_args()
    print(f'load configrations from: {args.path}')
    opt = config.load_config(args.path)
    print('-'*50)
    print(opt)
    print('-'*50)

    # load data
    full_dataset = DATASET_REGISTRY[opt.dataset.name](**opt.dataset.args)
    train_dataset, test_dataset = random_split(full_dataset, [opt.dataset.train_ratio, 1 - opt.dataset.train_ratio])

    train_dataloader = DATALOADER_REGISTRY[opt.dataloader.name](train_dataset, **opt.dataloader.args)
    test_dataloader = DATALOADER_REGISTRY[opt.dataloader.name](test_dataset, **opt.dataloader.args)

    # launch training
    model = MODEL_REGISTRY[opt.model.name](**opt.model.args).to(device)
    loss = LOSS_REGISTRY[opt.loss.name](**opt.loss.args)
    optimizer = OPTIMIZER_REGISTRY[opt.optimizer.name](**opt.optimizer.args, params=model.parameters())

    results = train.train(model, train_dataloader, test_dataloader, optimizer, loss, opt.optimizer.epoch, device)