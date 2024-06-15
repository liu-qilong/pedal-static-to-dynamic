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
    path = '/Users/knpob/Territory/Kolmo/code/fe-footprint-to-pressure/experiment/20210615/'
    opt = config.load_config(path)
    print('-'*50)
    print(opt)
    print('-'*50)

    # load data
    full_dataset = DATASET_REGISTRY[opt.data.dataset](static_path='data/processed/pedar_static.pkl', dynamic_path='data/processed/pedar_dynamic.pkl')
    train_dataset, test_dataset = random_split(full_dataset, [opt.data.train_ratio, 1 - opt.data.train_ratio])

    train_dataloader = DATALOADER_REGISTRY[opt.data.dataloader](train_dataset, batch_size=opt.data.batch_size, shuffle=True)
    test_dataloader = DATALOADER_REGISTRY[opt.data.dataloader](test_dataset, batch_size=opt.data.batch_size, shuffle=True)

    # launch training
    model = MODEL_REGISTRY[opt.model.name](hidden_size=256).to(device)
    loss = LOSS_REGISTRY[opt.train.loss]()
    optimizer = OPTIMIZER_REGISTRY[opt.train.optimizer](params=model.parameters(), lr=opt.train.lr)

    results = train.train(model, train_dataloader, test_dataloader, optimizer, loss, opt.train.epoch, device)