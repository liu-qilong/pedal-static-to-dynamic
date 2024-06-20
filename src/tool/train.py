import torch
from torch.utils.data import random_split

import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path

from src.tool.registry import DATASET_REGISTRY, DATALOADER_REGISTRY, MODEL_REGISTRY, LOSS_REGISTRY, METRIC_REGISTRY, OPTIMIZER_REGISTRY, SCRIPT_REGISTRY

@SCRIPT_REGISTRY.register()
class BasicTrainScript():
    def __init__(self, opt):
        self.opt = opt
        
        # device select
        if opt.device_select == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            if torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = opt.device_select

        print(f'training on {self.device}...')

        # init logs dict
        self.logs = {
            'epoch': [],
            'train_loss': [],
            'test_loss': [],
            'train_std': [],
        }

        # init metric dict
        self.metric_dict = {}

        for key, value in opt.metric.items():
            self.metric_dict[key] = METRIC_REGISTRY[value.name](**value.args)
            self.logs[key] = []

    def load_data(self):
        self.full_dataset = DATASET_REGISTRY[self.opt.dataset.name](device = self.device, **self.opt.dataset.args)
        self.train_dataset, self.test_dataset = random_split(self.full_dataset, [self.opt.dataset.train_ratio, 1 - self.opt.dataset.train_ratio])

        self.train_dataloader = DATALOADER_REGISTRY[self.opt.dataloader.name](self.train_dataset, **self.opt.dataloader.args)
        self.test_dataloader = DATALOADER_REGISTRY[self.opt.dataloader.name](self.test_dataset, **self.opt.dataloader.args)

    def train_prep(self):
        self.model = MODEL_REGISTRY[self.opt.model.name](device=self.device, **self.opt.model.args)
        self.loss_fn = LOSS_REGISTRY[self.opt.loss.name](**self.opt.loss.args)
        self.optimizer = OPTIMIZER_REGISTRY[self.opt.optimizer.name](**self.opt.optimizer.args, params=self.model.parameters())

        if self.opt.use_pretrain:
            self.model.load_state_dict(torch.load(Path(self.opt.path) / 'model.pth'))
            self.optimizer.load_state_dict(torch.load(Path(self.opt.path) / 'optimizer.pth'))

    def train_loop(self):
        for epoch in (pdar := tqdm(range(self.opt.optimizer.epochs))):
            self.logs['epoch'].append(epoch)
            self._train_step()
            self._test_step()
            self._log_step()
            pdar.set_description(f'epoch {epoch} | train_loss {self.logs["train_loss"][-1]:.4f} | test_loss {self.logs["test_loss"][-1]:.4f} |  train_std {self.logs["train_std"][-1]:.4f}')

    def _train_step(self):
        # put model in train mode
        self.model.train()
        train_loss = 0
        train_std = 0

        for batch, (x, y) in enumerate(self.train_dataloader):
            # forward pass
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item() 
            train_std += y_pred.std().item()

            # loss backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # append loss
        self.logs['train_loss'].append(train_loss / len(self.train_dataloader))
        self.logs['train_std'].append(train_std / len(self.train_dataloader))

    def _test_step(self):
        # put model in eval mode
        self.model.eval()
        test_loss = 0

        for key, metric_fn in self.metric_dict.items():
            self.logs[key].append(0)

        # turn on inference context manager
        with torch.inference_mode():
            for batch, (X, y) in enumerate(self.test_dataloader):
                # forward pass
                y_pred = self.model(X)

                # loss calculation
                loss = self.loss_fn(y_pred, y)
                test_loss += loss.item()

                # metric calculation
                for key, metric_fn in self.metric_dict.items():
                    self.logs[key][-1] += metric_fn(y, y_pred).item()

        # append loss & metrics
        self.logs['test_loss'].append(test_loss / len(self.test_dataloader))

        for key, metric_fn in self.metric_dict.items():
            self.logs[key][-1] = self.logs[key][-1] / len(self.test_dataloader)

    def _log_step(self):
        # print and save logs
        epoch = self.logs['epoch'][-1]

        # save logs
        pd.DataFrame(self.logs).to_csv(Path(self.opt.path) / 'logs.csv', index=False)

        # save model and optimizer if test loss is improved
        if epoch == 0 or self.logs['test_loss'][-1] < min(self.logs['test_loss'][:-1]):
            torch.save(self.model.state_dict(), Path(self.opt.path) / 'model.pth')
            torch.save(self.optimizer.state_dict(), Path(self.opt.path) / 'optimizer.pth')