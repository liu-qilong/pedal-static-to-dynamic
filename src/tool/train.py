from typing import Dict, List, Tuple
from pathlib import Path

import torch
import pandas as pd
from tqdm.auto import tqdm
from yacs.config import CfgNode

def train_step(
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
        loss_fn: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        device: torch.device) -> Tuple[float, float]:
    # put model in train mode
    model.train()
    train_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        # send data to device
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # loss backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # return metric
    train_loss = train_loss / len(dataloader)
    return train_loss


def test_step(
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
        loss_fn: torch.nn.Module,
        device: torch.device) -> Tuple[float, float]:
    # put model in eval mode
    model.eval() 
    test_loss = 0

    # turn on inference context manager
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # send data to target device
            X, y = X.to(device), y.to(device)

            # forward pass
            test_pred_logits = model(X)

            # loss calculation
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

    # return metric
    test_loss = test_loss / len(dataloader)
    return test_loss


def log_step(
        opt: CfgNode,
        logs: Dict[str, List],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
    # print and save logs
    epoch = logs['epoch'][-1]
    epochs = opt.optimizer.epoch
    print(
        f"epoch: {epoch} | "
        f"train_loss: {logs['train_loss'][-1]:.4f} | "
        f"test_loss: {logs['test_loss'][-1]:.4f}"
    )

    # save logs, model, and optimizer if test loss is improved
    if epoch == 0:
        is_improved = True
    else:
        is_improved = logs['test_loss'][-1] < min(logs['test_loss'][:-1])

    if is_improved and (epoch % opt.save_interval == 0 or epoch == epochs - 1):
        # torch.save(logs, Path(path) / 'logs.pth')
        pd.DataFrame(logs).to_csv(Path(opt.path) / 'logs.csv', index=False)
        print(f'logs saved to {Path(opt.path) / "logs.csv"}')

        torch.save(model.state_dict(), Path(opt.path) / 'model.pth')
        print(f'model saved to {Path(opt.path) / "model.pth"}')

        torch.save(optimizer.state_dict(), Path(opt.path) / 'optimizer.pth')
        print(f'optimizer saved {Path(opt.path) / "optimizer.pth"}')


def train_loop(
        opt: CfgNode,
        model: torch.nn.Module, 
        train_dataloader: torch.utils.data.DataLoader, 
        test_dataloader: torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        device: torch.device,
        ) -> Dict[str, List]:
    # create logs dict
    logs = {
        'epoch': [],
        'train_loss': [],
        'test_loss': [],
    }

    for epoch in tqdm(range(opt.optimizer.epoch)):
        train_loss = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            )
        
        test_loss = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            )
        
        # save logs and states
        logs['epoch'].append(epoch)
        logs['train_loss'].append(train_loss)
        logs['test_loss'].append(test_loss)
        log_step(opt, logs, model, optimizer)

    return logs