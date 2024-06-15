from typing import Dict, List, Tuple
from pathlib import Path

import torch
from tqdm.auto import tqdm

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
        epoch: int,
        epochs: int,
        save_interval: int,
        path: str,
        logs: Dict[str, List],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
    # print and save logs
    print(
        f"epoch: {epoch + 1} | "
        f"train_loss: {logs['train_loss'][-1]:.4f} | "
        f"test_loss: {logs['test_loss'][-1]:.4f}"
    )

    # save logs, model, and optimizer if test loss is improved
    if epoch == 0:
        is_improved = True
    else:
        is_improved = logs['test_loss'][-1] < min(logs['test_loss'][:-1])

    if is_improved and (epoch % save_interval == 0 or epoch == epochs - 1):
        torch.save(logs, Path(path) / 'logs.pth')
        print(f'logs saved to {Path(path) / "logs.pth"}')

        torch.save(model.state_dict(), Path(path) / 'model.pth')
        print(f'model saved to {Path(path) / "model.pth"}')

        torch.save(optimizer.state_dict(), Path(path) / 'optimizer.pth')
        print(f'optimizer saved {Path(path) / "optimizer.pth"}')


def train_loop(
        path: str,
        model: torch.nn.Module, 
        train_dataloader: torch.utils.data.DataLoader, 
        test_dataloader: torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epochs: int,
        device: torch.device,
        save_interval: int = 1,
        ) -> Dict[str, List]:
    # create logs dict
    logs = {
        "train_loss": [],
        "test_loss": [],
    }

    for epoch in tqdm(range(epochs)):
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
        logs["train_loss"].append(train_loss)
        logs["test_loss"].append(test_loss)
        log_step(epoch, epochs, save_interval, path, logs, model, optimizer)

    return logs