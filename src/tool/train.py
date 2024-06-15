from typing import Dict, List, Tuple

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


def train(
        model: torch.nn.Module, 
        train_dataloader: torch.utils.data.DataLoader, 
        test_dataloader: torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epochs: int,
        device: torch.device) -> Dict[str, List]:
    # create results dict
    results = {
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

        # logs
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {test_loss:.4f} | "
        )

        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

    return results