"""Mini-batch training with Adam, MSE loss and early stopping on validation loss."""
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import YieldNet


def train_model(X_train, y_train, X_val, y_val, model_cfg, train_cfg, seed):
    torch.manual_seed(seed)
    model = YieldNet(X_train.shape[1], model_cfg["hidden_layers"])
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"],
                                 weight_decay=train_cfg["weight_decay"])
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                        batch_size=train_cfg["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    X_val_t, y_val_t = torch.tensor(X_val), torch.tensor(y_val)

    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, best_epoch, waited = float("inf"), None, 0, 0
    for epoch in range(train_cfg["max_epochs"]):
        model.train()
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        history["train_loss"].append(running / len(loader.dataset))

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val, best_state, best_epoch, waited = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            waited += 1
            if waited >= train_cfg["patience"]:
                break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    return model, history
