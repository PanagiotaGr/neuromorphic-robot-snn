import torch
import torch.nn as nn

from models import rate_encode


def train_ann(model, train_loader, test_loader, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "test_loss": [], "test_acc": []}

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        total = correct = 0
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            preds = logits.argmax(dim=1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
            total_loss += loss.item() * yb.size(0)
        return total_loss / total, correct / total

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * yb.size(0)
            seen += yb.size(0)

        train_loss = running_loss / seen
        test_loss, test_acc = evaluate(test_loader)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(f"[ANN] Epoch {epoch+1:02d}/{epochs} | train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | test_acc={test_acc*100:.2f}%")
    return history


def train_snn(model, train_loader, test_loader, epochs, lr, num_steps, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "test_loss": [], "test_acc": []}

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        total = correct = 0
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            spk_in = rate_encode(xb, num_steps).to(device)
            spk_out, _ = model(spk_in)
            logits = spk_out.sum(dim=0)
            loss = loss_fn(logits, yb)
            preds = logits.argmax(dim=1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
            total_loss += loss.item() * yb.size(0)
        return total_loss / total, correct / total

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            spk_in = rate_encode(xb, num_steps).to(device)
            spk_out, _ = model(spk_in)
            logits = spk_out.sum(dim=0)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * yb.size(0)
            seen += yb.size(0)

        train_loss = running_loss / seen
        test_loss, test_acc = evaluate(test_loader)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(f"[SNN] Epoch {epoch+1:02d}/{epochs} | train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | test_acc={test_acc*100:.2f}%")
    return history
