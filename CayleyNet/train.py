"""
CayleyNet 训练脚本
"""
import torch
import torch.nn as nn
from model import CayleyNet
import time


def train_cayleynet():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = CayleyNet(dim=512, depth=6, num_heads=8, num_classes=1000, img_size=224).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.1f}M, Trainable: {trainable / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    ce_loss = nn.CrossEntropyLoss()

    dummy_images = torch.randn(32, 3, 224, 224).to(device)
    dummy_labels = torch.randint(0, 1000, (32,)).to(device)

    model.train()
    print(f"\nTraining CayleyNet on {device}...")
    for epoch in range(5):
        t0 = time.time()
        optimizer.zero_grad()
        logits, conf_loss = model(dummy_images)
        loss = ce_loss(logits, dummy_labels) + conf_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        acc = (logits.argmax(-1) == dummy_labels).float().mean()
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}: Loss={loss.item():.4f} CE={ce_loss(logits, dummy_labels).item():.4f} Conf={conf_loss.item():.4f} Acc={acc:.0%} Time={elapsed:.1f}s")

    model.eval()
    with torch.no_grad():
        logits = model(dummy_images)
        acc = (logits.argmax(-1) == dummy_labels).float().mean()
        print(f"\nEval Acc: {acc:.0%}")

    if device == "cuda":
        vram = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak VRAM: {vram:.1f}GB")

    print("Done!")


if __name__ == "__main__":
    train_cayleynet()
