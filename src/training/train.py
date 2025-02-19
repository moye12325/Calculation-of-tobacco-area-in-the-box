'''
File: train.py
Author: 19423
Description: ${1:简要描述可以写在这里}
Created: 11:03
Version: ${2:版本号 (如 v1.0)}

修改记录:
Date        Author        Modification Content
2023/12/14   19423       Create the file
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.model_loader import load_model
from data_loader.data_loader import get_dataloaders
from config.parameters import LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, DEVICE
from config.paths import MODEL_SAVE_DIR  # 确保 `MODEL_SAVE_DIR` 指向 `src/model_version_dir/`

# 创建目录
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

train_loader, val_loader = get_dataloaders()
model = load_model()

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

best_val_loss = float("inf")  # 记录最小的验证损失
best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model.pth")

def train():
    global best_val_loss
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 计算验证集损失
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # **保存最新模型**
        latest_model_path = os.path.join(MODEL_SAVE_DIR, "latest_model.pth")
        torch.save(model.state_dict(), latest_model_path)

        # **如果验证损失降低，保存最佳模型**
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ New best model saved at {best_model_path}")

if __name__ == "__main__":
    train()
    print(f"✅ Training complete! Best model saved at {best_model_path}")