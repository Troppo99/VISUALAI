import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import random


# 1. Dataset Dummy
class SimpleDetectionDataset(Dataset):
    def __init__(self, num_samples=1000, img_size=64):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        x1, y1 = random.randint(5, 30), random.randint(5, 30)
        x2, y2 = x1 + random.randint(10, 20), y1 + random.randint(10, 20)
        img[y1:y2, x1:x2, :] = 255
        bbox = torch.tensor([x1 / self.img_size, y1 / self.img_size, x2 / self.img_size, y2 / self.img_size], dtype=torch.float32)
        label = torch.tensor([1], dtype=torch.float32)  # 1 berarti ada objek
        return self.transform(img), bbox, label


# 2. Model YOLO Sederhana
class SimpleYOLO(nn.Module):
    def __init__(self):
        super(SimpleYOLO, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(16, 32, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 5))  # 4 untuk bbox (x1, y1, x2, y2) dan 1 untuk confidence

    def forward(self, preds, targets_bbox, targets_conf):
        pred_bbox = preds[:, :4]
        pred_conf = preds[:, 4]
        bbox_loss = self.mse(pred_bbox, targets_bbox)
        conf_loss = self.bce(pred_conf, targets_conf.squeeze())
        return bbox_loss + conf_loss


# 3. Loss Function
class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets_bbox, targets_conf):
        pred_bbox = preds[:, :4]
        pred_conf = preds[:, 4]
        bbox_loss = self.mse(pred_bbox, targets_bbox)
        conf_loss = self.bce(pred_conf, targets_conf)
        return bbox_loss + conf_loss


# 4. Training Loop
def train():
    dataset = SimpleDetectionDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = SimpleYOLO()
    criterion = YoloLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        total_loss = 0
        for imgs, bboxes, labels in dataloader:
            preds = model(imgs)
            loss = criterion(preds, bboxes, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")


if __name__ == "__main__":
    train()
