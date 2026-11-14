import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import cv2
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from staging_model import LesionStagingClassifier
class StagingDataset(Dataset):
    def __init__(self, csv_path, img_dir, transforms=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transforms = transforms
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.img_dir}/{row['filename']}"
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            crop = img
        crop = cv2.resize(crop, (224, 224))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        if self.transforms:
            crop = self.transforms(crop)
        labels = torch.tensor([
            row['size_gt3cm'], row['chest_wall_invasion'], row['obstructive_pneumonia'],
            row['lymph_node_enlargement'], row['contralateral_lesion'], row['bone_destruction'],
            row['has_tumor']
        ], dtype=torch.float)
        return crop, labels
def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from train_logger import TrainLogger
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 32
    EPOCHS = 30
    LR = 1e-4
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    train_ds = StagingDataset('staging_labels.csv', '../3/data/images/train', train_transforms)
    val_ds = StagingDataset('staging_labels.csv', '../3/data/images/train', val_transforms)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = LesionStagingClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    if os.path.isdir("/2026aicompetition/workspace"):
        log_dir = "/2026aicompetition/workspace/logs"
    else:
        log_dir = os.path.join(os.getcwd(), "logs")
    logger = TrainLogger(os.path.join(log_dir, "train_staging.jsonl"))
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else LR
        logger.log(
            epoch=epoch + 1,
            step=(epoch + 1) * len(train_loader),
            phase="train",
            mode="training",
            loss=round(train_loss / len(train_loader), 4),
            lr=lr,
            data_source="official/train_v1",
            checkpoint=None,
        )
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                logits = model(imgs)
                loss = criterion(logits, lbls)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f}, Val Loss {avg_val_loss:.4f}")
        saved_path = None
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'staging_classifier.pth')
            saved_path = 'staging_classifier.pth'
            print("  -> 保存最佳模型")
        logger.log(
            epoch=epoch + 1,
            step=(epoch + 1) * len(train_loader),
            phase="val",
            mode="training",
            loss=round(avg_val_loss, 4),
            lr=lr,
            data_source="official/val_v1",
            checkpoint=saved_path,
        )
    logger.close()
if __name__ == '__main__':
    main()
