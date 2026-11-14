import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np
from splicing_detector import SplicingDetector
import cv2
import os
import sys
from tqdm import tqdm
class SplicingDataset(Dataset):
    def __init__(self, root, split='train'):
        self.samples = []
        for cls, folder in enumerate(['0_real', '1_spliced']):
            for f in os.listdir(os.path.join(root, split, folder)):
                self.samples.append((os.path.join(root, split, folder, f), cls))
        self.transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((512, 512, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(img)
        return img, label
def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from train_logger import TrainLogger
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 1e-4
    train_ds = SplicingDataset('./data/splicing', 'train')
    val_ds = SplicingDataset('./data/splicing', 'val')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = SplicingDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    if os.path.isdir("/2026aicompetition/workspace"):
        log_dir = "/2026aicompetition/workspace/logs"
    else:
        log_dir = os.path.join(os.getcwd(), "logs")
    logger = TrainLogger(os.path.join(log_dir, "train_splicing_cnn.jsonl"))
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        logger.log(
            epoch=epoch + 1,
            step=(epoch + 1) * len(train_loader),
            phase="train",
            mode="training",
            loss=round(train_loss / len(train_loader), 4),
            lr=LR,
            data_source="official/train_v1",
            checkpoint=None,
        )
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f}, Val Acc {val_acc:.4f}")
        saved_path = None
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'splicing_cnn.pth')
            saved_path = 'splicing_cnn.pth'
            print("  -> 保存最佳模型")
        logger.log(
            epoch=epoch + 1,
            step=(epoch + 1) * len(train_loader),
            phase="val",
            mode="training",
            loss=None,
            lr=LR,
            data_source="official/val_v1",
            checkpoint=saved_path,
        )
    logger.close()
    print(f"训练完成，最佳验证准确率: {best_acc:.4f}")
if __name__ == '__main__':
    main()
