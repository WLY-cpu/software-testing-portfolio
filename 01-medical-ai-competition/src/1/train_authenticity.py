import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import sys
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
class AuthenticityDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.samples = []
        for class_idx, class_name in enumerate(['0_real', '1_fake', '2_nonhuman']):
            class_dir = os.path.join(root_dir, split, class_name)
            if not os.path.exists(class_dir):
                continue
            for fname in os.listdir(class_dir):
                self.samples.append((os.path.join(class_dir, fname), class_idx))
        self.train_resize_crop = T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(512, scale=(0.6, 1.0)),
            T.ToTensor()  # → [0,1] 张量
        ])
        self.train_extra = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.5),
            A.ImageCompression(quality_range=(30, 80), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        self.val_resize = T.Compose([
            T.ToPILImage(),
            T.Resize((512, 512)),
            T.ToTensor()
        ])
        self.val_norm = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        self.split = split
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((512, 512, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.split == 'train':
            img_tensor = self.train_resize_crop(img)          # 3x512x512, [0,1]
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = self.train_extra(image=img_np)['image']
        else:
            img_tensor = self.val_resize(img)
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = self.val_norm(image=img_np)['image']
        return img, label
class RealFakeDetector(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.spatial_branch = models.resnet50(weights='IMAGENET1K_V2')
        self.spatial_branch.fc = nn.Identity()
        self.freq_branch = models.resnet18(weights=None)
        self.freq_branch.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.freq_branch.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def compute_fft_magnitude(self, x):
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        gray = gray.unsqueeze(1)
        fft = torch.fft.fft2(gray)
        fft_mag = torch.log(1 + torch.abs(fft))
        fft_mag = torch.fft.fftshift(fft_mag, dim=(-2, -1))
        return fft_mag
    def forward(self, x):
        s_feat = self.spatial_branch(x)
        f_map = self.compute_fft_magnitude(x)
        f_feat = self.freq_branch(f_map)
        fused = torch.cat([s_feat, f_feat], dim=1)
        return self.classifier(fused)
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from train_logger import TrainLogger, log_training_step, log_validation_step
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    BATCH_SIZE = 16          # 根据显存调整
    EPOCHS = 30
    LR = 1e-4
    train_dataset = AuthenticityDataset('./data/authenticity', 'train')
    val_dataset = AuthenticityDataset('./data/authenticity', 'val')
    if len(train_dataset) == 0:
        raise RuntimeError("训练集为空，请先运行 build_dataset.py 生成数据！")
    labels = [s[1] for s in train_dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)
    model = RealFakeDetector(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    if os.path.isdir("/2026aicompetition/workspace"):
        log_dir = "/2026aicompetition/workspace/logs"
    else:
        log_dir = os.path.join(os.getcwd(), "logs")
    logger = TrainLogger(os.path.join(log_dir, "train_authenticity.jsonl"))
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        scheduler.step()
        lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else LR
        log_training_step(logger, epoch + 1, (epoch + 1) * len(train_loader),
                          train_loss / len(train_loader), lr)
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)
                all_labels.extend(lbls.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        val_acc = correct / total
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        real_idx = 0
        y_true_binary = (all_labels != real_idx).astype(int)
        y_score_binary = 1 - all_probs[:, real_idx]
        auc = roc_auc_score(y_true_binary, y_score_binary)
        print(f"Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f}, "
              f"Val Loss {val_loss/len(val_loader):.4f}, Val Acc {val_acc:.4f}, AUC {auc:.4f}")
        saved_path = None
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_real_fake_detector.pth')
            saved_path = 'best_real_fake_detector.pth'
            print(f"  -> 保存最佳模型 (val_acc={best_val_acc:.4f})")
        log_validation_step(logger, epoch + 1, (epoch + 1) * len(train_loader),
                            val_loss / len(val_loader), lr, saved_path)
    logger.close()
    print(f"训练完成，最佳验证准确率: {best_val_acc:.4f}")
