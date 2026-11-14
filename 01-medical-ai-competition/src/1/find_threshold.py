import torch
import numpy as np
from train_authenticity import RealFakeDetector, AuthenticityDataset
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealFakeDetector(num_classes=3)
model.load_state_dict(torch.load('best_real_fake_detector.pth', map_location=device))
model.to(device)
model.eval()
val_dataset = AuthenticityDataset('./data/authenticity', 'val')
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
all_labels = []
all_probs = []
with torch.no_grad():
    for imgs, lbls in val_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        all_labels.extend(lbls.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)
real_mask = (all_labels == 0)
real_count = np.sum(real_mask)
if real_count == 0:
    raise RuntimeError("验证集中没有真实样本，请检查数据！")
real_probs = all_probs[real_mask, 0]
best_thresh = 0.0
for thresh in np.arange(1.0, 0.0, -0.01):
    passed = np.sum(real_probs > thresh)
    rate = passed / real_count
    if rate >= 0.995:
        best_thresh = thresh
        break
print(f"真实样本数: {real_count}")
print(f"真实样本最低概率: {real_probs.min():.4f}")
print(f"推荐阈值: 真实概率 > {best_thresh:.2f} 时判为真实")
