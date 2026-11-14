import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from train_authenticity import RealFakeDetector, AuthenticityDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealFakeDetector(num_classes=3)
model.load_state_dict(torch.load('best_real_fake_detector.pth', map_location=device))
model.to(device)
model.eval()
val_dataset = AuthenticityDataset('./data/authenticity', 'val')
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
all_labels = []
all_preds = []
all_probs = []
with torch.no_grad():
    for imgs, lbls in val_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(lbls.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
acc = accuracy_score(all_labels, all_preds)
print(f"验证集准确率: {acc:.4f}")
cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2])
print("混淆矩阵 (行=真实, 列=预测):")
print("          真实  假体  非人体")
print(f"真实:    {cm[0]}")
print(f"假体:    {cm[1]}")
print(f"非人体:  {cm[2]}")
real_recall = cm[0,0] / cm[0].sum() if cm[0].sum() > 0 else 0
print(f"真人体通过率 (召回率): {real_recall:.4f}")
y_true_binary = (all_labels != 0).astype(int)
y_score_binary = 1 - all_probs[:, 0]
auc = roc_auc_score(y_true_binary, y_score_binary)
print(f"真实 vs 异常 AUC: {auc:.4f}")
