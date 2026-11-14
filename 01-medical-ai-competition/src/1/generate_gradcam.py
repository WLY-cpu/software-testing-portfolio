import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from train_authenticity import RealFakeDetector
from gradcam_utils import GradCAM
import matplotlib.pyplot as plt
MODEL_PATH = 'best_real_fake_detector.pth'
IMG_PATH = 'test_xray.png'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = RealFakeDetector(num_classes=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
target_layer = model.spatial_branch.layer4[-1]
gradcam = GradCAM(model, target_layer)
transform = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
img_bgr = cv2.imread(IMG_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"找不到图片：{IMG_PATH}")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
transformed = transform(image=img_rgb)
input_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
print(f"预测概率：真实={probs[0]:.4f}, 假体={probs[1]:.4f}, 非人体={probs[2]:.4f}")
class_names = ['Real', 'Fake', 'Non-human']
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
orig = cv2.cvtColor(cv2.imread(IMG_PATH), cv2.COLOR_BGR2RGB)
axes[0].imshow(orig)
axes[0].set_title('Original')
axes[0].axis('off')
for class_idx in range(3):
    cam = gradcam.generate(input_tensor, class_idx=class_idx)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed = cv2.addWeighted(orig, 0.5, cv2.resize(heatmap, (orig.shape[1], orig.shape[0])), 0.5, 0)
    axes[class_idx+1].imshow(overlayed)
    axes[class_idx+1].set_title(f'{class_names[class_idx]} (prob={probs[class_idx]:.3f})')
    axes[class_idx+1].axis('off')
plt.tight_layout()
plt.savefig('gradcam_results.png', dpi=150)
plt.show()
gradcam.remove_hooks()
print("Grad-CAM 结果已保存为 gradcam_results.png")
