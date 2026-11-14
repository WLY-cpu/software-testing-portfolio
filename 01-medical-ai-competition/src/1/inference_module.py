import cv2
import torch
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A
from train_authenticity import RealFakeDetector
class AuthenticityGate:
    def __init__(self, model_path='best_real_fake_detector.pth', threshold=0.85, device='cuda'):
        self.device = device
        self.threshold = threshold
        self.model = RealFakeDetector(num_classes=3)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.transform = A.Compose([
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2()
        ])
    def predict(self, image_path_or_array):
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_path_or_array
        img = cv2.resize(img, (512, 512))
        transformed = self.transform(image=img)['image'].unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(transformed)
            probs = torch.softmax(outputs, dim=1)
            real_prob = probs[0, 0].item()
            is_real = real_prob > self.threshold
            return {
                'is_authentic': bool(is_real),
                'real_prob': round(real_prob, 4),
                'fake_prob': round(probs[0, 1].item(), 4),
                'nonhuman_prob': round(probs[0, 2].item(), 4)
            }
if __name__ == '__main__':
    gate = AuthenticityGate(threshold=0.45)
    result = gate.predict('test_xray.png')
    print(result)
