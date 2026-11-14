import torch
import cv2
import numpy as np
from torchvision import transforms
from staging_model import LesionStagingClassifier
class StagingPredictor:
    def __init__(self, model_path='staging_classifier.pth', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = LesionStagingClassifier(num_features=7)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
    def predict(self, lesion_crop_bgr):
        """输入病灶裁剪图（BGR格式），返回征象字典和分期估计"""
        img_rgb = cv2.cvtColor(lesion_crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        features = {
            'size_gt3cm': probs[0] > 0.5,
            'chest_wall_invasion': probs[1] > 0.5,
            'obstructive_pneumonia': probs[2] > 0.5,
            'lymph_node_enlargement': probs[3] > 0.5,
            'contralateral_lesion': probs[4] > 0.5,
            'bone_destruction': probs[5] > 0.5,
            'probs': probs.tolist()
        }
        stage = self.determine_stage(features)
        return features, stage
    def determine_stage(self, features):
        if features['contralateral_lesion'] or features['bone_destruction']:
            return 'IV'
        if features['chest_wall_invasion'] or features['lymph_node_enlargement']:
            return 'IIB/IIIA'
        if features['size_gt3cm'] or features['obstructive_pneumonia']:
            return 'IB/IIA'
        return 'IA'
