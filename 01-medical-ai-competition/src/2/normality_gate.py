import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from duplicate_detector import DuplicateDetector
from ela_detector import ELADetector
from splicing_detector import SplicingDetector
class AdvancedSplicingDetector:
    def __init__(self, cnn_model_path='./2/splicing_cnn.pth', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.ela = ELADetector()
        self.cnn = SplicingDetector()
        self.cnn.load_state_dict(torch.load(cnn_model_path, map_location=self.device))
        self.cnn.to(self.device).eval()
        self.transform = A.Compose([
            A.Resize(height=512, width=512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    def predict(self, img_bgr):
        ela_susp = self.ela.is_spliced(img_bgr)[0]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=img_rgb)['image'].unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.cnn(tensor)
            cnn_prob = torch.softmax(logits, dim=1)[0, 1].item()
        is_spliced = (ela_susp and cnn_prob > 0.2) or (cnn_prob > 0.8)
        return {
            'is_spliced': bool(is_spliced),
            'ela_suspicious': ela_susp,
            'cnn_splice_prob': round(cnn_prob, 4)
        }
class NormalityGate:
    def __init__(self,
                 auth_model_path='./1/best_real_fake_detector.pth',
                 cnn_model_path='./2/splicing_cnn.pth',
                 hash_db_path='hash_db.pkl',
                 device='cuda',
                 feature_extractor=None):
        self.dup_detector = DuplicateDetector(
            model_path=auth_model_path,
            hash_db_path=hash_db_path,
            device=device,
            feature_extractor=feature_extractor
        )
        self.splice_detector = AdvancedSplicingDetector(
            cnn_model_path=cnn_model_path,
            device=device
        )
    def check(self, img_bgr, patient_id, exam_timestamp=None):
        splice_result = self.splice_detector.predict(img_bgr)
        dup_result = self.dup_detector.check_duplicate(img_bgr, patient_id, exam_timestamp)
        details = dict(splice_result)
        details.setdefault("cnn_splice_prob", splice_result.get("cnn_splice_prob", 0.0))
        details["duplicate"] = dup_result
        if splice_result['is_spliced']:
            return {'pass': False, 'reason': '拼接影像', 'details': details}
        if dup_result['is_duplicate']:
            return {'pass': False, 'reason': '重复影像', 'details': details}
        return {'pass': True, 'reason': '通过', 'details': details}
