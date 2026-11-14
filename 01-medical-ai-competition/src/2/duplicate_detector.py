import sys
import os
import cv2
import numpy as np
import torch
from scipy.fft import dct               # 替代 scipy.fftpack
from collections import defaultdict
import pickle
import albumentations as A
from albumentations.pytorch import ToTensorV2
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '1'))
from train_authenticity import RealFakeDetector
class DuplicateDetector:
    def __init__(self, model_path='./1/best_real_fake_detector.pth',
                 hash_db_path='hash_db.pkl', device='cuda',
                 feature_extractor=None):
        self.device = device if torch.cuda.is_available() else 'cpu'
        if getattr(sys, 'frozen', False) and hash_db_path == 'hash_db.pkl':
            hash_db_path = os.path.join(os.getcwd(), 'hash_db.pkl')
        self.hash_db_path = hash_db_path
        self.hash_db = self.load_db(hash_db_path)
        self.phash_threshold = 5
        self.cosine_threshold = 0.98
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            base_model = RealFakeDetector(num_classes=3)
            base_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.feature_extractor = base_model.spatial_branch
        self.feature_extractor.eval().to(self.device)
        self.transform = A.Compose([
            A.Resize(height=512, width=512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    def load_db(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            db = defaultdict(list)
            db.update(data)
            return db
        return defaultdict(list)
    def save_db(self):
        with open(self.hash_db_path, 'wb') as f:
            pickle.dump(dict(self.hash_db), f)
    def reset_db(self):
        """清空本机查重库，保证每次测评只在当前批次内找重复对。"""
        self.hash_db.clear()
        self.save_db()
    def compute_phash(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct_coeff = dct(dct(gray.T, norm='ortho').T, norm='ortho')
        dct_low = dct_coeff[:8, :8]
        avg = dct_low.mean()
        hash_str = ''.join(['1' if v > avg else '0' for v in dct_low.flatten()])
        return hash_str
    def hamming_distance(self, hash1, hash2):
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    def compute_deep_feature(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (512, 512))
        tensor = self.transform(image=img_resized)['image'].unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.feature_extractor(tensor)
        return feat.cpu().numpy().flatten()
    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    @staticmethod
    def _pair_probability(h_dist, cos_sim):
        """由感知哈希汉明距离与深度特征余弦相似度给出重复对概率估计。"""
        phash_sim = max(0.0, 1.0 - h_dist / 64.0)
        return round(0.5 * (cos_sim + phash_sim), 4)
    def check_duplicate(self, img_bgr, patient_id, exam_timestamp=None):
        phash_current = self.compute_phash(img_bgr)
        feat_current = self.compute_deep_feature(img_bgr)
        for pid, records in self.hash_db.items():
            for ts, phash_hist, feat_hist in records:
                h_dist = self.hamming_distance(phash_current, phash_hist)
                cos_sim = self.cosine_similarity(feat_current, feat_hist)
                if h_dist == 0:
                    return {
                        'is_duplicate': True,
                        'reason': f'图像与检查 {pid} 的历史检查({ts})完全相同',
                        'matched_uid': pid,
                        'pair_prob': 1.0,
                    }
                if h_dist <= self.phash_threshold and cos_sim > self.cosine_threshold:
                    return {
                        'is_duplicate': True,
                        'reason': f'图像与检查 {pid} 的历史检查({ts})高度相似',
                        'matched_uid': pid,
                        'pair_prob': self._pair_probability(h_dist, cos_sim),
                    }
        if patient_id in self.hash_db:
            for ts, phash_hist, feat_hist in self.hash_db[patient_id]:
                h_dist = self.hamming_distance(phash_current, phash_hist)
                cos_sim = self.cosine_similarity(feat_current, feat_hist)
                if h_dist <= self.phash_threshold and cos_sim > self.cosine_threshold:
                    return {
                        'is_duplicate': True,
                        'reason': f'与检查 {patient_id} 的历史检查({ts})高度相似',
                        'matched_uid': patient_id,
                        'pair_prob': self._pair_probability(h_dist, cos_sim),
                    }
        if patient_id not in self.hash_db:
            self.hash_db[patient_id] = []
        self.hash_db[patient_id].append(
            (exam_timestamp or 'unknown', phash_current, feat_current)
        )
        self.save_db()
        return {'is_duplicate': False, 'reason': ''}
