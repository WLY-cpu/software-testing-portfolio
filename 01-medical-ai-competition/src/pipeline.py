import sys
import os
import time
import cv2
import torch
import numpy as np
from torchvision import transforms
from path_utils import resource_path, setup_module_paths
setup_module_paths()
from inference_module import AuthenticityGate
from normality_gate import NormalityGate
from lesion_detector import LesionDetector
from staging_predictor import StagingPredictor
from keypoint_detector import KeypointDetector, draw_keypoints_and_measurements
from explainer import GradCAM, generate_explanation
from audit_risk import generate_audit_flags  
from competition_output import generate_all_outputs
class CompetitionPipeline:
    def __init__(self, device='cpu'):
        self.device = device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu'
        self.auth_gate = AuthenticityGate(
            model_path=resource_path('1/best_real_fake_detector.pth'),
            device=self.device,
            threshold=0.48
        )
        self.norm_gate = NormalityGate(
            auth_model_path=resource_path('1/best_real_fake_detector.pth'),
            cnn_model_path=resource_path('2/splicing_cnn.pth'),
            hash_db_path=os.path.join(os.getcwd(), 'hash_db.pkl'),
            device=self.device,
            feature_extractor=self.auth_gate.model.spatial_branch
        )
        self.lesion_detector = LesionDetector(
            model_path=resource_path('3/runs/detect/cxr_lesion/weights/best.pt'),
            device=self.device
        )
        self.staging_predictor = StagingPredictor(
            model_path=resource_path('4/staging_classifier.pth'),
            device=self.device
        )
        self.keypoint_detector = KeypointDetector()
        self.staging_model_for_cam = self.staging_predictor.model
        target_layer = self.staging_model_for_cam.features[7][-1]
        self.gradcam = GradCAM(self.staging_model_for_cam, target_layer)
        self.cam_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    def process(self, img_path, patient_id, exam_timestamp=None):
        img = cv2.imread(img_path)
        if img is None:
            return {'error': f'无法读取图片：{img_path}'}
        auth_result = self.auth_gate.predict(img)
        norm_result = self.norm_gate.check(img, patient_id, exam_timestamp)
        detections = self.lesion_detector.predict(img)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                det['estimated_stage'] = 'N/A'
                det['staging_features'] = {}
                det['stage_probs'] = []
                continue
            features, stage = self.staging_predictor.predict(crop)
            det['staging_features'] = {k: v for k, v in features.items() if k != 'probs'}
            det['stage_probs'] = features['probs']
            det['estimated_stage'] = stage
        tl = {"肺部肿块", "肺结节"}
        tumor_dets = [d for d in detections if d['class_name'] in tl]
        other = [d for d in detections if d['class_name'] not in tl]
        report = "胸部X光影像智能辅助诊断意见：\n"
        if not detections:
            report += "双肺未见明确实变、肿块、结节、气胸、胸腔积液等异常征象。心影大小及纵隔未见异常。\n"
        else:
            for i, d in enumerate(detections):
                laterality = d.get('laterality', '')
                lat_str = {"left": "左肺", "right": "右肺", "bilateral": "双侧"}.get(laterality, "")
                diam = d.get('max_diameter_mm', '')
                report += f"病灶{i+1}: {d['class_name']}（{lat_str}）"
                if diam:
                    report += f"，最大径{diam}mm"
                report += f"，置信度{d.get('confidence', 0):.0%}"
                if d.get('estimated_stage'):
                    report += f"，分期推测: {d['estimated_stage']}"
                report += "\n"
        keypoints = self.keypoint_detector.predict(img)
        keypoint_img = draw_keypoints_and_measurements(img, keypoints)
        heatmap_img = None
        explanation = ''
        if detections:
            first_det = detections[0]
            x1, y1, x2, y2 = first_det['bbox']
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = self.cam_transform(crop_rgb).unsqueeze(0).to(self.device)
                cam = self.gradcam.generate(tensor, class_idx=0)
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = cv2.resize(heatmap, (x2 - x1, y2 - y1))
                overlay = img.copy()
                overlay[y1:y2, x1:x2] = cv2.addWeighted(
                    overlay[y1:y2, x1:x2], 0.5, heatmap, 0.5, 0)
                heatmap_img = overlay
                features = first_det.get('staging_features', {})
                explanation = generate_explanation(first_det, features,
                                                   first_det.get('estimated_stage', 'N/A'))
        if not auth_result.get('is_authentic'):
            stage = 'authenticity_failed'
        elif not norm_result.get('pass'):
            stage = 'normality_failed'
        else:
            stage = 'complete'
        result = {
            'stage': stage,
            'authenticity': auth_result,
            'normality': norm_result,
            'detections': detections,
            'report': report,
            'keypoints_img': keypoint_img,
            'heatmap_img': heatmap_img,
            'explanation': explanation
        }
        result['audit_flags'] = generate_audit_flags(result)
        return result
    def process_with_competition_output(
        self,
        img_path: str,
        patient_id: str,
        team_id: str = "2054451790241292288",
        evaluation_id: str = "default",
        answer_base: str = "/2026aicompetition/workspace/answer",
        exam_timestamp: str = None,
        dup_pairs: list = None,
    ) -> dict:
        proc_start = time.time()
        result = self.process(img_path, patient_id, exam_timestamp)
        if 'error' in result:
            return {'result': result, 'outputs': {}}
        auth = result.get("authenticity", {})
        norm = result.get("normality", {})
        detections = result.get("detections", [])
        splice_prob = 0.0
        if norm and "details" in norm:
            splice_prob = norm["details"].get("cnn_splice_prob", 0.0)
        img = cv2.imread(img_path)
        h, w = img.shape[:2] if img is not None else (0, 0)
        processing_ms = max(1, int((time.time() - proc_start) * 1000))
        out_json = generate_all_outputs(
            team_id=team_id,
            evaluation_id=evaluation_id,
            answer_base=answer_base,
            accession_number=patient_id,
            image_size={"width": w, "height": h},
            pixel_spacing=[0.194, 0.194],
            auth_result=auth,
            norm_result={"splice_prob": splice_prob, "normality": norm},
            detections=detections,
            processing_time_ms=processing_ms,
        )
        return {"result": result, "outputs": {"json": out_json}}
if __name__ == '__main__':
    pipeline = CompetitionPipeline()
    test_img = resource_path('1/data/authenticity/val/0_real/P002_P002_frontal.png')
    result = pipeline.process(test_img, patient_id='P_test', exam_timestamp='2026-06-14')
    print("阶段:", result.get('stage'))
    print("病灶数:", len(result.get('detections', [])))
    print("报告:\n", result.get('report'))
    print("审核风险:", result.get('audit_flags'))
    if result.get('keypoints_img') is not None:
        cv2.imwrite('keypoints_output.jpg', result['keypoints_img'])
    if result.get('heatmap_img') is not None:
        cv2.imwrite('heatmap_output.jpg', result['heatmap_img'])
