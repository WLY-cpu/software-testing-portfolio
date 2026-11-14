import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self.hook_handles.append(
            target_layer.register_forward_hook(self._forward_hook)
        )
        self.hook_handles.append(
            target_layer.register_full_backward_hook(self._backward_hook)
        )
    def _forward_hook(self, module, input, output):
        self.activations = output
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
def generate_explanation(detection, staging_features, estimated_stage):
    """生成自然语言解释（赛道八 17 类兼容）"""
    parts = []
    laterality = detection.get('laterality', '')
    lat_str = {"left": "左肺", "right": "右肺", "bilateral": "双侧"}.get(laterality, "")
    parts.append(f"病灶位于{lat_str}（{detection['bbox']}），")
    parts.append(f"AI判定为{detection['class_name']}（置信度{detection['confidence']:.0%}）。")
    tl = {"肺部肿块", "肺结节"}
    if detection['class_name'] in tl:
        if staging_features.get('size_gt3cm'):
            parts.append("肿块长径超过3cm，符合T2期特征。")
        if staging_features.get('lymph_node_enlargement'):
            parts.append("肺门/纵隔淋巴结肿大，提示N1/N2可能。")
        if staging_features.get('bone_destruction'):
            parts.append("肋骨骨质破坏，提示M1b。")
        if not any(staging_features.values()):
            parts.append("未发现明确分期相关征象。")
        parts.append(f"综合分期推测：{estimated_stage}。")
    else:
        parts.append(f"非肿瘤性征象，按临床常规处理。")
    return ''.join(parts)
