import torch
import cv2
import numpy as np
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
        """
        input_tensor: [1, 3, H, W] 已归一化的张量
        class_idx: 要可视化的类别索引，默认取预测类别
        """
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
