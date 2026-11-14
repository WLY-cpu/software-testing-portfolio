import sys
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import cv2
from path_utils import setup_module_paths
setup_module_paths()
from pipeline import CompetitionPipeline
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("胸部X光智能辅助诊断系统")
        self.geometry("900x800")
        self.pipeline = None
        self.img_path = None
        frame_top = tk.Frame(self)
        frame_top.pack(pady=10)
        tk.Label(frame_top, text="选择胸片文件:").grid(row=0, column=0, padx=5, sticky='e')
        self.entry_path = tk.Entry(frame_top, width=50)
        self.entry_path.grid(row=0, column=1, padx=5)
        tk.Button(frame_top, text="浏览", command=self.browse_file).grid(row=0, column=2, padx=5)
        tk.Label(frame_top, text="患者ID:").grid(row=1, column=0, padx=5, pady=10, sticky='e')
        self.entry_pid = tk.Entry(frame_top, width=20)
        self.entry_pid.grid(row=1, column=1, padx=5, pady=10, sticky='w')
        self.btn_run = tk.Button(self, text="开始分析", command=self.run_analysis,
                                 bg='#4CAF50', fg='white', font=('微软雅黑', 12))
        self.btn_run.pack(pady=10)
        self.progress = tk.Label(self, text="")
        self.progress.pack()
        frame_result = tk.Frame(self)
        frame_result.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.text_result = scrolledtext.ScrolledText(frame_result, height=25, font=('Consolas', 10))
        self.text_result.pack(fill=tk.BOTH, expand=True)
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=5)
        self.btn_save_kp = tk.Button(btn_frame, text="保存关键点标注图",
                                     command=self.save_keypoint_img, state='disabled')
        self.btn_save_kp.grid(row=0, column=0, padx=5)
        self.btn_save_heat = tk.Button(btn_frame, text="保存热力图",
                                       command=self.save_heatmap_img, state='disabled')
        self.btn_save_heat.grid(row=0, column=1, padx=5)
        self.last_result = None
    def browse_file(self):
        filetypes = [("图像文件", "*.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*")]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self.entry_path.delete(0, tk.END)
            self.entry_path.insert(0, path)
            self.img_path = path
    def run_analysis(self):
        img_path = self.entry_path.get().strip()
        patient_id = self.entry_pid.get().strip()
        if not img_path:
            messagebox.showerror("错误", "请选择胸片文件！")
            return
        if not patient_id:
            messagebox.showerror("错误", "请输入患者ID！")
            return
        self.btn_run.config(state='disabled', text="分析中...")
        self.progress.config(text="正在分析，请稍候...")
        self.text_result.delete(1.0, tk.END)
        thread = threading.Thread(target=self._analysis_worker, args=(img_path, patient_id))
        thread.daemon = True
        thread.start()
    def _analysis_worker(self, img_path, patient_id):
        try:
            if self.pipeline is None:
                self.pipeline = CompetitionPipeline(device='cpu')
            result = self.pipeline.process(img_path, patient_id, exam_timestamp=None)
            self.last_result = result
            self.after(0, self._show_result, result)
        except Exception as e:
            self.after(0, self._show_error, str(e))
    def _show_result(self, result):
        self.progress.config(text="分析完成")
        self.btn_run.config(state='normal', text="开始分析")
        self.text_result.delete(1.0, tk.END)
        self.text_result.insert(tk.END, "========== 分析结果 ==========\n\n")
        auth = result.get('authenticity', {})
        self.text_result.insert(tk.END, "【影像真实性检测】\n")
        if result.get('stage') == 'authenticity_failed':
            self.text_result.insert(tk.END, "❌ 未通过！可能不是真实人体胸片。\n")
        else:
            self.text_result.insert(tk.END, f"✅ 通过  真实概率: {auth.get('real_prob', 'N/A')}\n")
        self.text_result.insert(tk.END, f"   假体概率: {auth.get('fake_prob')}, 非人体概率: {auth.get('nonhuman_prob')}\n\n")
        norm = result.get('normality', {})
        self.text_result.insert(tk.END, "【影像规范性审核】\n")
        if result.get('stage') == 'normality_failed':
            self.text_result.insert(tk.END, f"❌ 未通过！原因: {norm.get('reason')}\n")
        else:
            self.text_result.insert(tk.END, "✅ 通过  无重复或拼接异常\n")
        self.text_result.insert(tk.END, "\n")
        detections = result.get('detections', [])
        self.text_result.insert(tk.END, "【病灶检测与辅助诊断】\n")
        if not detections:
            self.text_result.insert(tk.END, "未检出明确病灶。\n")
        else:
            for i, det in enumerate(detections):
                self.text_result.insert(tk.END, f"病灶{i+1}: {det['class_name']} 置信度: {det.get('confidence', 'N/A')}\n")
                self.text_result.insert(tk.END, f"   位置: {det['bbox']}\n")
                if 'estimated_stage' in det:
                    self.text_result.insert(tk.END, f"   推测分期: {det['estimated_stage']}\n")
        self.text_result.insert(tk.END, "\n")
        if result.get('explanation'):
            self.text_result.insert(tk.END, "【诊断可解释性】\n")
            self.text_result.insert(tk.END, result['explanation'] + "\n\n")
        audit = result.get('audit_flags', {})
        self.text_result.insert(tk.END, "【医保审核风险提示】\n")
        if audit.get('suspicious_fraud'):
            self.text_result.insert(tk.END, "⚠️ 疑似欺诈案例，建议人工复核。\n")
            if audit.get('authenticity_issue'):
                self.text_result.insert(tk.END, "  - 影像真实性异常（非真实人体/翻拍/体模）\n")
            if audit.get('duplicate_or_spliced'):
                self.text_result.insert(tk.END, "  - 影像规范性异常（重复或拼接）\n")
        else:
            self.text_result.insert(tk.END, "✅ 无欺诈风险\n")
        self.text_result.insert(tk.END, f"临床依据: {audit.get('clinical_evidence', '无')}\n\n")
        self.text_result.insert(tk.END, "【最终诊断报告】\n")
        self.text_result.insert(tk.END, result.get('report', '') + "\n")
        self.btn_save_kp.config(state='normal' if result.get('keypoints_img') is not None else 'disabled')
        self.btn_save_heat.config(state='normal' if result.get('heatmap_img') is not None else 'disabled')
    def _show_error(self, msg):
        self.progress.config(text="")
        self.btn_run.config(state='normal', text="开始分析")
        messagebox.showerror("分析失败", f"发生错误:\n{msg}")
    def save_keypoint_img(self):
        if self.last_result and self.last_result.get('keypoints_img') is not None:
            path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg")])
            if path:
                cv2.imwrite(path, self.last_result['keypoints_img'])
                messagebox.showinfo("保存成功", f"关键点标注图已保存至:\n{path}")
    def save_heatmap_img(self):
        if self.last_result and self.last_result.get('heatmap_img') is not None:
            path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg")])
            if path:
                cv2.imwrite(path, self.last_result['heatmap_img'])
                messagebox.showinfo("保存成功", f"热力图已保存至:\n{path}")
if __name__ == '__main__':
    app = Application()
    app.mainloop()
