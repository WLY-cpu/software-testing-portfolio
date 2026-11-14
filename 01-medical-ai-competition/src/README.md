# 2026 AI 大赛 · 赛道八：胸部 X 光智能诊断系统

> 参赛作品源码（个人留存版）。输入一张胸片，输出「真伪判定 → 规范性审核 → 病灶检测 → 分期推测 → 可解释热力图 → 结构化报告 + 医保审核风险提示」。
>
> ⚠️ 这是比赛原型，**不是医疗器械**，不能用于临床诊断。

---

## 一、它能做什么

| 关卡 | 做什么 | 技术方案 |
|---|---|---|
| 1 真实性检测 | 判断是不是真实人体胸片（排除翻拍、合成、非人体图） | **双分支**：空域 ResNet50 + 频域 FFT-ResNet18 → 拼接 → 3 分类（real / fake / nonhuman） |
| 2 规范性审核 | 查重复片、拼接篡改 | 感知哈希去重（`hash_db.pkl`）+ 拼接 CNN + ELA 误差分析 |
| 3 病灶检测 | 找出肿块、结节、气胸、积液等 | YOLOv8m 目标检测（`ultralytics`），7 类病灶 |
| 4 分期推测 | 对肿块/结节区域估分期 | ResNet50 分类头（7 类）+ 病灶裁剪 |
| 5 可解释与审计 | 热力图 + 关键点测量 + 风险提示 | GradCAM（挂在分期模型上）+ 关键点检测 + 规则审计 |

## 二、目录结构

```
src/
├── 1/                         # 关卡1 真实性
│   ├── train_authenticity.py  # 双分支模型训练（空域+频域）
│   ├── inference_module.py    # AuthenticityGate 推理封装
│   ├── eval_authenticity.py   # 评估（AUC 等）
│   ├── find_threshold.py      # 阈值搜索
│   ├── generate_gradcam.py
│   ├── gradcam_utils.py
│   └── build_dataset.py       # 数据集构建
├── 2/                         # 关卡2 规范性
│   ├── train_splicing_cnn.py
│   ├── splicing_detector.py
│   ├── ela_detector.py        # ELA 误差水平分析
│   ├── duplicate_detector.py  # 感知哈希去重
│   ├── normality_gate.py      # 门控：重复/拼接/异常
│   └── generate_spliced_data.py
├── 3/                         # 关卡3 病灶检测
│   ├── train_yolo.py          # YOLOv8m 训练
│   ├── lesion_detector.py     # 推理 + 左右侧/直径换算
│   ├── convert_public_dataset.py
│   └── generate_mock_data.py
├── 4/                         # 关卡4 分期
│   ├── train_staging.py
│   ├── staging_model.py       # ResNet50 → 7 类
│   ├── staging_predictor.py
│   └── generate_staging_mock.py
├── 5/                         # 关卡5 审计与解释
│   ├── keypoint_detector.py
│   ├── explainer.py           # GradCAM + 文本解释
│   └── audit_risk.py          # 医保审核风险规则
├── pipeline.py                # 五关卡串联（核心）
├── competition_output.py      # 竞赛要求的 JSON 输出
├── inference_server.py        # HTTP 推理服务（供平台回调）
├── gui.py                     # Tkinter 图形界面
├── zong.py                    # 总入口（CLI）
├── path_utils.py              # 路径与 sys.path 管理（关键！）
├── train_logger.py
├── tools/merge_weights.py     # 权重分片合并
└── requirements.txt
```

## 三、怎么跑

```bash
# 1. 装依赖（建议用虚拟环境；CPU 也能跑，只是慢）
pip install -r requirements.txt

# 2. 单张图片，命令行看报告
python zong.py 你的胸片.png P001

# 3. 图形界面
python zong.py --gui

# 4. 启动 HTTP 服务（平台回调用）
python zong.py --server --port 8000
```

`path_utils.setup_module_paths()` 会把 `1~5` 目录加入 `sys.path`，所以各关卡之间可以直接 `from lesion_detector import LesionDetector`，**不要**改成包导入（目录名是数字，不能当包名）。

## 四、还缺什么才能跑起来（重要）

代码是完整的，但**权重和数据不在**，需要补齐这 5 个文件：

| 路径 | 是什么 | 大小量级 |
|---|---|---|
| `1/best_real_fake_detector.pth` | 真实性双分支权重 | 约 100–150 MB |
| `2/splicing_cnn.pth` | 拼接检测 CNN | 约 45 MB |
| `3/runs/detect/cxr_lesion/weights/best.pt` | YOLOv8m 病灶权重 | 约 50 MB |
| `4/staging_classifier.pth` | 分期分类器 | 约 100 MB |
| `hash_db.pkl` | 重复片哈希库（可空跑，会退化为只查拼接） | 视数据量 |

另外还需要 `1/data/authenticity/{train,val}/{0_real,1_fake,2_nonhuman}/` 这样的数据目录，才能在 `pipeline.py` 的 `__main__` 里跑自测。

**没有权重时**：可以用 `generate_*_mock_data.py` 造测试数据，用 `train_*.py` 重新训一个小模型验证流程是否通。

## 五、平台相关路径

代码里有 `/2026aicompetition/workspace/...` 这样的绝对路径（比赛容器内的目录），但都做了 `os.path.isdir(...)` 判断，**在本地会自动跳过**，不影响运行。

## 六、已知局限

- 分期只做参考，训练样本不均衡，越靠后的分期越不可靠
- 拼接检测对重度压缩的图片敏感度下降（ELA 依赖压缩痕迹）
- GradCAM 只解释分期模型，不解释检测模型
- 没有做并发与显存管理，`inference_server.py` 是单进程顺序处理
