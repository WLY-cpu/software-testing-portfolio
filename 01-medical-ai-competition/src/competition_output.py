"""赛道八 V2.0 竞赛输出生成。
V2.0 输出结构：
  - 每个 AccessionNumber 一个 JSON：{AccessionNumber}.json
  - 顶层字段：AccessionNumber / IsNotHumanBodyProb / IsStitchedProb /
    ProcessingTime_ms / Prediction
  - 重复影像对单独输出 duplicate_pairs.jsonl
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
FINDING_CLASSES = [
    "纤维条索影", "片状模糊影", "空气支气管征", "间质性改变",
    "肺水肿", "肺不张", "肺气肿", "支气管扩张",
    "肺部肿块", "肺结节", "空洞", "肺内钙化",
    "胸膜增厚", "胸腔积液", "气胸", "纵隔增宽", "心脏增大",
]
FINDING_EN_MAP = {
    "纤维条索影": "Fibrosis",
    "片状模糊影": "Consolidation",
    "空气支气管征": "AirBronchogram",
    "间质性改变": "Infiltration",
    "肺水肿": "Edema",
    "肺不张": "Atelectasis",
    "肺气肿": "Emphysema",
    "支气管扩张": "Bronchiectasis",
    "肺部肿块": "Mass",
    "肺结节": "Nodule",
    "空洞": "Cavity",
    "肺内钙化": "Calcification",
    "胸膜增厚": "PleuralThickening",
    "胸腔积液": "PleuralEffusion",
    "气胸": "Pneumothorax",
    "纵隔增宽": "MediastinalWidening",
    "心脏增大": "Cardiomegaly",
}
FINDING_EN_KEYS = list(FINDING_EN_MAP.values())
def _finding_probs_en(probs):
    """把概率字典规整为 17 个英文键、独立 Sigmoid 概率。
    V2.0：Finding_probabilities 各值独立 ∈[0,1]，不要求总和为 1。
    """
    allowed = set(FINDING_EN_KEYS)
    out = {}
    for key, value in (probs or {}).items():
        en = key if key in allowed else FINDING_EN_MAP.get(key)
        if en:
            out[en] = max(0.0, min(1.0, float(value)))
    for en in FINDING_EN_KEYS:
        out.setdefault(en, 0.0)
    return out
def _normalize_probabilities(probs, decimals=4):
    """Softmax 型概率归一化（用于 EffusionGrade.probabilities 等）。"""
    items = [(k, max(0.0, float(v))) for k, v in probs.items()]
    total = sum(v for _, v in items)
    if total <= 0:
        n = max(1, len(items))
        items = [(k, 1.0 / n) for k, _ in items]
        total = 1.0
    out = {k: round(v / total, decimals) for k, v in items}
    diff = round(1.0 - sum(out.values()), decimals)
    if diff:
        key = max(out, key=out.get)
        out[key] = round(max(0.0, min(1.0, out[key] + diff)), decimals)
    return out
def _not_human_body_prob(auth_result: Dict[str, Any]) -> float:
    """由真实性结果计算“假人体”概率 = 1 - 真人体概率。"""
    if not isinstance(auth_result, dict):
        return 0.0
    for key in ("is_not_human_body_prob", "not_human_body_prob",
                "IsNotHumanBodyProb"):
        if key in auth_result and auth_result[key] is not None:
            return max(0.0, min(1.0, float(auth_result[key])))
    real_prob = auth_result.get("real_prob")
    if real_prob is not None:
        return round(max(0.0, min(1.0, 1.0 - float(real_prob))), 4)
    return 0.0
def _interpretation(detections: List[Dict[str, Any]]):
    """生成 Conclusion / KeyFindings。"""
    if detections:
        parts = []
        key_findings = []
        for d in detections:
            finding_cn = d.get("class_name", "未知")
            conf = d.get("confidence", 0)
            diam = d.get("max_diameter_mm", "")
            diam_str = f"，最大径{diam}mm" if diam else ""
            parts.append(f"{finding_cn}{diam_str}")
            key_findings.append(f"{finding_cn}：置信度{conf:.0%}"
                                + (f"，最大径{diam}mm" if diam else ""))
        conclusion = f"共检出{len(detections)}个病灶：" + "，".join(parts) + "。"
    else:
        conclusion = ("双肺未见明确实变、肿块、结节、气胸、胸腔积液等异常征象。"
                      "心影大小及纵隔未见异常。")
        key_findings = ["双肺清晰，未见活动性病变", "心影不大，纵隔不宽"]
    return conclusion, key_findings
def generate_all_outputs(
    team_id: str,
    evaluation_id: str,
    answer_base: str,
    accession_number: str,
    image_size: Dict[str, int],
    pixel_spacing: List[float],
    auth_result: Dict[str, Any],
    norm_result: Dict[str, Any],
    detections: List[Dict[str, Any]],
    processing_time_ms: int,
) -> str:
    """生成 {AccessionNumber}.json（V2.0 输出）。返回文件路径。"""
    output_dir = os.path.join(answer_base, str(evaluation_id))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    is_not_human = _not_human_body_prob(auth_result)
    if isinstance(norm_result, dict):
        is_stitched = norm_result.get("splice_prob", 0.0)
    else:
        is_stitched = 0.0
    conclusion, key_findings = _interpretation(detections)
    dets_out = []
    for det in detections:
        bbox_pixel = det.get("bbox", [0, 0, 0, 0])
        bbox_mm = [
            round(bbox_pixel[0] * pixel_spacing[0], 2),
            round(bbox_pixel[1] * pixel_spacing[1], 2),
            round(bbox_pixel[2] * pixel_spacing[0], 2),
            round(bbox_pixel[3] * pixel_spacing[1], 2),
        ]
        finding_cn = det.get("class_name", "")
        finding = FINDING_EN_MAP.get(finding_cn, finding_cn)
        confidence = det.get("confidence", 0.0)
        finding_probs = det.get("finding_probs", None)
        if finding_probs is None:
            finding_probs = {en: 0.01 for en in FINDING_EN_KEYS}
            finding_probs[finding] = confidence
        finding_probs = _finding_probs_en(finding_probs)
        det_entry = {
            "LesionID": det.get("LesionID", f"DET_{len(dets_out)+1:03d}"),
            "BBox_mm": bbox_mm,
            "Finding": finding,
            "Confidence": round(confidence, 4),
            "Finding_probabilities": finding_probs,
        }
        add_attr = {}
        if finding_cn in ("肺部肿块", "肺肿块") or finding == "Mass":
            diam_mm = round(max(
                bbox_mm[2] - bbox_mm[0],
                bbox_mm[3] - bbox_mm[1],
            ), 2)
            add_attr["MaxDiameter_mm"] = max(0.0, diam_mm)
        if finding_cn == "胸腔积液" or finding == "PleuralEffusion":
            eff_probs = _normalize_probabilities(det.get(
                "effusion_probs", {"1": 0.05, "2": 0.90, "3": 0.05}))
            add_attr["EffusionGrade"] = {
                "predicted": det.get("effusion_grade", 2),
                "probabilities": eff_probs,
            }
        if add_attr:
            det_entry["AdditionalAttributes"] = add_attr
        dets_out.append(det_entry)
    prediction = {
        "ImageSize": {
            "width_px": image_size.get("width", 0),
            "height_px": image_size.get("height", 0),
            "pixel_spacing_mm": pixel_spacing,
        },
        "Detections": dets_out,
        "Interpretation": {
            "Conclusion": conclusion,
            "KeyFindings": key_findings,
        },
    }
    payload = {
        "AccessionNumber": accession_number,
        "IsNotHumanBodyProb": round(float(is_not_human), 4),
        "IsStitchedProb": round(float(is_stitched), 4),
        "ProcessingTime_ms": int(processing_time_ms),
        "Prediction": prediction,
    }
    out_path = os.path.join(output_dir, f"{accession_number}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path
def generate_duplicate_pairs_jsonl(
    dup_pairs: List[Tuple[str, str, float]],
    output_dir: str,
) -> str:
    """生成 V2.0 的 duplicate_pairs.jsonl。
    每行：{"AccessionNumber": A, "AccessionNumber_dup": B, "PairProb": p}
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(output_dir, "duplicate_pairs.jsonl")
    lines = []
    for acc_a, acc_b, prob in dup_pairs:
        if not acc_a or not acc_b or acc_a == acc_b:
            continue
        lines.append(json.dumps({
            "AccessionNumber": acc_a,
            "AccessionNumber_dup": acc_b,
            "PairProb": round(max(0.0, min(1.0, float(prob))), 4),
        }, ensure_ascii=False))
    if lines:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    return out_path
