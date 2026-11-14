import sys
import os
import argparse
import cv2
from path_utils import setup_module_paths
setup_module_paths()
from pipeline import CompetitionPipeline
def print_report(result):
    """"""
    if 'error' in result:
        print(f'系统错误：{result["error"]}')
        return
    stage = result.get('stage', '未知状态')
    auth = result.get('authenticity', {})
    norm = result.get('normality', {})
    detections = result.get('detections', [])
    audit = result.get('audit_flags', {})
    print("=" * 50)
    print("【影像真实性检测】")
    if stage == 'authenticity_failed':
        print("  未通过 - 可能不是真实人体胸片")
    else:
        print(f"  通过  真实概率: {auth.get('real_prob', 'N/A')}")
    print(f"  假体概率: {auth.get('fake_prob')}, 非人体概率: {auth.get('nonhuman_prob')}")
    print("\n【影像规范性审核】")
    if stage == 'normality_failed':
        print(f"  未通过 - {norm.get('reason')}")
    else:
        print("  通过 - 无重复或拼接异常")
    print("\n【病灶检测与辅助诊断】")
    if not detections:
        print("  未检出明确病灶。")
    else:
        for i, det in enumerate(detections):
            print(f"  病灶{i+1}: {det['class_name']} 置信度: {det.get('confidence', 'N/A')}")
            print(f"    位置(px): {det['bbox']}")
            print(f"    左右侧: {det.get('laterality', 'N/A')}")
            if 'max_diameter_mm' in det:
                print(f"    最大径: {det['max_diameter_mm']}mm")
            if 'estimated_stage' in det:
                print(f"    分期: {det.get('estimated_stage')}")
    if result.get('explanation'):
        print("\n【诊断可解释性】")
        print(f"  {result['explanation']}")
    print("\n【医保审核风险提示】")
    if audit.get('suspicious_fraud'):
        print("  疑似欺诈案例，建议人工复核。")
    else:
        print("  无欺诈风险")
    print(f"  临床依据: {audit.get('clinical_evidence', '无')}")
    print("\n【最终诊断报告】")
    print(f"  {result.get('report', '')}")
    if result.get('keypoints_img') is not None:
        cv2.imwrite('keypoints_output.jpg', result['keypoints_img'])
        print("\n关键点标注图已保存为 keypoints_output.jpg")
    if result.get('heatmap_img') is not None:
        cv2.imwrite('heatmap_output.jpg', result['heatmap_img'])
        print("病灶热力图已保存为 heatmap_output.jpg")
def run_single_image(img_path, patient_id, device='cpu'):
    """处理单张图像并输出报告"""
    if not os.path.exists(img_path):
        print(f"错误：图片文件不存在 - {img_path}")
        return
    pipeline = CompetitionPipeline(device=device)
    result = pipeline.process(img_path, patient_id, exam_timestamp=None)
    print_report(result)
def run_competition(img_path, patient_id, team_id, eval_id, answer_base):
    """处理单张图像并生成竞赛格式输出"""
    pipeline = CompetitionPipeline(device='cpu')
    out = pipeline.process_with_competition_output(
        img_path, patient_id,
        team_id=team_id,
        evaluation_id=eval_id,
        answer_base=answer_base,
    )
    print("竞赛输出已生成:")
    for key, path in out.get('outputs', {}).items():
        if path:
            print(f"  {key}: {path}")
    print_report(out.get('result', {}))
def main():
    parser = argparse.ArgumentParser(description="赛道八 胸部X光智能诊断系统")
    parser.add_argument("image", nargs="?", help="胸片图像路径")
    parser.add_argument("patient_id", nargs="?", help="患者ID")
    parser.add_argument("--server", action="store_true", help="启动推理服务 (HTTP)")
    parser.add_argument("--port", type=int, default=8000, help="推理服务端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="推理服务地址")
    parser.add_argument("--callback", type=str, default=None,
                        help="平台回调地址（不传时使用环境变量或默认内网地址）")
    parser.add_argument("--competition", action="store_true", help="以竞赛格式输出")
    parser.add_argument("--team-id", default="2054451790241292288", help="队伍ID")
    parser.add_argument("--eval-id", default="default", help="测评ID")
    parser.add_argument("--answer-base", default="/2026aicompetition/workspace/answer", help="输出根目录")
    parser.add_argument("--device", default="cpu", help="计算设备 (cpu/cuda)")
    parser.add_argument("--gui", action="store_true", help="启动GUI界面")
    args = parser.parse_args()
    if args.server:
        from inference_server import serve
        serve(host=args.host, port=args.port,
              callback=args.callback, team_id=args.team_id)
        return
    if args.gui:
        from gui import Application
        app = Application()
        app.mainloop()
        return
    if args.image and args.patient_id:
        if args.competition:
            run_competition(
                args.image, args.patient_id,
                args.team_id, args.eval_id, args.answer_base,
            )
        else:
            run_single_image(args.image, args.patient_id, device=args.device)
        return
    parser.print_help()
if __name__ == '__main__':
    main()
