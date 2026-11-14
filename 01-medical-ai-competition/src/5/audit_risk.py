def generate_audit_flags(result):
    flags = {
        'authenticity_issue': False,
        'duplicate_or_spliced': False,
        'suspicious_fraud': False,
        'clinical_evidence': ''
    }
    if result.get('stage') == 'authenticity_failed':
        flags['authenticity_issue'] = True
        flags['suspicious_fraud'] = True
    if result.get('stage') == 'normality_failed':
        norm_detail = result.get('normality', {})
        if norm_detail.get('reason') in ['重复影像', '拼接影像']:
            flags['duplicate_or_spliced'] = True
            flags['suspicious_fraud'] = True
    if result.get('stage') == 'complete':
        detections = result.get('detections', [])
        tl = {"肺部肿块", "肺结节"}
        tumor_dets = [d for d in detections if d.get('class_name') in tl]
        if detections:
            finding_summary = "; ".join([f"{d['class_name']}({d.get('laterality','')})" for d in detections[:5]])
            stages = [d.get('estimated_stage', '') for d in tumor_dets if d.get('estimated_stage')]
            stage_str = f"，分期{', '.join(set(stages))}" if stages else ""
            flags['clinical_evidence'] = f"检出{len(detections)}处病灶：{finding_summary}{stage_str}"
        else:
            flags['clinical_evidence'] = "未检出明确病灶"
    return flags
