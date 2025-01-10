def gsm8k_eval_func(q, pred, gt):
    try:
        gt = gt[gt.rfind("####") :].split()[1]
        pred = pred[pred.rfind("####") :].split()[1]
        return float(gt == pred)
    except Exception:
        return 0.0
