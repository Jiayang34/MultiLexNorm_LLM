from sklearn.metrics import accuracy_score
def evaluate_metrics(y_raw, y_true, y_pred, ignCaps=True):
    """
    原作者逻辑适配版：处理平铺的 List[str]
    """
    if not (len(y_raw) == len(y_true) == len(y_pred)):
        raise ValueError(f"长度不一致: Raw({len(y_raw)}), True({len(y_true)}), Pred({len(y_pred)})")

    cor = 0        # 预测正确的总数
    changed = 0    # 原始文本中需要修改的总数
    total = len(y_true)

    # 四大核心计数器
    true_positive = 0   # wrong raw, correct replacement
    true_negative = 0   # correct raw, no replacement
    false_positive = 0  # correct raw, wrong replacement
    false_negative = 0  # wrong raw, wrong replacement

    false_positives_idx = []
    false_negatives_idx = []

    for i, (wordRaw, wordGold, wordPred) in enumerate(zip(y_raw, y_true, y_pred)):
        # change all words to lower case
        if ignCaps:
            wordRaw = wordRaw.lower()
            wordGold = wordGold.lower()
            wordPred = wordPred.lower()
        # model did replacement
        if wordRaw != wordGold:
            changed += 1
        # model did correct replacement
        if wordGold == wordPred:
            cor += 1

        if wordRaw == wordGold and wordGold == wordPred:
            true_negative += 1
        elif wordRaw == wordGold and wordGold != wordPred:
            false_positive += 1
            false_positives_idx.append(i)
        elif wordRaw != wordGold and wordGold != wordPred:
            false_negative += 1
            false_negatives_idx.append(i)
        elif wordRaw != wordGold and wordGold == wordPred:
            true_positive += 1

    accuracy = cor / total if total > 0 else 0.0
    lai = (total - changed) / total if total > 0 else 0.0
    err = (accuracy - lai) / (1 - lai) if lai < 1.0 else float("-inf")
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0.0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0.0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0

    return {
        "accuracy": accuracy,
        "lai": lai,
        "err": err,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "fp_indices": false_positives_idx
    }