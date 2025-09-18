# jc 23.10.31

from datetime import datetime
from sklearn.metrics import accuracy_score
import numpy as np

class Evaluator(object):

    def __init__(self, target, pred, metric_type='f1') -> None:
        self.target = target
        self.pred = pred
        self.metric_type = metric_type

    def evaluate(self, dataset, task):
        return eval_f1_micro(self.pred, self.target)

def eval_f1_micro(prediction, target):
    prediction, target = lowercase_and_strip(prediction), lowercase_and_strip(target)
    tp, pred, gold = 0, 0, 0
    tp_list, pred_list, gold_list = [], [], []
    for i, _ in enumerate(prediction):
        try:
            cnt = len(set(prediction[i]) & set(target[i])) if not (len(prediction[i]) == 0 and len(target[i]) == 0) else 1
            tp += cnt
            pred += len(prediction[i]) if len(prediction[i]) != 0 else 1    # In case of type: empty entity
            gold += len(target[i]) if len(target[i]) != 0 else 1
        except:
            print(prediction[i], target[i])
        tp_list.append(tp)
        pred_list.append(pred)
        gold_list.append(gold)
    precision = tp / pred if pred != 0 else 0
    recall = tp / gold if gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    return round(precision*100, 2), round(recall*100, 2), round(f1 * 100, 2)#, tp_list, pred_list, gold_list

def eval_f1_macro(prediction, target):
    prediction, target = lowercase_and_strip(prediction), lowercase_and_strip(target)
    precision_list, recall_list, f1_list = [], [], []
    for i, _ in enumerate(prediction):
        cnt = len(set(prediction[i]) & set(target[i]))
        precision = cnt / len(prediction[i]) if len(prediction[i]) != 0 else 0
        recall = cnt / len(target[i]) if len(target[i]) != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    precision_list = np.array(precision_list)
    recall_list = np.array(recall_list)
    f1_list = np.array(f1_list)
    return round(precision_list.mean() * 100, 2), round(recall_list.mean() * 100, 2), round(f1_list.mean() * 100, 2)

def lowercase_and_strip(item):
    if isinstance(item, str):
        # 如果 item 是字符串，就将它转换为小写并去除头尾空格
        return item.lower().strip()
    elif isinstance(item, list):
        # 如果 item 是列表，就对它的每个元素调用 lowercase_and_strip
        return [lowercase_and_strip(subitem) for subitem in item]
    else:
        # 如果 item 不是字符串也不是列表，就直接返回 item
        return item


if __name__ == '__main__':
    # p = "/home/jc/workspace/exp/figs/data/llama/ner/bbn/check/bbn_ner_natural_llama-7b_shot-15_maxline-300_rpenal-1.0_maxtoken-300_beam-1.json.1.checked-25-shot.log"
    pass
    # sentence, target, prediction, checked, metrics_before_check = load_checked(p)
    # _, _, _, tp_un, pred_un, gold_un = eval_f1_micro(prediction, target)
    # _, _, _, tp_ch, pred_ch, gold_ch = eval_f1_micro(checked, target)
    # for i in range(len(sentence)):
    #     print(f"Unchecked: {tp_un[i]}, {pred_un[i]}, {gold_un[i]}")
    #     p_un, r_un = tp_un[i]/pred_un[i], tp_un[i]/gold_un[i]
    #     f_un = 2 * p_un * r_un / (p_un + r_un) if (p_un + r_un) != 0 else 0

    #     p_ch, r_ch = tp_ch[i]/pred_ch[i], tp_ch[i]/gold_ch[i]
    #     f_ch = 2 * p_ch * r_ch / (p_ch + r_ch) if (p_un + r_un) != 0 else 0
    #     print(f"Checked  : {tp_ch[i]}, {pred_ch[i]}, {gold_ch[i]}")

    #     print(f"Unchecked: {round(p_un, 3)}, {round(r_un, 3)}, {round(f_un, 3)}")
    #     print(f"Checked  : {round(p_ch, 3)}, {round(r_ch, 3)}, {round(f_ch, 3)}\n\n")

    

