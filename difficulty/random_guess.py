import math
import random

# sentence-level
path = '/remote-home/txsun/proj/hash-early-exit/analysis/snli_model_difficulty/test.txt'
num_pos = [0 for _ in range(12)]
num_neg = [0 for _ in range(12)]
golden_labels = []
with open(path, 'r') as fin:
    lines = fin.readlines()[1:]
    for line in lines:
        sentence_1, sentence_2, label = line.split('\t')
        label = label.strip().strip('[').strip(']')
        res = []
        for layer_id, l in enumerate(label.split(',')):
            tmp_label = int(l.strip())
            res.append(tmp_label)
            if tmp_label == 1:
                num_pos[layer_id] += 1
            else:
                num_neg[layer_id] += 1
        golden_labels.append(res)
print(num_pos)
print(num_neg)
print(len(golden_labels))
pos_rate = [x/len(golden_labels) for x in num_pos]
print(pos_rate)
# print('# pos: {}, # neg: {}, rate: {}'.format(num_pos, num_neg, pos_rate))

sens, spec = 0., 0.
num_sens, num_spec = 0., 0.
for trial in range(10):
    for i in range(len(golden_labels)):
        ins = golden_labels[i]
        tp, tn = 0., 0.
        pos_labels, neg_labels = 0., 0.
        for layer_id, layer_label in enumerate(ins):
            prob = random.uniform(0, 1)
            if layer_label == 1:
                pos_labels += 1
                if prob < pos_rate[layer_id]:  # guess 1
                    tp += 1
            else:
                neg_labels += 1
                if prob >= pos_rate[layer_id]:  # guess 0
                    tn += 1
        if pos_labels:
            sens += tp / pos_labels
            num_sens += 1
        if neg_labels:
            spec += tn / neg_labels
            num_spec += 1
    # print("Trial {}. tp: {} | tn : {}".format(trial, tp, tn))
macro_sens = sens / num_sens
macro_spec = spec / num_spec
print('TPR: {}, TNR: {}, GMean: {}'.format(macro_sens, macro_spec, math.sqrt(macro_sens * macro_spec)))


# token-level
