import os

data_dir = './snli_1.0/'
difficult_samples = []
simple_neutral_samples = []
simple_entail_samples = []
simple_contradict_samples = []

with open(os.path.join(data_dir, 'snli_1.0_train.txt'), 'r') as fin:
    lines = fin.readlines()[1:]
    for line in lines:
        columns = line.strip().split('\t')
        label = columns[0]
        sentence_1 = columns[5]
        sentence_2 = columns[6]
        if label == 'neutral':
            sample = {
                'sentence_1': sentence_1,
                'sentence_2': sentence_2,
                'label': 0,
            }
            simple_neutral_samples.append(sample)
        elif label == 'contradiction':
            sample = {
                'sentence_1': sentence_1,
                'sentence_2': sentence_2,
                'label': 0,
            }
            simple_contradict_samples.append(sample)
        elif label == 'entailment':
            sample = {
                'sentence_1': sentence_1,
                'sentence_2': sentence_2,
                'label': 0,
            }
            simple_entail_samples.append(sample)
        elif label == '-':
            sample = {
                'sentence_1': sentence_1,
                'sentence_2': sentence_2,
                'label': 1,
            }
            difficult_samples.append(sample)
        else:
            print("error label: {}".format(label))

# print(len(difficult_samples))

with open(os.path.join(data_dir, 'snli_1.0_dev.txt'), 'r') as fin:
    lines = fin.readlines()[1:]
    for line in lines:
        columns = line.strip().split('\t')
        label = columns[0]
        sentence_1 = columns[5]
        sentence_2 = columns[6]
        if label == 'neutral':
            sample = {
                'sentence_1': sentence_1,
                'sentence_2': sentence_2,
                'label': 0,
            }
            simple_neutral_samples.append(sample)
        elif label == 'contradiction':
            sample = {
                'sentence_1': sentence_1,
                'sentence_2': sentence_2,
                'label': 0,
            }
            simple_contradict_samples.append(sample)
        elif label == 'entailment':
            sample = {
                'sentence_1': sentence_1,
                'sentence_2': sentence_2,
                'label': 0,
            }
            simple_entail_samples.append(sample)
        elif label == '-':
            sample = {
                'sentence_1': sentence_1,
                'sentence_2': sentence_2,
                'label': 1,
            }
            difficult_samples.append(sample)
        else:
            print("error label: {}".format(label))

# print(len(difficult_samples))

with open(os.path.join(data_dir, 'snli_1.0_test.txt'), 'r') as fin:
    lines = fin.readlines()[1:]
    for line in lines:
        columns = line.strip().split('\t')
        label = columns[0]
        sentence_1 = columns[5]
        sentence_2 = columns[6]
        if label == 'neutral':
            sample = {
                'sentence_1': sentence_1,
                'sentence_2': sentence_2,
                'label': 0,
            }
            simple_neutral_samples.append(sample)
        elif label == 'contradiction':
            sample = {
                'sentence_1': sentence_1,
                'sentence_2': sentence_2,
                'label': 0,
            }
            simple_contradict_samples.append(sample)
        elif label == 'entailment':
            sample = {
                'sentence_1': sentence_1,
                'sentence_2': sentence_2,
                'label': 0,
            }
            simple_entail_samples.append(sample)
        elif label == '-':
            sample = {
                'sentence_1': sentence_1,
                'sentence_2': sentence_2,
                'label': 1,
            }
            difficult_samples.append(sample)
        else:
            print("error label: {}".format(label))

num_difficult_samples = len(difficult_samples)
print(num_difficult_samples)

import random

random.shuffle(simple_entail_samples)
random.shuffle(simple_contradict_samples)
random.shuffle(simple_neutral_samples)
random.shuffle(difficult_samples)

split_num = num_difficult_samples - 500
split_per_class = split_num // 3
train_difficult = difficult_samples[:split_num]
train_simple = simple_neutral_samples[:split_per_class] + simple_contradict_samples[
                                                          :split_per_class] + simple_entail_samples[:split_per_class]
train_set = train_difficult + train_simple
random.shuffle(train_set)
print(len(train_set))

num_test = 500
new_split_per_class = num_test // 3
test_difficult = difficult_samples[split_num:]
test_simple = simple_neutral_samples[split_per_class:split_per_class + new_split_per_class] + simple_contradict_samples[
                                                                                              split_per_class:split_per_class + new_split_per_class] + simple_entail_samples[
                                                                                                                                                       split_per_class:split_per_class + new_split_per_class + 2]
test_set = test_difficult + test_simple
random.shuffle(test_set)
print(len(test_set))

with open('./snli_difficulty/train.txt', 'w') as fout:
    fout.write("sentence1\tsentence2\tlabel\n")
    for sample in train_set:
        fout.write("{}\t{}\t{}\n".format(sample['sentence_1'], sample['sentence_2'], sample['label']))

with open('./snli_difficulty/test.txt', 'w') as fout:
    fout.write("sentence1\tsentence2\tlabel\n")
    for sample in test_set:
        fout.write("{}\t{}\t{}\n".format(sample['sentence_1'], sample['sentence_2'], sample['label']))
