import os
import torch
import json
from torch.utils.data import Dataset
from transformers import BertTokenizer
from fastNLP import DataSet, Instance
from fastNLP.io import Loader
from transformers import glue_convert_examples_to_features as convert_examples_to_features


class SNLILoader(Loader):
    def __init__(self, sep='\t', has_header=True, tokenizer=None):
        super().__init__()
        self.sep = sep
        self.has_header = has_header
        self.tokenizer = tokenizer

    def _load(self, path: str) -> DataSet:
        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as f:
            read_header = self.has_header
            for line in f:
                if read_header:
                    read_header = False
                    continue
                line = line.strip()
                sentence_1, sentence_2, label = line.split('\t')
                input = self.tokenizer([(sentence_1, sentence_2)])
                ds.append(Instance(input_ids=input['input_ids'][0], token_type_ids=input['token_type_ids'][0],
                                   attention_mask=input['attention_mask'][0], labels=int(label)))

        ds.apply_field(func=len, field_name="input_ids", new_field_name="seq_len")
        ds.set_input("input_ids", "token_type_ids", "attention_mask", "labels")
        ds.set_target("labels")
        return ds


class MultiLabelSNLILoader(Loader):
    def __init__(self, sep='\t', has_header=True, tokenizer=None):
        super().__init__()
        self.sep = sep
        self.has_header = has_header
        self.tokenizer = tokenizer

    def _parse_label(self, label: str):
        label = label.strip().strip('[').strip(']')
        res = []
        for l in label.split(','):
            res.append(int(l.strip()))
        assert len(res) == 12
        return res

    def _load(self, path: str) -> DataSet:
        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as f:
            read_header = self.has_header
            for line in f:
                if read_header:
                    read_header = False
                    continue
                line = line.strip()
                sentence_1, sentence_2, label = line.split('\t')
                label = self._parse_label(label)
                input = self.tokenizer([(sentence_1, sentence_2)])
                ds.append(Instance(input_ids=input['input_ids'][0], token_type_ids=input['token_type_ids'][0],
                                   attention_mask=input['attention_mask'][0], labels=label))

        ds.apply_field(func=len, field_name="input_ids", new_field_name="seq_len")
        ds.set_input("input_ids", "token_type_ids", "attention_mask", "labels")
        ds.set_target("labels")
        return ds


def stat_majority(path):
    layer_acc = [0 for _ in range(12)]
    total = 0
    with open(path, 'r') as fin:
        lines = fin.readlines()[1:]
        for line in lines:
            sentence_1, sentence_2, label = line.split('\t')
            label = label.strip().strip('[').strip(']')
            res = []
            for l in label.split(','):
                res.append(int(l.strip()))
            assert len(res) == 12
            for i, l in enumerate(res):
                if l == 1:
                    layer_acc[i] += 1
            total += 1
    layer_acc = [x * 1.0 / total for x in layer_acc]
    return layer_acc


# layer_acc = stat_majority(path='./snli_model_difficulty/test.txt')
# print(layer_acc)
# majority = []
# for acc in layer_acc:
#     if acc > 0.5:
#         majority.append(acc)
#     else:
#         majority.append(1 - acc)
# print(majority)