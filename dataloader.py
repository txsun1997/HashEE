import os
import torch
import json
from torch.utils.data import Dataset
from transformers import BertTokenizer
from fastNLP import DataSet, Instance
from fastNLP.io import Loader
from functools import partial
from transformers import glue_convert_examples_to_features as convert_examples_to_features


class SSTLoader(Loader):
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
                raw_words, target = line.split(self.sep)
                if raw_words.endswith("\""):
                    raw_words = raw_words[:-1]
                if raw_words.startswith('"'):
                    raw_words = raw_words[1:]
                raw_words = raw_words.replace('""', '"')
                ds.append(Instance(raw_words=raw_words, labels=int(target)))
        ds.apply_field(func=self.tokenizer.encode, field_name="raw_words", new_field_name="input_ids")
        ds.apply_field(func=len, field_name="input_ids", new_field_name="seq_len")
        ds.set_input("input_ids", "labels")
        ds.set_target("labels")
        return ds


class MRPCLoader(Loader):
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
                target, sentence1, sentence2 = line.split('\t')[0], line.split('\t')[3], line.split('\t')[4]
                # raw_words, target = line.split(self.sep)
                # if raw_words.endswith("\""):
                #     raw_words = raw_words[:-1]
                # if raw_words.startswith('"'):
                #     raw_words = raw_words[1:]
                # raw_words = raw_words.replace('""', '"')
                input = self.tokenizer([(sentence1, sentence2)])
                ds.append(Instance(input_ids=input['input_ids'][0], token_type_ids=input['token_type_ids'][0],
                                   attention_mask=input['attention_mask'][0], labels=int(target)))
        ds.apply_field(func=len, field_name="input_ids", new_field_name="seq_len")
        ds.set_input("input_ids", "token_type_ids", "attention_mask", "labels")
        ds.set_target("labels")
        return ds


class STSBLoader(Loader):
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
                sentence1, sentence2, target = line.split('\t')
                input = self.tokenizer([(sentence1, sentence2)])
                ds.append(Instance(input_ids=input['input_ids'][0], token_type_ids=input['token_type_ids'][0],
                                   attention_mask=input['attention_mask'][0], labels=float(target)))
        ds.apply_field(func=len, field_name="input_ids", new_field_name="seq_len")
        ds.set_input("input_ids", "token_type_ids", "attention_mask", "labels")
        ds.set_target("labels")
        return ds


class IMDbLoader(Loader):
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
                raw_words, target = line.split(self.sep)
                if raw_words.endswith("\""):
                    raw_words = raw_words[:-1]
                if raw_words.startswith('"'):
                    raw_words = raw_words[1:]
                raw_words = raw_words.replace('""', '"')
                ds.append(Instance(raw_words=raw_words, labels=int(target)))
        ds.apply_field(func=partial(self.tokenizer.encode, max_length=512, truncation=True), field_name="raw_words",
                       new_field_name="input_ids")
        ds.apply_field(func=len, field_name="input_ids", new_field_name="seq_len")
        ds.set_input("input_ids", "labels")
        ds.set_target("labels")
        return ds


class NLILoader(Loader):
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
                sentence1, sentence2, target = line.split('\t')
                input = self.tokenizer([(sentence1, sentence2)])
                ds.append(Instance(input_ids=input['input_ids'][0], token_type_ids=input['token_type_ids'][0],
                                   attention_mask=input['attention_mask'][0], labels=int(target)))
        ds.apply_field(func=len, field_name="input_ids", new_field_name="seq_len")
        ds.set_input("input_ids", "token_type_ids", "attention_mask", "labels")
        ds.set_target("labels")
        return ds