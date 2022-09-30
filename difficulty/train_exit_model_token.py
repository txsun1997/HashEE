import os
import sys
import random
import numpy as np

sys.path.append('../')
import argparse

import torch
import fitlog
from fastNLP import Trainer, Tester, CrossEntropyLoss, LossInForward, AccuracyMetric, SequentialSampler, DataSetIter
from fastNLP import BucketSampler, GradientClipCallback, WarmupCallback, FitlogCallback, cache_results
from transformers import AdamW, BertTokenizer, BertConfig
from analysis.modeling_bert_exit import BertForTokenClassification
from fastNLP.core.utils import _move_dict_value_to_device
from fastNLP.io import OntoNotesNERLoader
from fastNLP import Vocabulary, FieldArray, DataSet


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--data_dir", default="./ontonotes/", type=str)
    parser.add_argument("--lr", default=3e-5, type=float, required=False)
    parser.add_argument("--batch_size", default=32, type=int, required=False)
    parser.add_argument("--n_epochs", default=5, type=int, required=False)
    parser.add_argument("--seed", default=6, type=int, required=False)
    parser.add_argument("--warmup", default=0.1, type=float, required=False)
    parser.add_argument("--weight_decay", default=0.1, type=float, required=False)
    parser.add_argument("--logging_steps", default=50, type=int, required=False)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument('--debug', action='store_true', help="do not log")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.weight_decay = 0.1
    args.adam_epsilon = 1e-8
    args.debug = True
    set_seed(args)
    if args.debug:
        fitlog.debug()

    log_dir = './logs_NER'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    fitlog.set_log_dir(log_dir)
    # fitlog.commit(__file__)
    fitlog.add_hyper(args)
    if args.gpu != 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    cache_fn = f"caches/data_Ontonotes.pt"

    @cache_results(cache_fn, _refresh=False)
    def get_data(args, tokenizer):
        data_paths = {
            'train': os.path.join(args.data_dir, 'train.txt'),
            'dev': os.path.join(args.data_dir, 'dev.txt'),
        }
        data_bundle = OntoNotesNERLoader().load(data_paths)
        train_data, dev_data = data_bundle.get_dataset('train'), data_bundle.get_dataset('dev')
        target_vocab = Vocabulary(padding=None, unknown=None)
        target_vocab.from_dataset(train_data, dev_data, field_name='target')
        target_vocab.index_dataset(train_data, field_name='target')
        target_vocab.index_dataset(dev_data, field_name='target')

        input_ids = []
        labels = []
        for ins in train_data:
            tmp_input_ids, tmp_labels = [101], [target_vocab.word2idx['O']]
            for x, y in zip(ins['raw_words'], ins['target']):
                tokenized_x = tokenizer.encode(x, add_special_tokens=False)
                tmp_input_ids.extend(tokenized_x)
                tmp_labels.extend([y] * len(tokenized_x))
            tmp_input_ids.append(102)
            tmp_labels.append(target_vocab.word2idx['O'])
            input_ids.append(tmp_input_ids)
            labels.append(tmp_labels)
        field_array = FieldArray('input_ids', input_ids)
        train_data.add_fieldarray('input_ids', field_array)
        field_array = FieldArray('labels', labels)
        train_data.add_fieldarray('labels', field_array)

        input_ids = []
        labels = []
        for ins in dev_data:
            tmp_input_ids, tmp_labels = [101], [target_vocab.word2idx['O']]
            for x, y in zip(ins['raw_words'], ins['target']):
                tokenized_x = tokenizer.encode(x, add_special_tokens=False)
                tmp_input_ids.extend(tokenized_x)
                tmp_labels.extend([y] * len(tokenized_x))
            tmp_input_ids.append(102)
            tmp_labels.append(target_vocab.word2idx['O'])
            input_ids.append(tmp_input_ids)
            labels.append(tmp_labels)
        field_array = FieldArray('input_ids', input_ids)
        dev_data.add_fieldarray('input_ids', field_array)
        field_array = FieldArray('labels', labels)
        dev_data.add_fieldarray('labels', field_array)
        return data_bundle, target_vocab


    data_bundle, target_vocab = get_data(args, tokenizer)
    train_data, dev_data = data_bundle.get_dataset('train'), data_bundle.get_dataset('dev')
    train_data.apply_field(func=len, field_name="input_ids", new_field_name="seq_len")
    train_data.apply_field(func=lambda x: [1] * x, field_name="seq_len", new_field_name="attention_mask")
    train_data.set_input('input_ids', 'attention_mask', 'labels')
    train_data.set_target('labels')

    dev_data.apply_field(func=len, field_name="input_ids", new_field_name="seq_len")
    dev_data.apply_field(func=lambda x: [1] * x, field_name="seq_len", new_field_name="attention_mask")
    dev_data.set_input('input_ids', 'attention_mask', 'labels')
    dev_data.set_target('labels')
    # train_data = train_data[:15000]
    print('# of train data: {}'.format(len(train_data)))
    print('# of dev data: {}'.format(len(dev_data)))

    num_labels = len(target_vocab)

    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    model = BertForTokenClassification.from_pretrained(args.model_name_or_path, config=config)
    # no_decay = ['bias', 'LayerNorm.weight', 'embedding']
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    # ]
    #
    # optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
    # metric = AccuracyMetric(target="labels")
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # # tester = Tester(data=dev_data, model=model, metrics=metric, device=device)
    #
    # callbacks = []
    # callbacks.append(GradientClipCallback(clip_value=1, clip_type='norm'))
    # callbacks.append(WarmupCallback(warmup=args.warmup, schedule='linear'))
    # # callbacks.append(FitlogCallback(tester=tester, log_loss_every=args.logging_steps, verbose=1))
    #
    # trainer = Trainer(train_data=train_data, model=model, loss=LossInForward(), optimizer=optimizer, batch_size=32,
    #                   sampler=BucketSampler(seq_len_field_name='seq_len'), drop_last=False, update_every=1,
    #                   num_workers=4, n_epochs=args.n_epochs, print_every=5, dev_data=dev_data, metrics=metric,
    #                   validate_every=args.logging_steps, save_path='./saved_models/', use_tqdm=True, device=device,
    #                   callbacks=callbacks, dev_batch_size=args.batch_size * 2)
    # trainer.train(load_best_model=True)
    fitlog.finish()

    # annotate train and dev data
    saved_model = torch.load('./saved_models/best_BertForTokenClassification_acc_2021-11-03-08-05-37-047535')
    model.load_state_dict(saved_model.state_dict())
    metric = AccuracyMetric(target="labels")
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model.to(device)
    input_ids, targets = [], []
    dev_data.set_input('seq_len')
    data_iterator = DataSetIter(dataset=dev_data, batch_size=64, sampler=SequentialSampler(), num_workers=4)
    print(len(data_iterator))
    for batch_x, batch_y in data_iterator:
        _move_dict_value_to_device(batch_x, batch_y, device=torch.device(device))
        model_input = {
            'input_ids': batch_x['input_ids'],
            'attention_mask': batch_x['attention_mask'],
            'labels': batch_x['labels'],
        }
        difficulty_label = model.annotate(**model_input)
        difficulty_label = torch.stack(difficulty_label)  # num_layer x batch_size x seq_len
        difficulty_label = difficulty_label.transpose(0, 1).transpose(1,2)  # batch_size x seq_len x num_layer
        print(difficulty_label.shape)
        batch_seq_len = batch_x['seq_len']
        batch_input_ids = batch_x['input_ids']
        for i, tmp_input_ids in enumerate(batch_input_ids):
            input_ids.append(tmp_input_ids[:batch_seq_len[i]])
        for i, i_label in enumerate(difficulty_label):
            targets.append(i_label[:batch_seq_len[i], :].long().numpy().tolist())  # seq_len x num_layer
    print(len(targets))

    from fastNLP import Instance
    difficulty_dataset = DataSet()
    for ipt_ids, tgt in zip(input_ids, targets):
        difficulty_dataset.append(Instance(input_ids=ipt_ids, labels=tgt))
    diff_train, diff_dev = difficulty_dataset.split(ratio=0.2806)

    input_ids_field = FieldArray('input_ids', input_ids)
    targets_field = FieldArray('labels', targets)
    difficulty_dataset.add_fieldarray('input_ids', input_ids_field)
    difficulty_dataset.add_fieldarray('labels', targets_field)

    import pickle
    data_to_dump = {
        'train': diff_train,
        'dev': diff_dev,
    }
    fout = open('./ontonotes_difficulty.bin', 'wb')
    pickle.dump(data_to_dump, fout)
    fout.close()
    # read_header = True
    # s1, s2 = [], []
    # with open(os.path.join(args.data_dir, 'dev.tsv'), 'r') as fin:
    #     for line in fin:
    #         if read_header:
    #             read_header = False
    #             continue
    #         line = line.strip()
    #         sentence_1, sentence_2, _ = line.split('\t')
    #         s1.append(sentence_1)
    #         s2.append(sentence_2)
    # assert len(s1) == len(targets)
    # num_train = 8000
    # with open('./snli_model_difficulty/train.txt', 'w') as fout:
    #     fout.write("sentence1\tsentence2\tlabel\n")
    #     for i in range(num_train):
    #         print(i)
    #         fout.write("{}\t{}\t{}\n".format(s1[i], s2[i], targets[i]))
    # print('\n')
    # with open('./snli_model_difficulty/test.txt', 'w') as fout:
    #     fout.write("sentence1\tsentence2\tlabel\n")
    #     for i in range(num_train, len(targets)):
    #         print(i)
    #         fout.write("{}\t{}\t{}\n".format(s1[i], s2[i], targets[i]))
