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
from analysis.modeling_bert_exit import BertForSequenceClassification
from fastNLP.core.utils import _move_dict_value_to_device

from analysis.dataloader import SNLILoader


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str)
    parser.add_argument("--data_dir", default="../efficientNLP_dataset/SNLI", type=str)
    parser.add_argument("--task_name", default="SNLI", type=str)
    parser.add_argument("--lr", default=3e-5, type=float, required=False)
    parser.add_argument("--batch_size", default=32, type=int, required=False)
    parser.add_argument("--n_epochs", default=5, type=int, required=False)
    parser.add_argument("--seed", default=6, type=int, required=False)
    parser.add_argument("--warmup", default=0.1, type=float, required=False)
    parser.add_argument("--weight_decay", default=0.1, type=float, required=False)
    parser.add_argument("--logging_steps", default=50, type=int, required=False)
    parser.add_argument('--gpu', type=str, default='0')
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

    log_dir = './logs_{}'.format(args.task_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    fitlog.set_log_dir(log_dir)
    fitlog.commit(__file__)
    fitlog.add_hyper(args)
    if args.gpu != 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    num_labels = 3

    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    cache_fn = f"caches/data_{args.task_name}.pt"


    @cache_results(cache_fn, _refresh=False)
    def get_data(args, tokenizer):
        data_paths = {
            'train': os.path.join(args.data_dir, 'train.tsv'),
            'dev': os.path.join(args.data_dir, 'dev.tsv'),
        }
        data_bundle = SNLILoader(tokenizer=tokenizer).load(data_paths)
        return data_bundle


    data_bundle = get_data(args, tokenizer)
    train_data, dev_data = data_bundle.get_dataset('train'), data_bundle.get_dataset('dev')
    train_data = train_data[:15000]
    print('# of train data: {}'.format(len(train_data)))
    print('# of dev data: {}'.format(len(dev_data)))

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
    saved_model = torch.load('./saved_models/best_BertForSequenceClassification_acc_2021-10-28-13-03-26-302213')
    model.load_state_dict(saved_model.state_dict())
    metric = AccuracyMetric(target="labels")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    targets = []
    data_iterator = DataSetIter(dataset=dev_data, batch_size=64, sampler=SequentialSampler(), num_workers=4)
    for batch_x, batch_y in data_iterator:
        _move_dict_value_to_device(batch_x, batch_y, device=torch.device(device))
        difficulty_label = model.annotate(**batch_x)
        difficulty_label = torch.stack(difficulty_label)  # num_layer x batch_size
        difficulty_label = difficulty_label.transpose(0, 1)  # batch_size x num_layer
        for i_label in difficulty_label:
            targets.append(i_label.long().numpy().tolist())
    print(len(targets))
    read_header = True
    s1, s2 = [], []
    with open(os.path.join(args.data_dir, 'dev.tsv'), 'r') as fin:
        for line in fin:
            if read_header:
                read_header = False
                continue
            line = line.strip()
            sentence_1, sentence_2, _ = line.split('\t')
            s1.append(sentence_1)
            s2.append(sentence_2)
    assert len(s1) == len(targets)
    num_train = 8000
    with open('./snli_model_difficulty/train.txt', 'w') as fout:
        fout.write("sentence1\tsentence2\tlabel\n")
        for i in range(num_train):
            print(i)
            fout.write("{}\t{}\t{}\n".format(s1[i], s2[i], targets[i]))
    print('\n')
    with open('./snli_model_difficulty/test.txt', 'w') as fout:
        fout.write("sentence1\tsentence2\tlabel\n")
        for i in range(num_train, len(targets)):
            print(i)
            fout.write("{}\t{}\t{}\n".format(s1[i], s2[i], targets[i]))
