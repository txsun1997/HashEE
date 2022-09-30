import os
import sys
import random
import numpy as np

sys.path.append('../')
import argparse

import torch
import fitlog
from fastNLP import Trainer, Tester, CrossEntropyLoss, LossInForward, AccuracyMetric
from fastNLP import BucketSampler, GradientClipCallback, WarmupCallback, FitlogCallback, cache_results
from transformers import AdamW, BertTokenizer

from dataloader import SSTLoader, MRPCLoader, STSBLoader, IMDbLoader, NLILoader
from models.configuration_elasticbert import ElasticBertConfig
# from models.modeling_elasticbert import ElasticBertForSequenceClassification
from models.modeling_elasticbert_hashee import ElasticBertHeeForSequenceClassification
from utils import hash_tokens, cluster_hash_tokens, random_incons_hash_tokens, random_cons_hash_tokens, mi_hash_tokens
from metrics import AccAndF1Metric, PearsonSpearmanCorr


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--task_name", default=None, type=str, required=True)
    parser.add_argument("--lr", default=3e-5, type=float, required=False)
    parser.add_argument("--batch_size", default=32, type=int, required=False)
    parser.add_argument("--n_epochs", default=5, type=int, required=False)
    parser.add_argument("--seed", default=6, type=int, required=False)
    parser.add_argument("--max_layer", default=13, type=int, required=False)
    parser.add_argument("--num_buckets", default=6, type=int, required=False)
    parser.add_argument("--warmup", default=0.1, type=float, required=False)
    parser.add_argument("--weight_decay", default=0.1, type=float, required=False)
    parser.add_argument("--logging_steps", default=50, type=int, required=False)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--hash', type=str, default='frenquency')
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument('--debug', action='store_true', help="do not log")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.adam_epsilon = 1e-8
    set_seed(args)
    if args.debug:
        fitlog.debug()

    log_dir = './new_logs_token_{}'.format(args.task_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    fitlog.set_log_dir(log_dir)
    fitlog.commit(__file__)
    fitlog.add_hyper(args)
    if args.gpu != 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.task_name in ['SNLI']:
        num_labels = 3
    elif args.task_name in ['STS-B']:
        num_labels = 1
    else:
        num_labels = 2

    config = ElasticBertConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    config.num_base_layers = 0
    config.num_output_layers = args.max_layer - 1
    config.num_hidden_layers = args.max_layer - 1
    print('# layers: {}'.format(config.num_hidden_layers))
    model = ElasticBertHeeForSequenceClassification.from_pretrained(args.model_name_or_path,
                                                                    config=config,
                                                                    add_pooling_layer=True)
    cache_fn = f"caches/data_{args.task_name}.pt"
    DataLoader = {
        'SST-2': SSTLoader,
        'MRPC': MRPCLoader,
        'STS-B': STSBLoader,
        'IMDb': IMDbLoader,
        'SciTail': NLILoader,
        'SNLI': NLILoader,
    }

    @cache_results(cache_fn, _refresh=False)
    def get_data(args, tokenizer):
        data_paths = {
            'train': os.path.join(args.data_dir, args.task_name, 'train.tsv'),
            'dev': os.path.join(args.data_dir, args.task_name, 'dev.tsv'),
            'test': os.path.join(args.data_dir, args.task_name, 'test_full.tsv'),
        }
        data_bundle = DataLoader[args.task_name](tokenizer=tokenizer).load(data_paths)
        return data_bundle


    data_bundle = get_data(args, tokenizer)
    hash_map = {
        'random-incons': random_incons_hash_tokens,
        'random-cons': random_cons_hash_tokens,
        'frequency': hash_tokens,
        'mi': mi_hash_tokens,
        'cluster': cluster_hash_tokens,
    }
    data_bundle = hash_map[args.hash](data_bundle, tokenizer, args.max_layer, args.num_buckets)
    train_data, dev_data, test_data = data_bundle.get_dataset('train'), data_bundle.get_dataset(
        'dev'), data_bundle.get_dataset('test')
    print('# of train data: {}'.format(len(train_data)))
    print('# of dev data: {}'.format(len(dev_data)))
    print('# of test data: {}'.format(len(test_data)))

    no_decay = ['bias', 'LayerNorm.weight', 'embedding']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
    if args.task_name == 'MRPC':
        metric = AccAndF1Metric(target='labels')
        metric_key = 'acc_and_f1'
    elif args.task_name == 'STS-B':
        metric = PearsonSpearmanCorr(target='labels')
        metric_key = 'corr'
    else:
        metric = AccuracyMetric(target='labels')
        metric_key = 'acc'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tester = Tester(data=test_data, model=model, metrics=metric, device=device)

    callbacks = []
    callbacks.append(GradientClipCallback(clip_value=1, clip_type='norm'))
    callbacks.append(WarmupCallback(warmup=args.warmup, schedule='linear'))
    callbacks.append(FitlogCallback(tester=tester, log_loss_every=args.logging_steps, verbose=1))

    trainer = Trainer(train_data=train_data, model=model, loss=LossInForward(), optimizer=optimizer,
                      batch_size=args.batch_size, sampler=BucketSampler(seq_len_field_name='seq_len'), drop_last=False,
                      update_every=1, num_workers=4, n_epochs=args.n_epochs, print_every=5, dev_data=test_data,
                      metrics=metric, validate_every=args.logging_steps, save_path=None, use_tqdm=True, device=device,
                      callbacks=callbacks, dev_batch_size=args.batch_size * 2, metric_key=metric_key)
    trainer.train(load_best_model=False)
    fitlog.finish()
