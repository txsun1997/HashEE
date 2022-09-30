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
from transformers import BertTokenizer, BertConfig
from torch.optim import Adam
from analysis.modeling_bert import BertForMultiLabelSequenceClassification
from analysis.metrics import MultiLabelMetric

from analysis.dataloader import MultiLabelSNLILoader
from other_models import LSTMModel, LinearMModel


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str)
    parser.add_argument("--data_dir", default="./snli_model_difficulty", type=str)
    parser.add_argument("--task_name", default="SNLI", type=str)
    parser.add_argument("--lr", default=1e-2, type=float, required=False)
    parser.add_argument("--batch_size", default=32, type=int, required=False)
    parser.add_argument("--n_epochs", default=10, type=int, required=False)
    parser.add_argument("--seed", default=6, type=int, required=False)
    parser.add_argument("--warmup", default=0.0, type=float, required=False)
    parser.add_argument("--weight_decay", default=0.1, type=float, required=False)
    parser.add_argument("--logging_steps", default=100, type=int, required=False)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument('--debug', action='store_true', help="do not log")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.weight_decay = 0.1
    args.adam_epsilon = 1e-8
    set_seed(args)
    args.debug = True
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

    num_labels = 12

    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    bert_model = BertForMultiLabelSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    bert_embedding = bert_model.bert.embeddings.word_embeddings.weight.data
    # model = LSTMModel(
    #     embedding=bert_embedding,
    #     hidden_size=256,
    #     dropout=0.3,
    #     num_layers=1,
    # )
    model = LinearMModel(embedding=bert_embedding, num_labels=num_labels)
    cache_fn = f"caches/anal_data_{args.task_name}.pt"


    @cache_results(cache_fn, _refresh=False)
    def get_data(args, tokenizer):
        data_paths = {
            'train': os.path.join(args.data_dir, 'train.txt'),
            'test': os.path.join(args.data_dir, 'test.txt'),
        }
        data_bundle = MultiLabelSNLILoader(tokenizer=tokenizer).load(data_paths)
        return data_bundle


    data_bundle = get_data(args, tokenizer)
    train_data, test_data = data_bundle.get_dataset('train'), data_bundle.get_dataset('test')
    train_data.set_input('token_type_ids', flag=False)
    train_data.set_input('attention_mask', flag=False)
    test_data.set_input('token_type_ids', flag=False)
    test_data.set_input('attention_mask', flag=False)
    print('# of train data: {}'.format(len(train_data)))
    print('# of test data: {}'.format(len(test_data)))

    optimizer = Adam(model.parameters(), lr=args.lr)
    # metric = AccuracyMetric(target="labels")
    metric = MultiLabelMetric(target="labels")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tester = Tester(data=test_data, model=model, metrics=metric, device=device)

    callbacks = []
    callbacks.append(GradientClipCallback(clip_value=1, clip_type='norm'))
    callbacks.append(WarmupCallback(warmup=args.warmup, schedule='linear'))
    # callbacks.append(FitlogCallback(tester=tester, log_loss_every=args.logging_steps, verbose=1))

    trainer = Trainer(train_data=train_data, model=model, loss=LossInForward(), optimizer=optimizer, batch_size=32,
                      sampler=BucketSampler(seq_len_field_name='seq_len'), drop_last=False, update_every=1,
                      num_workers=4, n_epochs=args.n_epochs, print_every=5, dev_data=test_data, metrics=metric,
                      validate_every=args.logging_steps, save_path=None, use_tqdm=True, device=device,
                      callbacks=callbacks, dev_batch_size=args.batch_size * 2, metric_key='mean')
    trainer.train(load_best_model=False)
    fitlog.finish()
