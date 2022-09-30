import torch
import numpy as np
from fastNLP.core.metrics import MetricBase
from fastNLP.core.utils import _get_func_signature
from fastNLP.core import seq_len_to_mask
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import pearsonr, spearmanr
import warnings


class AccAndF1Metric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self.total = 0
        self._target = []
        self._pred = []

    def evaluate(self, pred, target, seq_len=None):
        '''
        :param pred: batch_size x num_labels
        :param target: batch_size
        :param seq_len: not uesed when doing text classification
        :return:
        '''

        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        target = target.to('cpu').numpy().tolist()
        self._pred.extend(pred)
        self._target.extend(target)

    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        f_score = f1_score(self._target, self._pred)
        evaluate_result = {
            'acc_and_f1': (acc + f_score) / 2,
            'acc': acc,
            'f1': f_score,
        }
        if reset:
            self.total = 0
            self._target = []
            self._pred = []
        return evaluate_result


class PearsonSpearmanCorr(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._target = []
        self._pred = []

    def evaluate(self, pred, target, seq_len=None):
        '''
        :param pred: batch_size
        :param target: batch_size
        :param seq_len: not uesed when doing text classification
        :return:
        '''

        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        pred = pred.squeeze().detach().cpu().numpy().tolist()
        target = target.to('cpu').numpy().tolist()
        self._pred.extend(pred)
        self._target.extend(target)

    def get_metric(self, reset=True):
        pearson_corr = pearsonr(self._pred, self._target)[0]
        spearman_corr = spearmanr(self._pred, self._target)[0]
        evaluate_result = {
            'pearson': round(pearson_corr, 3),
            'spearmanr': round(spearman_corr, 3),
            'corr': round((pearson_corr + spearman_corr) / 2, 3),
        }
        if reset:
            self._target = []
            self._pred = []
        return evaluate_result