import torch
import math
import numpy as np
from fastNLP.core.metrics import MetricBase
from fastNLP.core.utils import _get_func_signature
import random


class MultiLabelMetric(MetricBase):
    def __init__(self, pred=None, target=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=None)
        self.acc_count = 0
        self.total = 0
        self._target = []
        self._pred = []
        self._layer_acc = [0 for _ in range(12)]
        # self.pos_rate = [0.36264929424538545, 0.43322475570032576, 0.5548317046688382, 0.7801302931596091, 0.7991313789359392, 0.8127035830618893, 0.8159609120521173, 0.8203040173724213, 0.8262757871878393, 0.8295331161780674, 0.8333333333333334, 0.8360477741585234]


    def evaluate(self, pred, target, seq_len=None):
        '''
        :param pred: batch_size x num_labels
        :param target: batch_size x num_labels
        :param seq_len: not uesed when doing text classification
        :return:
        '''

        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if pred.dim() != target.dim():
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        pred = pred.detach().cpu().numpy()
        target = target.to('cpu').numpy()
        cnt = 0
        y1, y2 = [], []
        for x1, x2 in zip(pred, target):  # for each instance
            yy1 = []
            yy2 = []
            for i in range(len(x1)):  # for each layer
                # prob = random.uniform(0, 1)
                # if prob < self.pos_rate[i]:
                #     x1[i] = 1
                # else:
                #     x1[i] = 0

                if x1[i] <= 0:
                    yy1.append(i)
                if x2[i] <= 0:
                    yy2.append(i)
                if (x1[i] > 0 and x2[i] > 0) or (x1[i] <= 0 and x2[i] <= 0):
                    self._layer_acc[i] += 1
                # if np.sign(x1[i]) == np.sign(x2[i]):  # compute each layer's accuracy
                #     self._layer_acc[i] += 1
            y1.append(yy1)
            y2.append(yy2)
            cnt += set(yy1) == set(yy2)

        self.acc_count += cnt
        self.total += len(pred)
        self._pred.extend(y1)
        self._target.extend(y2)

    def get_metric(self, reset=True):
        # for calculating macro F1
        num_predict, num_golden = 0, 0
        p = 0.
        r = 0.
        # for calculating micro F1
        num_predicted_labels = 0.
        num_golden_labels = 0.
        num_correct_labels = 0.

        for true_labels, predicted_labels in zip(self._target, self._pred):
            overlap = len(set(predicted_labels).intersection(set(true_labels)))
            # calculating macro F1
            if len(predicted_labels) > 0:
                p += overlap / float(len(predicted_labels))
                num_predict += 1
            if len(true_labels) > 0:
                r += overlap / float(len(true_labels))  # r就是sensitivity
                num_golden += 1
            # calculating micro F1
            num_predicted_labels += len(predicted_labels)
            num_golden_labels += len(true_labels)
            num_correct_labels += overlap

        if num_predict > 0:
            macro_precision = p / num_predict
        else:
            macro_precision = 0.
        macro_recall = r / num_golden  # sensitivity
        macro_f = self._calculate_f1(macro_precision, macro_recall)

        if num_predicted_labels > 0:
            micro_precision = num_correct_labels / num_predicted_labels
        else:
            micro_precision = 0.
        micro_recall = num_correct_labels / num_golden_labels
        micro_f = self._calculate_f1(micro_precision, micro_recall)

        layer_acc = [round(num_correct * 1.0 / self.total, 3) for num_correct in self._layer_acc]
        # evaluate_result = {'micro_f': micro,
        #                    'micro_p': micro_precision,
        #                    'micro_r': micro_recall,
        #                    'acc': round(float(self.acc_count) / (self.total + 1e-12), 6),
        #                    # 'macro_p': macro_precision,
        #                    # 'macro_r': macro_recall,
        #                    # 'macro_f': macro,
        #                    }
        mean_layer_acc = 0.0
        for x in layer_acc:
            mean_layer_acc += x
        mean_layer_acc = mean_layer_acc / len(layer_acc)
        evaluate_result = {
            # 'acc': round(float(self.acc_count) / (self.total + 1e-12), 3),
            'layer_acc': layer_acc,
            'mean': mean_layer_acc,
            # 'macro_p': macro_precision,
            # 'macro_r': macro_recall,
            # 'macro_f': macro_f,
            # 'micro_p': micro_precision,
            # 'micro_r': micro_recall,
            'micro_f': micro_f,
        }
        if reset:
            self.acc_count = 0
            self.total = 0
            self._pred = []
            self._target = []
            self._layer_acc = [0 for _ in range(12)]

        return evaluate_result

    def _calculate_f1(self, p, r):
        if r == 0.:
            return 0.
        return 2 * p * r / float(p + r)


class MultiLabelMetricToken(MetricBase):
    def __init__(self, pred=None, target=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=None)
        self.total_tokens = 0
        self._layer_acc = [0 for _ in range(12)]
        self.acc_count = 0
        self._target = []
        self._pred = []

    def evaluate(self, pred, target, seq_len=None):
        '''
        :param pred: batch_size x seq_len x num_labels
        :param target: batch_size x seq_len x num_labels
        :param seq_len: not uesed when doing text classification
        :return:
        '''

        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if pred.dim() != target.dim():
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")
        pred = pred.transpose(1, 2)
        target = target.transpose(1, 2)
        pred = pred.detach().cpu().numpy()
        target = target.to('cpu').numpy()
        cnt = 0
        y1, y2 = [], []
        for x1, x2 in zip(pred, target):  # for each instance
            yy1 = []
            yy2 = []
            for i in range(len(x1)):  # for each layer
                for j in range(len(x1[i])):  # for each token
                    if x1[i][j] <= 0:
                        yy1.append((i,j))
                    if x2[i][j] <= 0:
                        yy2.append((i,j))
                    if (x1[i][j] > 0 and x2[i][j] > 0) or (x1[i][j] <= 0 and x2[i][j] <= 0):
                        self._layer_acc[i] += 1
                    self.total_tokens += 1
            y1.append(yy1)
            y2.append(yy2)
            cnt += set(yy1) == set(yy2)

        self.acc_count += cnt
        self._pred.extend(y1)
        self._target.extend(y2)


    def get_metric(self, reset=True):
        # for calculating macro F1
        num_predict, num_golden = 0, 0
        p = 0.
        r = 0.
        # for calculating micro F1
        num_predicted_labels = 0.
        num_golden_labels = 0.
        num_correct_labels = 0.

        for true_labels, predicted_labels in zip(self._target, self._pred):
            overlap = len(set(predicted_labels).intersection(set(true_labels)))
            # calculating macro F1
            if len(predicted_labels) > 0:
                p += overlap / float(len(predicted_labels))
                num_predict += 1
            if len(true_labels) > 0:
                r += overlap / float(len(true_labels))
                num_golden += 1
            # calculating micro F1
            num_predicted_labels += len(predicted_labels)
            num_golden_labels += len(true_labels)
            num_correct_labels += overlap

        if num_predict > 0:
            macro_precision = p / num_predict
        else:
            macro_precision = 0.
        macro_recall = r / num_golden  # sensitivity
        macro_f = self._calculate_f1(macro_precision, macro_recall)

        if num_predicted_labels > 0:
            micro_precision = num_correct_labels / num_predicted_labels
        else:
            micro_precision = 0.
        micro_recall = num_correct_labels / num_golden_labels
        micro_f = self._calculate_f1(micro_precision, micro_recall)

        layer_acc = [round(num_correct * 12.0 / self.total_tokens, 3) for num_correct in self._layer_acc]
        # evaluate_result = {'micro_f': micro,
        #                    'micro_p': micro_precision,
        #                    'micro_r': micro_recall,
        #                    'acc': round(float(self.acc_count) / (self.total + 1e-12), 6),
        #                    # 'macro_p': macro_precision,
        #                    # 'macro_r': macro_recall,
        #                    # 'macro_f': macro,
        #                    }
        mean_layer_acc = 0.0
        for x in layer_acc:
            mean_layer_acc += x
        mean_layer_acc = mean_layer_acc / len(layer_acc)
        evaluate_result = {
            # 'acc': round(float(self.acc_count) / (self.total_tokens + 1e-12), 3),
            'layer_acc': layer_acc,
            'mean': mean_layer_acc,
            # 'macro_p': macro_precision,
            # 'macro_r': macro_recall,
            # 'macro_f': macro_f,
            # 'micro_p': micro_precision,
            # 'micro_r': micro_recall,
            'micro_f': micro_f,
        }
        if reset:
            self.acc_count = 0
            self.total_tokens = 0
            self._pred = []
            self._target = []
            self._layer_acc = [0 for _ in range(12)]

        return evaluate_result

    def _calculate_f1(self, p, r):
        if r == 0.:
            return 0.
        return 2 * p * r / float(p + r)



# class MultiLabelMetric(MetricBase):
#     def __init__(self, pred=None, target=None):
#         super().__init__()
#         self._init_param_map(pred=pred, target=target, seq_len=None)
#         self.acc_count = 0
#         self.total = 0
#         self._target = []
#         self._pred = []
#         self._neg_target = []
#         self._neg_pred = []
#         self._layer_acc = [0 for _ in range(12)]
#         # self.pos_rate = [0.36264929424538545, 0.43322475570032576, 0.5548317046688382, 0.7801302931596091, 0.7991313789359392, 0.8127035830618893, 0.8159609120521173, 0.8203040173724213, 0.8262757871878393, 0.8295331161780674, 0.8333333333333334, 0.8360477741585234]
#
#
#     def evaluate(self, pred, target, seq_len=None):
#         '''
#         :param pred: batch_size x num_labels
#         :param target: batch_size x num_labels
#         :param seq_len: not uesed when doing text classification
#         :return:
#         '''
#
#         if not isinstance(pred, torch.Tensor):
#             raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
#                             f"got {type(pred)}.")
#         if not isinstance(target, torch.Tensor):
#             raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
#                             f"got {type(target)}.")
#
#         if pred.dim() != target.dim():
#             raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
#                                f"size:{pred.size()}, target should have size: {pred.size()} or "
#                                f"{pred.size()[:-1]}, got {target.size()}.")
#
#         pred = pred.detach().cpu().numpy()
#         target = target.to('cpu').numpy()
#         cnt = 0
#         y1, y2 = [], []
#         n1, n2 = [], []
#         for x1, x2 in zip(pred, target):  # for each instance
#             yy1 = []
#             yy2 = []
#             nn1 = []
#             nn2 = []
#             for i in range(len(x1)):  # for each layer
#                 # prob = random.uniform(0, 1)
#                 # if prob < self.pos_rate[i]:
#                 #     x1[i] = 1
#                 # else:
#                 #     x1[i] = 0
#
#                 if x1[i] > 0:
#                     yy1.append(i)
#                 if x2[i] > 0:
#                     yy2.append(i)
#                 if x1[i] <= 0:
#                     nn1.append(i)
#                 if x2[i] <= 0:
#                     nn2.append(i)
#                 if (x1[i] > 0 and x2[i] > 0) or (x1[i] <= 0 and x2[i] <= 0):
#                     self._layer_acc[i] += 1
#                 # if np.sign(x1[i]) == np.sign(x2[i]):  # compute each layer's accuracy
#                 #     self._layer_acc[i] += 1
#             y1.append(yy1)
#             y2.append(yy2)
#             n1.append(nn1)
#             n2.append(nn2)
#             cnt += set(yy1) == set(yy2)
#
#         self.acc_count += cnt
#         self.total += len(pred)
#         self._pred.extend(y1)
#         self._target.extend(y2)
#         self._neg_pred.extend(n1)
#         self._neg_target.extend(n2)
#
#     def get_metric(self, reset=True):
#         # for calculating macro F1
#         num_predict, num_golden = 0, 0
#         p = 0.
#         r = 0.
#         s = 0.
#         # for calculating micro F1
#         num_predicted_labels = 0.
#         num_golden_labels = 0.
#         num_correct_labels = 0.
#
#         for true_labels, predicted_labels in zip(self._target, self._pred):
#             overlap = len(set(predicted_labels).intersection(set(true_labels)))
#             # calculating macro F1
#             if len(predicted_labels) > 0:
#                 p += overlap / float(len(predicted_labels))
#                 num_predict += 1
#             if len(true_labels) > 0:
#                 r += overlap / float(len(true_labels))  # r就是sensitivity
#                 num_golden += 1
#             # calculating micro F1
#             num_predicted_labels += len(predicted_labels)
#             num_golden_labels += len(true_labels)
#             num_correct_labels += overlap
#
#         if num_predict > 0:
#             macro_precision = p / num_predict
#         else:
#             macro_precision = 0.
#         macro_recall = r / num_golden  # sensitivity
#         macro_f = self._calculate_f1(macro_precision, macro_recall)
#
#         if num_predicted_labels > 0:
#             micro_precision = num_correct_labels / num_predicted_labels
#         else:
#             micro_precision = 0.
#         micro_recall = num_correct_labels / num_golden_labels
#         micro_f = self._calculate_f1(micro_precision, micro_recall)
#
#         # calculate specificity metric
#         num_neg_golden = 0
#         for true_labels, predicted_labels in zip(self._neg_target, self._neg_pred):
#             overlap = len(set(predicted_labels).intersection(set(true_labels)))
#             # calculating macro G-Mean
#             if len(true_labels) > 0:
#                 s += overlap / float(len(true_labels))
#                 num_neg_golden += 1
#
#         macro_specificity = s / num_neg_golden
#         macro_g = math.sqrt(macro_specificity * macro_recall)
#
#         layer_acc = [round(num_correct * 1.0 / self.total, 3) for num_correct in self._layer_acc]
#         # evaluate_result = {'micro_f': micro,
#         #                    'micro_p': micro_precision,
#         #                    'micro_r': micro_recall,
#         #                    'acc': round(float(self.acc_count) / (self.total + 1e-12), 6),
#         #                    # 'macro_p': macro_precision,
#         #                    # 'macro_r': macro_recall,
#         #                    # 'macro_f': macro,
#         #                    }
#         evaluate_result = {
#             'acc': round(float(self.acc_count) / (self.total + 1e-12), 3),
#             # 'layer_acc': layer_acc,
#             'macro_p': macro_precision,
#             'macro_r': macro_recall,
#             'macro_f': macro_f,
#             'macro_sensitivity': macro_recall,
#             'macro_specificity': macro_specificity,
#             'macro_g': macro_g,
#         }
#         if reset:
#             self.acc_count = 0
#             self.total = 0
#             self._pred = []
#             self._target = []
#             self._neg_pred = []
#             self._neg_target = []
#             self._layer_acc = [0 for _ in range(12)]
#
#         return evaluate_result
#
#     def _calculate_f1(self, p, r):
#         if r == 0.:
#             return 0.
#         return 2 * p * r / float(p + r)


# class MultiLabelMetricToken(MetricBase):
#     def __init__(self, pred=None, target=None):
#         super().__init__()
#         self._init_param_map(pred=pred, target=target, seq_len=None)
#         self.total_tokens = 0
#         self._layer_acc = [0 for _ in range(12)]
#         self.acc_count = 0
#         self._target = []
#         self._pred = []
#         self._neg_target = []
#         self._neg_pred = []
#
#     def evaluate(self, pred, target, seq_len=None):
#         '''
#         :param pred: batch_size x seq_len x num_labels
#         :param target: batch_size x seq_len x num_labels
#         :param seq_len: not uesed when doing text classification
#         :return:
#         '''
#
#         if not isinstance(pred, torch.Tensor):
#             raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
#                             f"got {type(pred)}.")
#         if not isinstance(target, torch.Tensor):
#             raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
#                             f"got {type(target)}.")
#
#         if pred.dim() != target.dim():
#             raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
#                                f"size:{pred.size()}, target should have size: {pred.size()} or "
#                                f"{pred.size()[:-1]}, got {target.size()}.")
#         pred = pred.transpose(1, 2)
#         target = target.transpose(1, 2)
#         pred = pred.detach().cpu().numpy()
#         target = target.to('cpu').numpy()
#         cnt = 0
#         y1, y2 = [], []
#         n1, n2 = [], []
#         for x1, x2 in zip(pred, target):  # for each instance
#             yy1 = []
#             yy2 = []
#             nn1 = []
#             nn2 = []
#             for i in range(len(x1)):  # for each layer
#                 for j in range(len(x1[i])):  # for each token
#                     if x1[i][j] > 0:
#                         yy1.append((i,j))
#                     if x2[i][j] > 0:
#                         yy2.append((i,j))
#                     if x1[i][j] <= 0:
#                         nn1.append((i,j))
#                     if x2[i][j] <= 0:
#                         nn2.append((i,j))
#                     if (x1[i][j] > 0 and x2[i][j] > 0) or (x1[i][j] <= 0 and x2[i][j] <= 0):
#                         self._layer_acc[i] += 1
#                         # print('RIGHT')
#                     self.total_tokens += 1
#                 # if np.sign(x1[i]) == np.sign(x2[i]):  # compute each layer's accuracy
#                 #     self._layer_acc[i] += 1
#             y1.append(yy1)
#             y2.append(yy2)
#             n1.append(nn1)
#             n2.append(nn2)
#             cnt += set(yy1) == set(yy2)
#
#         self.acc_count += cnt
#         self._pred.extend(y1)
#         self._target.extend(y2)
#         self._neg_pred.extend(n1)
#         self._neg_target.extend(n2)
#
#
#     def get_metric(self, reset=True):
#         num_golden = 0
#         r = 0.
#         s = 0.
#         for true_labels, predicted_labels in zip(self._target, self._pred):
#             overlap = len(set(predicted_labels).intersection(set(true_labels)))
#             # calculating macro F1
#             if len(true_labels) > 0:
#                 r += overlap / float(len(true_labels))  # r就是sensitivity
#                 num_golden += 1
#
#         macro_recall = r / num_golden  # sensitivity
#
#         # calculate specificity metric
#         num_neg_golden = 0
#         for true_labels, predicted_labels in zip(self._neg_target, self._neg_pred):
#             overlap = len(set(predicted_labels).intersection(set(true_labels)))
#             # calculating macro G-Mean
#             if len(true_labels) > 0:
#                 s += overlap / float(len(true_labels))
#                 num_neg_golden += 1
#
#         macro_specificity = s / num_neg_golden
#         macro_g = math.sqrt(macro_specificity * macro_recall)
#
#         layer_acc = [round(num_correct * 12.0 / self.total_tokens, 3) for num_correct in self._layer_acc]
#
#         evaluate_result = {
#             # 'acc': round(float(self.acc_count) / (self.total + 1e-12), 3),
#             'layer_acc': layer_acc,
#             'macro_sensitivity': macro_recall,
#             'macro_specificity': macro_specificity,
#             'macro_g': macro_g,
#         }
#         if reset:
#             self.acc_count = 0
#             self.total_tokens = 0
#             self._pred = []
#             self._target = []
#             self._neg_pred = []
#             self._neg_target = []
#             self._layer_acc = [0 for _ in range(12)]
#
#         # layer_acc = [round(num_correct * 12.0 / self.total_tokens, 3) for num_correct in self._layer_acc]
#
#         # evaluate_result = {
#         #     'layer_acc': layer_acc,
#         # }
#         # if reset:
#         #     self.total_tokens = 0
#         #     self._layer_acc = [0 for _ in range(12)]
#
#         return evaluate_result
#
#     def _calculate_f1(self, p, r):
#         if r == 0.:
#             return 0.
#         return 2 * p * r / float(p + r)
