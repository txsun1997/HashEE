import torch
import torch.nn as nn
import numpy as np


def init_embedding(input_embedding, seed=1337):
    """initiate weights in embedding layer
    """
    torch.manual_seed(seed)
    scope = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -scope, scope)


def init_linear(input_linear, seed=1337):
    """initiate weights in linear layer
    """
    torch.manual_seed(seed)
    scope = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform(input_linear.weight, -scope, scope)
    # nn.init.uniform(input_linear.bias, -scope, scope)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


class LSTMModel(nn.Module):

    def __init__(self, embedding, hidden_size, num_labels=12, dropout=0.5, num_layers=1):
        super(LSTMModel, self).__init__()

        self.num_labels = num_labels
        self.hidden_size = hidden_size
        vocab_size, embed_dim = embedding.shape
        self.embed = nn.Embedding(vocab_size, embed_dim, _weight=embedding)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True)

        self.out = nn.Linear(hidden_size * 2, num_labels)

        init_linear(self.out)
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, labels):
        '''
        :param x: batch_size x seq_len
        :param y: batch_size
        :return:
            loss: scale
            pred: batch_size
        '''
        batch_size, seq_len = input_ids.size()
        x = self.dropout(self.embed(input_ids))
        rnn_out, _ = self.lstm(x)
        rnn_out = rnn_out.view(batch_size, seq_len, 2, self.hidden_size)
        rnn_out = torch.cat([rnn_out[:, -1, 0, :], rnn_out[:, 0, 1, :]], dim=-1)
        # rnn_out = rnn_out[:, -1, :]  # take the last hidden
        rnn_out = self.dropout(rnn_out)
        logits = self.out(rnn_out)
        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.float().view(-1, self.num_labels))
        # pred = torch.argmax(logits, dim=-1)

        return {
            'loss': loss,
            'pred': logits,
        }


class LSTMTokenModel(nn.Module):

    def __init__(self, embedding, hidden_size, num_labels=12, dropout=0.5, num_layers=1):
        super(LSTMTokenModel, self).__init__()

        self.num_labels = num_labels
        self.hidden_size = hidden_size
        vocab_size, embed_dim = embedding.shape
        self.embed = nn.Embedding(vocab_size, embed_dim, _weight=embedding)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True)

        self.out = nn.Linear(hidden_size * 2, num_labels)

        init_linear(self.out)
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, labels):
        '''
        :param x: batch_size x seq_len
        :param y: batch_size
        :return:
            loss: scale
            pred: batch_size
        '''
        batch_size, seq_len = input_ids.size()
        x = self.dropout(self.embed(input_ids))
        rnn_out, _ = self.lstm(x)
        rnn_out = rnn_out.view(batch_size, seq_len, 2 * self.hidden_size)
        # rnn_out = rnn_out.view(batch_size, seq_len, 2, self.hidden_size)
        # rnn_out = torch.cat([rnn_out[:, -1, 0, :], rnn_out[:, 0, 1, :]], dim=-1)
        # rnn_out = rnn_out[:, -1, :]  # take the last hidden
        rnn_out = self.dropout(rnn_out)
        logits = self.out(rnn_out)
        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.float().view(-1, self.num_labels))
        # pred = torch.argmax(logits, dim=-1)

        return {
            'loss': loss,
            'pred': logits,
        }



class LinearMModel(nn.Module):

    def __init__(self, embedding, num_labels):
        super(LinearMModel, self).__init__()
        vocab_size, embed_dim = embedding.shape
        self.num_labels = num_labels
        self.embed = nn.Embedding(vocab_size, embed_dim, _weight=embedding)
        self.linear = nn.Linear(embed_dim, num_labels)
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, labels):
        pooled_repr = self.embed(input_ids).mean(dim=1)
        logits = self.linear(pooled_repr)
        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.float().view(-1, self.num_labels))
        return {
            'loss': loss,
            'pred': logits,
        }
