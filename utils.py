import torch
import math
import random
import numpy as np
from numpy.linalg import norm
from fastNLP import cache_results
from sklearn.cluster import KMeans
from fastNLP.core.field import FieldArray
from transformers import BertForSequenceClassification


def save_for_elue(seq_lens, num_layers):
    assert len(seq_lens) == len(num_layers)
    with open('tmp.txt', 'w') as fout:
        for i in range(len(seq_lens)):
            fout.write("{}\t{}\n".format(seq_lens[i], num_layers[i]))
    return


def clustering(data_bundle, model):
    train_data, dev_data, test_data = data_bundle.get_dataset('train'), data_bundle.get_dataset(
        'dev'), data_bundle.get_dataset('test')
    cluster2layer = {}
    train_dist, dev_dist, test_dist = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    add_marker = True

    def insert_marker(x, a):
        return [x[0]] + [int(a + 10)] + x[1:]

    # fit train data
    clustered_train_data = []
    for data in train_data:
        input_ids = torch.tensor(data['input_ids']).unsqueeze(0)
        embedding = model.elasticbert.embeddings.word_embeddings(input_ids).squeeze().mean(dim=0)
        clustered_train_data.append(embedding)
    clustered_train_data = torch.stack(clustered_train_data, dim=0).detach().numpy()
    print(clustered_train_data.shape)
    km_cluster = KMeans(n_clusters=3, max_iter=300, n_init=40, init='k-means++', n_jobs=-1)
    exit_layers = km_cluster.fit_predict(clustered_train_data)
    for i in range(len(exit_layers)):
        train_dist[exit_layers[i]] += 1
    print(train_dist)
    sort_index = np.argsort(train_dist)
    assign_layer = 5
    for cluster_id in sort_index:
        cluster2layer[cluster_id] = assign_layer
        assign_layer -= 2
    print(cluster2layer)
    for i in range(len(exit_layers)):
        exit_layers[i] = cluster2layer[exit_layers[i]]
    field = FieldArray(name='exit_layers', content=exit_layers)
    train_data.add_fieldarray('exit_layers', field)
    train_data.set_input('exit_layers')
    # modify input_ids
    if add_marker:
        input_ids = []
        for i, x in enumerate(train_data['input_ids']):
            input_ids.append(insert_marker(x, exit_layers[i]))
        print(input_ids[:3])
        # for i in input_ids[0]:
        #     print(type(i))
        field = FieldArray(name='input_ids', content=input_ids)
        train_data.add_fieldarray('input_ids', field)
        train_data.set_input('input_ids')
        train_data.apply_field(func=len, field_name="input_ids", new_field_name="seq_len")

    # for dev and test, the exit_layers are the nearest class centers
    # predict dev data
    clustered_dev_data = []
    for data in dev_data:
        input_ids = torch.tensor(data['input_ids']).unsqueeze(0)
        embedding = model.elasticbert.embeddings.word_embeddings(input_ids).squeeze().mean(dim=0)
        clustered_dev_data.append(embedding)
    clustered_dev_data = torch.stack(clustered_dev_data, dim=0).detach().numpy()
    print(clustered_dev_data.shape)
    exit_layers = km_cluster.predict(clustered_dev_data)
    for i in range(len(exit_layers)):
        dev_dist[exit_layers[i]] += 1
        exit_layers[i] = cluster2layer[exit_layers[i]]
    print(dev_dist)
    field = FieldArray(name='exit_layers', content=exit_layers)
    dev_data.add_fieldarray('exit_layers', field)
    dev_data.set_input('exit_layers')
    # modify input_ids
    if add_marker:
        input_ids = []
        for i, x in enumerate(dev_data['input_ids']):
            input_ids.append(insert_marker(x, exit_layers[i]))
        print(input_ids[:3])
        field = FieldArray(name='input_ids', content=input_ids)
        dev_data.add_fieldarray('input_ids', field)
        dev_data.set_input('input_ids')
        dev_data.apply_field(func=len, field_name="input_ids", new_field_name="seq_len")

    # predict test data
    clustered_test_data = []
    for data in test_data:
        input_ids = torch.tensor(data['input_ids']).unsqueeze(0)
        embedding = model.elasticbert.embeddings.word_embeddings(input_ids).squeeze().mean(dim=0)
        clustered_test_data.append(embedding)
    clustered_test_data = torch.stack(clustered_test_data, dim=0).detach().numpy()
    print(clustered_test_data.shape)
    exit_layers = km_cluster.predict(clustered_test_data)
    for i in range(len(exit_layers)):
        test_dist[exit_layers[i]] += 1
        exit_layers[i] = cluster2layer[exit_layers[i]]
    print(test_dist)
    field = FieldArray(name='exit_layers', content=exit_layers)
    test_data.add_fieldarray('exit_layers', field)
    test_data.set_input('exit_layers')
    # modify input_ids
    if add_marker:
        input_ids = []
        for i, x in enumerate(test_data['input_ids']):
            input_ids.append(insert_marker(x, exit_layers[i]))
        print(input_ids[:3])
        field = FieldArray(name='input_ids', content=input_ids)
        test_data.add_fieldarray('input_ids', field)
        test_data.set_input('input_ids')
        test_data.apply_field(func=len, field_name="input_ids", new_field_name="seq_len")
    # save_for_elue(test_data.get_field('seq_len'), test_data.get_field('exit_layers'))
    return data_bundle


# def hash_tokens(data_bundle, tokenizer):
#     train_data, dev_data, test_data = data_bundle.get_dataset('train'), data_bundle.get_dataset(
#         'dev'), data_bundle.get_dataset('test')
#     token_freq = {}
#     total_tokens = 0
#     special_tokens = tokenizer.all_special_ids  # UNK, SEP, PAD, CLS, MASK
#     for data in train_data:
#         input_ids = data['input_ids']
#         for input_id in input_ids:
#             if input_id not in special_tokens:
#                 if input_id in token_freq:
#                     token_freq[input_id] += 1
#                     total_tokens += 1
#                 else:
#                     token_freq[input_id] = 1
#                     total_tokens += 1
#
#     for data in dev_data:
#         input_ids = data['input_ids']
#         for input_id in input_ids:
#             if input_id not in special_tokens:
#                 if input_id in token_freq:
#                     token_freq[input_id] += 1
#                     total_tokens += 1
#                 else:
#                     token_freq[input_id] = 1
#                     total_tokens += 1
#
#     vocab_size = len(token_freq)
#     print("vocab size: {}".format(vocab_size))
#     sorted_token_freq = sorted(token_freq.items(), key = lambda x:x[1], reverse=True)
#     print("top token frequency: {}".format(sorted_token_freq[:10]))
#     for item in sorted_token_freq[:10]:
#         print("{}: {}".format(tokenizer.decode([item[0]]), item[1]))
#     token2bucket = {}
#     bucket_size = total_tokens // 6  # we have 10 buckets
#     print("total tokens: {}, bucket_cap: {}".format(total_tokens, bucket_size))
#     MIN_LAYER = 2
#     MAX_LAYER = 7
#     bucket_id = MIN_LAYER  # 2~7 corresponds to layer 1~6
#     num_elements_in_bucket, num_tokens = 0, 0
#     for item in sorted_token_freq:
#         token_id, token_freq = item
#         token2bucket[token_id] = bucket_id
#         num_elements_in_bucket += 1
#         num_tokens += token_freq
#         if num_tokens >= bucket_size and bucket_id < MAX_LAYER:
#             print("num elements in bucket {}: {}".format(bucket_id, num_elements_in_bucket))
#             num_elements_in_bucket = 0
#             num_tokens = 0
#             bucket_id += 1
#     print("num elements in bucket {}: {}".format(bucket_id, num_elements_in_bucket))
#     for token_id in special_tokens:
#         token2bucket[token_id] = MAX_LAYER  # always put special tokens in the last layer
#
#     # add exit_layers in train_data
#     exit_layers = []
#     for data in train_data:
#         input_ids = data['input_ids']
#         tmp = []
#         for input_id in input_ids:
#             tmp.append(token2bucket[input_id])
#         exit_layers.append(tmp)
#     field = FieldArray(name='exit_layers', content=exit_layers)
#     train_data.add_fieldarray('exit_layers', field)
#     train_data.set_input('exit_layers')
#
#     # add exit_layers in dev_data
#     exit_layers = []
#     for data in dev_data:
#         input_ids = data['input_ids']
#         tmp = []
#         for input_id in input_ids:
#             tmp.append(token2bucket[input_id])
#         exit_layers.append(tmp)
#     field = FieldArray(name='exit_layers', content=exit_layers)
#     dev_data.add_fieldarray('exit_layers', field)
#     dev_data.set_input('exit_layers')
#
#     # add exit_layers in test_data
#     num_unseen = 0
#     exit_layers = []
#     for data in test_data:
#         input_ids = data['input_ids']
#         tmp = []
#         for input_id in input_ids:
#             if input_id in token2bucket:
#                 tmp.append(token2bucket[input_id])
#             else:
#                 tmp.append(13)
#                 num_unseen += 1
#         exit_layers.append(tmp)
#     field = FieldArray(name='exit_layers', content=exit_layers)
#     test_data.add_fieldarray('exit_layers', field)
#     test_data.set_input('exit_layers')
#     print('num unseen tokens in test set: {}'.format(num_unseen))
#     return data_bundle


def hash_tokens(data_bundle, tokenizer, max_layer, num_buckets):
    train_data, dev_data, test_data = data_bundle.get_dataset('train'), data_bundle.get_dataset(
        'dev'), data_bundle.get_dataset('test')
    token_freq = {}
    special_tokens = tokenizer.all_special_ids  # UNK, SEP, PAD, CLS, MASK
    for data in train_data:
        input_ids = data['input_ids']
        for input_id in input_ids:
            if input_id not in special_tokens:
                if input_id in token_freq:
                    token_freq[input_id] += 1
                else:
                    token_freq[input_id] = 1

    for data in dev_data:
        input_ids = data['input_ids']
        for input_id in input_ids:
            if input_id not in special_tokens:
                if input_id in token_freq:
                    token_freq[input_id] += 1
                else:
                    token_freq[input_id] = 1

    vocab_size = len(token_freq)
    print("vocab size: {}".format(vocab_size))
    sorted_token_freq = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
    print("top token frequency: {}".format(sorted_token_freq[:10]))
    for item in sorted_token_freq[:10]:
        print("{}: {}".format(tokenizer.decode([item[0]]), item[1]))
    # print("{}: {}".format(6000, sorted_token_freq[6000][1]))
    # print("{}: {}".format(8000, sorted_token_freq[8000][1]))
    # print("{}: {}".format(10000, sorted_token_freq[10000][1]))
    token2bucket = {}
    # bucket_size = vocab_size // 10  # we have 10 buckets
    # bucket_id = 4  # 4~13 corresponds to layer 3~12
    MIN_LAYER = 2
    MAX_LAYER = max_layer
    print("MIN_LAYER: {}".format(MIN_LAYER))
    print("MAX_LAYER: {}".format(max_layer))
    # NUM_BUCKETS = 1 + 2 + 4 + 8 + 16 + 32
    # NUM_BUCKETS = MAX_LAYER - MIN_LAYER + 1
    NUM_BUCKETS = num_buckets
    interval = (MAX_LAYER - 1) // num_buckets
    bucket_size = vocab_size // NUM_BUCKETS
    # NUM_BUCKETS = MAX_LAYER - MIN_LAYER + 1
    # bucket_size = vocab_size // NUM_BUCKETS  # we have 6 buckets
    bucket_id = MIN_LAYER  # 2~7 corresponds to layer 1~6
    num_elements_in_bucket = 0
    for item in sorted_token_freq:
        token_id, token_freq = item
        token2bucket[token_id] = bucket_id
        num_elements_in_bucket += 1
        if num_elements_in_bucket >= bucket_size and bucket_id < MAX_LAYER:
            print("num elements in bucket {}: {}".format(bucket_id, num_elements_in_bucket))
            num_elements_in_bucket = 0
            bucket_id += interval
            # bucket_size *= 2
    print("num elements in bucket {}: {}".format(bucket_id, num_elements_in_bucket))
    for token_id in special_tokens:
        token2bucket[token_id] = MAX_LAYER  # always put special tokens in the last layer

    # add exit_layers in train_data
    exit_layers = []
    for data in train_data:
        input_ids = data['input_ids']
        tmp = []
        for input_id in input_ids:
            tmp.append(token2bucket[input_id])
        exit_layers.append(tmp)
    field = FieldArray(name='exit_layers', content=exit_layers)
    train_data.add_fieldarray('exit_layers', field)
    train_data.set_input('exit_layers')

    # add exit_layers in dev_data
    exit_layers = []
    for data in dev_data:
        input_ids = data['input_ids']
        tmp = []
        for input_id in input_ids:
            tmp.append(token2bucket[input_id])
        exit_layers.append(tmp)
    field = FieldArray(name='exit_layers', content=exit_layers)
    dev_data.add_fieldarray('exit_layers', field)
    dev_data.set_input('exit_layers')

    # add exit_layers in test_data
    num_unseen = 0
    exit_layers = []
    for data in test_data:
        input_ids = data['input_ids']
        tmp = []
        for input_id in input_ids:
            if input_id in token2bucket:
                tmp.append(token2bucket[input_id])
            else:
                tmp.append(MAX_LAYER)
                num_unseen += 1
        exit_layers.append(tmp)
    field = FieldArray(name='exit_layers', content=exit_layers)
    test_data.add_fieldarray('exit_layers', field)
    test_data.set_input('exit_layers')
    print('num unseen tokens in test set: {}'.format(num_unseen))
    return data_bundle


def mi_hash_tokens(data_bundle, tokenizer, max_layer, num_buckets):
    train_data, dev_data, test_data = data_bundle.get_dataset('train'), data_bundle.get_dataset(
        'dev'), data_bundle.get_dataset('test')
    px = {}
    py = {}
    pxy = {}
    special_tokens = tokenizer.all_special_ids  # UNK, SEP, PAD, CLS, MASK
    for data in train_data:
        input_ids = data['input_ids']
        labels = data['labels']
        for input_id in input_ids:
            if input_id not in special_tokens:
                if input_id in px:
                    px[input_id] += 1
                    pxy[input_id][labels] += 1
                else:
                    px[input_id] = 1
                    pxy[input_id] = {0: 0, 1: 0}
        if labels in py:
            py[labels] += 1
        else:
            py[labels] = 1
    # normalize
    total_px = 0
    for _, v in px.items():
        total_px += v
    for k, v in px.items():
        px[k] = v / total_px

    total_py = 0
    for _, v in py.items():
        total_py += v
    for k, v in py.items():
        py[k] = v / total_py

    for k, v in pxy.items():
        v[0] = v[0] / total_px
        v[1] = v[1] / total_px

    token_mi = {}
    for token_id in px.keys():
        # print(pxy[token_id][0] / (px[token_id] * py[0]))
        mi_0 = math.log(pxy[token_id][0] / (px[token_id] * py[0])+1e-6) * pxy[token_id][0]
        mi_1 = math.log(pxy[token_id][1] / (px[token_id] * py[1])+1e-6) * pxy[token_id][1]
        token_mi[token_id] = mi_0 + mi_1

    vocab_size = len(px)
    print("vocab size: {}".format(vocab_size))
    sorted_token_mi = sorted(token_mi.items(), key=lambda x: x[1], reverse=True)
    print("top token MI: {}".format(sorted_token_mi[:10]))
    for item in sorted_token_mi[:10]:
        print("{}: {}".format(tokenizer.decode([item[0]]), item[1]))
    # print("{}: {}".format(6000, sorted_token_freq[6000][1]))
    # print("{}: {}".format(8000, sorted_token_freq[8000][1]))
    # print("{}: {}".format(10000, sorted_token_freq[10000][1]))
    token2bucket = {}
    # bucket_size = vocab_size // 10  # we have 10 buckets
    # bucket_id = 4  # 4~13 corresponds to layer 3~12
    MIN_LAYER = 2
    MAX_LAYER = max_layer
    print("MIN_LAYER: {}".format(MIN_LAYER))
    print("MAX_LAYER: {}".format(max_layer))
    # NUM_BUCKETS = 1 + 2 + 4 + 8 + 16 + 32
    # NUM_BUCKETS = MAX_LAYER - MIN_LAYER + 1
    NUM_BUCKETS = num_buckets
    interval = (MAX_LAYER - 1) // num_buckets
    bucket_size = vocab_size // NUM_BUCKETS
    # NUM_BUCKETS = MAX_LAYER - MIN_LAYER + 1
    # bucket_size = vocab_size // NUM_BUCKETS  # we have 6 buckets
    bucket_id = MIN_LAYER  # 2~7 corresponds to layer 1~6
    num_elements_in_bucket = 0
    for item in sorted_token_mi:
        token_id, token_freq = item
        token2bucket[token_id] = bucket_id
        num_elements_in_bucket += 1
        if num_elements_in_bucket >= bucket_size and bucket_id < MAX_LAYER:
            print("num elements in bucket {}: {}".format(bucket_id, num_elements_in_bucket))
            num_elements_in_bucket = 0
            bucket_id += interval
            # bucket_size *= 2
    print("num elements in bucket {}: {}".format(bucket_id, num_elements_in_bucket))
    for token_id in special_tokens:
        token2bucket[token_id] = MAX_LAYER  # always put special tokens in the last layer

    # add exit_layers in train_data
    exit_layers = []
    for data in train_data:
        input_ids = data['input_ids']
        tmp = []
        for input_id in input_ids:
            tmp.append(token2bucket[input_id])
        exit_layers.append(tmp)
    field = FieldArray(name='exit_layers', content=exit_layers)
    train_data.add_fieldarray('exit_layers', field)
    train_data.set_input('exit_layers')

    # add exit_layers in dev_data
    exit_layers = []
    for data in dev_data:
        input_ids = data['input_ids']
        tmp = []
        for input_id in input_ids:
            if input_id in token2bucket:
                tmp.append(token2bucket[input_id])
            else:
                tmp.append(MAX_LAYER)
        exit_layers.append(tmp)
    field = FieldArray(name='exit_layers', content=exit_layers)
    dev_data.add_fieldarray('exit_layers', field)
    dev_data.set_input('exit_layers')

    # add exit_layers in test_data
    num_unseen = 0
    exit_layers = []
    for data in test_data:
        input_ids = data['input_ids']
        tmp = []
        for input_id in input_ids:
            if input_id in token2bucket:
                tmp.append(token2bucket[input_id])
            else:
                tmp.append(MAX_LAYER)
                num_unseen += 1
        exit_layers.append(tmp)
    field = FieldArray(name='exit_layers', content=exit_layers)
    test_data.add_fieldarray('exit_layers', field)
    test_data.set_input('exit_layers')
    print('num unseen tokens in test set: {}'.format(num_unseen))
    return data_bundle


# @cache_results(_cache_fp=f'./random_hash_mrpc.pt', _refresh=False)
def random_incons_hash_tokens(data_bundle, tokenizer, max_layer, num_buckets):
    train_data, dev_data, test_data = data_bundle.get_dataset('train'), data_bundle.get_dataset(
        'dev'), data_bundle.get_dataset('test')
    special_tokens = tokenizer.all_special_ids  # UNK, SEP, PAD, CLS, MASK
    MIN_LAYER = 2
    MAX_LAYER = max_layer
    NUM_BUCKETS = num_buckets
    interval = (MAX_LAYER - 1) // NUM_BUCKETS

    exit_layers = []
    for data in train_data:
        input_ids = data['input_ids']
        tmp = np.random.randint(0, NUM_BUCKETS, size=len(input_ids))
        tmp = tmp * interval + MIN_LAYER
        exit_layers.append(tmp)
    field = FieldArray(name='exit_layers', content=exit_layers)
    train_data.add_fieldarray('exit_layers', field)
    train_data.set_input('exit_layers')

    exit_layers = []
    for data in dev_data:
        input_ids = data['input_ids']
        tmp = np.random.randint(0, NUM_BUCKETS, size=len(input_ids))
        tmp = tmp * interval + MIN_LAYER
        exit_layers.append(tmp)
    field = FieldArray(name='exit_layers', content=exit_layers)
    dev_data.add_fieldarray('exit_layers', field)
    dev_data.set_input('exit_layers')

    exit_layers = []
    for data in test_data:
        input_ids = data['input_ids']
        tmp = np.random.randint(0, NUM_BUCKETS, size=len(input_ids))
        tmp = tmp * interval + MIN_LAYER
        exit_layers.append(tmp)
    field = FieldArray(name='exit_layers', content=exit_layers)
    test_data.add_fieldarray('exit_layers', field)
    test_data.set_input('exit_layers')
    return data_bundle


# @cache_results(_cache_fp=f'./random_cons_hash_mrpc.pt', _refresh=True)
def random_cons_hash_tokens(data_bundle, tokenizer, max_layer, num_buckets):
    train_data, dev_data, test_data = data_bundle.get_dataset('train'), data_bundle.get_dataset(
        'dev'), data_bundle.get_dataset('test')
    special_tokens = tokenizer.all_special_ids  # UNK, SEP, PAD, CLS, MASK
    MIN_LAYER = 2
    MAX_LAYER = max_layer
    NUM_BUCKETS = num_buckets
    interval = (MAX_LAYER - 1) // NUM_BUCKETS

    token2bucket = {}
    for token_id in range(tokenizer.vocab_size):
        if token_id in special_tokens:
            token2bucket[token_id] = MAX_LAYER
        else:
            tmp = np.random.randint(0, NUM_BUCKETS)
            tmp = tmp * interval + MIN_LAYER
            token2bucket[token_id] = tmp

    exit_layers = []
    for data in train_data:
        input_ids = data['input_ids']
        tmp = []
        for input_id in input_ids:
            tmp.append(token2bucket[input_id])
        exit_layers.append(tmp)
    field = FieldArray(name='exit_layers', content=exit_layers)
    train_data.add_fieldarray('exit_layers', field)
    train_data.set_input('exit_layers')

    exit_layers = []
    for data in dev_data:
        input_ids = data['input_ids']
        tmp = []
        for input_id in input_ids:
            tmp.append(token2bucket[input_id])
        exit_layers.append(tmp)
    field = FieldArray(name='exit_layers', content=exit_layers)
    dev_data.add_fieldarray('exit_layers', field)
    dev_data.set_input('exit_layers')

    exit_layers = []
    for data in test_data:
        input_ids = data['input_ids']
        tmp = []
        for input_id in input_ids:
            tmp.append(token2bucket[input_id])
        exit_layers.append(tmp)
    field = FieldArray(name='exit_layers', content=exit_layers)
    test_data.add_fieldarray('exit_layers', field)
    test_data.set_input('exit_layers')
    return data_bundle


def cluster_hash_tokens(data_bundle, tokenizer, max_layer, num_buckets, model=None):
    # clustering tokens
    if model is None:
        model = BertForSequenceClassification.from_pretrained('fnlp/elasticbert-base')
    MIN_LAYER = 2
    MAX_LAYER = max_layer
    NUM_BUCKETS = num_buckets
    interval = (MAX_LAYER - 1) // NUM_BUCKETS

    word_embeddings = model.get_input_embeddings().weight.detach().numpy()
    km_cluster = KMeans(n_clusters=NUM_BUCKETS, max_iter=300, n_init=40, init='k-means++', n_jobs=-1)
    exit_layers = km_cluster.fit_predict(word_embeddings).tolist()
    # TODO: 需要将聚类id重映射到layer，可以按照norm来排序
    # bucket_norms = [0. for _ in range(MAX_LAYER - MIN_LAYER + 1)]  # 计算每个bucket的norm sum
    cluster_norms = {}  # cluster_id2norm, 计算每个cluster的norm sum
    for i, cluster_id in enumerate(exit_layers):
        tmp_norm = norm(word_embeddings[i])
        if cluster_id in cluster_norms:
            cluster_norms[cluster_id] += tmp_norm
        else:
            cluster_norms[cluster_id] = tmp_norm
    sorted_cluster_norms = sorted(cluster_norms.items(), key=lambda x: x[1], reverse=False)
    print("sorted cluster norms: {}".format(sorted_cluster_norms))
    cluster2bucket = {}
    bucket_id = MIN_LAYER
    for k, v in sorted_cluster_norms:
        cluster2bucket[k] = bucket_id
        bucket_id += interval
    print("cluster2bucket: {}".format(cluster2bucket))
    token2bucket = {}
    bucket_elements = [0 for _ in range(MAX_LAYER)]
    special_tokens = tokenizer.all_special_ids  # UNK, SEP, PAD, CLS, MASK
    for token_id, cluster_id in enumerate(exit_layers):
        if token_id in special_tokens:
            token2bucket[token_id] = MAX_LAYER
            bucket_elements[MAX_LAYER - 1] += 1
        else:
            token2bucket[token_id] = cluster2bucket[cluster_id]
            bucket_elements[cluster2bucket[cluster_id] - 1] += 1
    print(bucket_elements)

    train_data, dev_data, test_data = data_bundle.get_dataset('train'), data_bundle.get_dataset(
        'dev'), data_bundle.get_dataset('test')

    # add exit_layers in train_data
    exit_layers = []
    for data in train_data:
        input_ids = data['input_ids']
        tmp = []
        for input_id in input_ids:
            tmp.append(token2bucket[input_id])
        exit_layers.append(tmp)
    field = FieldArray(name='exit_layers', content=exit_layers)
    train_data.add_fieldarray('exit_layers', field)
    train_data.set_input('exit_layers')

    # add exit_layers in dev_data
    exit_layers = []
    for data in dev_data:
        input_ids = data['input_ids']
        tmp = []
        for input_id in input_ids:
            tmp.append(token2bucket[input_id])
        exit_layers.append(tmp)
    field = FieldArray(name='exit_layers', content=exit_layers)
    dev_data.add_fieldarray('exit_layers', field)
    dev_data.set_input('exit_layers')

    # add exit_layers in test_data
    num_unseen = 0
    exit_layers = []
    for data in test_data:
        input_ids = data['input_ids']
        tmp = []
        for input_id in input_ids:
            if input_id in token2bucket:
                tmp.append(token2bucket[input_id])
            else:
                tmp.append(MAX_LAYER)
                num_unseen += 1
        exit_layers.append(tmp)
    field = FieldArray(name='exit_layers', content=exit_layers)
    test_data.add_fieldarray('exit_layers', field)
    test_data.set_input('exit_layers')
    print('num unseen tokens in test set: {}'.format(num_unseen))
    return data_bundle
