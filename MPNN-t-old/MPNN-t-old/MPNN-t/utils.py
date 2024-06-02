import json
import torch
import numpy as np
import torch.nn as nn
import logging
import math
import torch.nn.functional as F


def euclidean_distance(x, y):
    """This is the squared Euclidean distance."""
    return torch.sum((x - y) ** 2, dim=-1)


def approximate_hamming_similarity(x, y):
    """Approximate Hamming similarity."""
    return torch.mean(torch.tanh(x) * torch.tanh(y), dim=1)


def pairwise_loss(x, y, labels, loss_type='margin', margin=1.0):
    """Compute pairwise loss.

    Args:
      x: [N, D] float tensor, representations for N examples.
      y: [N, D] float tensor, representations for another N examples.
      labels: [N] int tensor, with values in -1 or +1.  labels[i] = +1 if x[i]
        and y[i] are similar, and -1 otherwise.
      loss_type: margin or hamming.
      margin: float scalar, margin for the margin loss.

    Returns:
      loss: [N] float tensor.  Loss for each pair of representations.
    """

    labels = labels.float()

    if loss_type == 'margin':
        return torch.relu(margin - labels * (1 - euclidean_distance(x, y)))
    elif loss_type == 'hamming':
        return 0.25 * (labels - approximate_hamming_similarity(x, y)) ** 2
    else:
        raise ValueError('Unknown loss_type %s' % loss_type)


def triplet_loss(x, y, z, loss_type='bpr', margin=20): #计算loss
    """Compute triplet loss.

    This function computes loss on a triplet of inputs (x, y, z).  A similarity or
    distance value is computed for each pair of (x, y) and (x, z).  Since the
    representations for x can be different in the two pairs (like our matching
    model) we distinguish the two x representations by x_1 and x_2.

    Args:
      x_1: [N, D] float tensor.
      y: [N, D] float tensor.
      x_2: [N, D] float tensor.
      z: [N, D] float tensor.
      loss_type: margin or hamming.
      margin: float scalar, margin for the margin loss.

    Returns:
      loss: [N] float tensor.  Loss for each pair of representations.
    """
    if loss_type == 'margin':
        #print(f'loss计算方式margin') euclidean_distance计算欧氏距离
        return torch.relu(margin +
                          euclidean_distance(x, y) -
                          euclidean_distance(x, z))
    elif loss_type == 'hamming':
        print(f'loss计算方式hamming')
        return 0.125 * ((approximate_hamming_similarity(x, y) - 1) ** 2 +
                        (approximate_hamming_similarity(x, z) + 1) ** 2)
    elif loss_type == 'bpr':
        print(f'loss计算方式bpr')
        pos = euclidean_distance(x, y)
        neg = euclidean_distance(x, z)
        rst = pos-neg
        return torch.sum(torch.log(1 + torch.exp(-rst)))
    else:
        raise ValueError('Unknown loss_type %s' % loss_type)


# similarity calculation functions

def euclidean_distance(x, y):
    """This is the squared Euclidean distance."""
    return torch.sum((x - y) ** 2, dim=-1)
    
def exact_hamming_similarity(x, y):
    """Compute the binary Hamming similarity."""
    match = ((x > 0) * (y > 0)).float()
    return torch.mean(match, dim=1)


def compute_similarity(args, x, y):   #计算距离
    """Compute the distance between x and y vectors.

    The distance will be computed based on the training loss type.

    Args:
      config: a config dict.
      x: [n_examples, feature_dim] float tensor.
      y: [n_examples, feature_dim] float tensor.

    Returns:
      dist: [n_examples] float tensor.

    Raises:
      ValueError: if loss type is not supported.
    """
    if args.sim_fun == 'margin':
        # similarity is negative distance
        return -euclidean_distance(x, y)
    elif args.sim_fun == 'hamming':
        return exact_hamming_similarity(x, y)
    else:
        raise ValueError('Unknown loss type')

def compute_metrics(args, sim_pos, sim_neg):   #计算指标
    data_len = sim_pos.shape[0]

    
    # acc
    acc = np.mean([min(sim_pos[i] > sim_neg[i]) for i in range(data_len)])

    rank_num = np.array([(sim_pos[i] > sim_neg[i]).sum() for i in range(data_len)])
    # AUC
    auc = np.mean([rank_num[i].astype(float) / sim_neg.shape[1] for i in range(data_len)])
    # hr@k
    hrk_per_trip = [(sim_neg.shape[1] - rank_num[i].astype(float)) < args.hrk for i in range(data_len)]
    hr_k = np.mean(hrk_per_trip)
    # ndcg@k
    ndcg_list = []
    for i in range(data_len):
        if hrk_per_trip[i]:
            ndcg_per_trip = math.log(2) / math.log(sim_neg.shape[1] - rank_num[i] + 2)
            ndcg_list.append(ndcg_per_trip)
        else:
            ndcg_per_trip = 0
            ndcg_list.append(ndcg_per_trip)
    ndcg_k = np.mean(ndcg_list)

    return acc, auc, hr_k, ndcg_k


def compute_metrics_twoclass(args, sim_pos, sim_neg):  # 计算指标
    y_pred = [(sim_pos[i] >= sim_neg[i]) for i in range(len(sim_pos))]

    sorted_numbers = sorted(sim_neg,reverse=True)          #负样本距离排序 小----大
    print(f'长度：{len(sorted_numbers)}')

    #  pos:[1,1,2,3]    [0.8,1.2,0.7,2]
    # 将模型输出与阈值进行比较，将其划分为二进制预测值
    y_pred_binary = [(sim_pos[i] >= np.mean(sim_neg)) for i in range(len(sim_pos))]       #平均值
    #y_pred_binary = [(sim_pos[i] >= min(sim_neg)) for i in range(len(sim_pos))]      #最小值
    #y_pred_binary = [(sim_pos[i] >= sorted_numbers[65]) for i in range(len(sim_pos))]             #最大值开始寻找


    # 计算 TP、TN、FP 和 FN
    TP = sum((y_pred_binary[i] == 1) and (y_pred[i] == 1) for i in range(len(y_pred)))
    TN = sum((y_pred_binary[i] == 0) and (y_pred[i] == 0) for i in range(len(y_pred)))
    FP = sum((y_pred_binary[i] == 1) and (y_pred[i] == 0) for i in range(len(y_pred)))
    FN = sum((y_pred_binary[i] == 0) and (y_pred[i] == 1) for i in range(len(y_pred)))

    return TP, TN, FP, FN


def unsorted_segment_sum(data, segment_ids, num_segments):   #沿段求和
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """

    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    # Encourage to use the below code when a deterministic result is
    # needed (reproducibility). However, the code below is with low efficiency.

    # tensor = torch.zeros(num_segments, data.shape[1]).cuda()
    # for index in range(num_segments):
    #     tensor[index, :] = torch.sum(data[segment_ids == index, :], dim=0)
    # return tensor

    if len(segment_ids.shape) == 1:
        #s = torch.prod(torch.tensor(data.shape[1:])).long()    # cpu version
        s = torch.prod(torch.tensor(data.shape[1:])).long().cuda()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    #tensor = torch.zeros(*shape).scatter_add(0, segment_ids, data)   # cpu version
    # tensor = torch.zeros(*shape).cuda().scatter_add(0, segment_ids, data)
    tensor = torch.zeros(*shape).cuda().scatter_add(0, segment_ids, data)
    tensor = tensor.type(data.dtype)
    return tensor

"""tool functions"""

def judge_pre_stop(prelist):
    _sort = sorted(prelist)
    _sort.reverse()
    if _sort == prelist:
        return True
    else:
        return False

def new_contract(fun_node_num_per_g,function_ndoe_non):
    graph_total = []
    max_length = max(fun_node_num_per_g)
    for j in range(len(fun_node_num_per_g)):
        index = fun_node_num_per_g[j]
        small_contract = function_ndoe_non[:index].clone()

        temp_ll=small_contract

        if temp_ll.shape[0] != max_length:
            padding_rows = max_length - temp_ll.shape[0]
            padded_tensor = F.pad(temp_ll, (0, 0, 0, padding_rows), "constant", 999)

        else:
            padded_tensor=temp_ll

        graph_total.append(padded_tensor)

    stacked_tensor = torch.stack(graph_total)  # 输入BiLSTM构造完毕
    return stacked_tensor

