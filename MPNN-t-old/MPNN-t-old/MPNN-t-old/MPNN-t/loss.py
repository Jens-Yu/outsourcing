import torch


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


def triplet_loss(x, y, z, loss_type='bpr', margin=20):
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
        return torch.relu(margin +
                          euclidean_distance(x, y) -
                          euclidean_distance(x, z))
    elif loss_type == 'hamming':
        return 0.125 * ((approximate_hamming_similarity(x, y) - 1) ** 2 +
                        (approximate_hamming_similarity(x, z) + 1) ** 2)
    elif loss_type == 'bpr':
        pos = euclidean_distance(x, y)
        neg = euclidean_distance(x, z)
        rst = pos-neg
        return torch.sum(torch.log(1 + torch.exp(-rst)))
    else:
        raise ValueError('Unknown loss_type %s' % loss_type)
