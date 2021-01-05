import torch


def bidaf_loss(p_start, p_end, y_start, y_end):
    bs = p_start.size(0)
    # Retrieve the probability of the correct start and end indexes
    p_true_start = torch.gather(p_start, 1, y_start.unsqueeze(1))  # equivalent to p_start[y_start]

    p_true_end = torch.gather(p_end, 1, y_end.unsqueeze(1))  # equivalent to p_end[y_end]
    # Sum logarithms of start and end probabilities
    sum_vec = torch.log(p_true_start) + torch.log(p_true_end)
    # Sum over batch dimension
    return - sum_vec.sum() / bs
