import torch


def bidaf_loss(p_start: torch.FloatTensor, p_end: torch.FloatTensor,
               y_start: torch.LongTensor, y_end: torch.LongTensor) -> torch.FloatTensor:
    bs = p_start.size(0)
    # Retrieve the probability of the correct start and end indexes
    p_true_start = torch.gather(p_start, 1, y_start.unsqueeze(1))  # equivalent to p_start[y_start]
    p_true_end = torch.gather(p_end, 1, y_end.unsqueeze(1))  # equivalent to p_end[y_end]
    # Sum logarithms of start and end probabilities (adding eps to avoid NaN log)
    eps = 1e-7
    sum_vec = torch.log(torch.add(p_true_start, eps)) + torch.log(torch.add(p_true_end, eps))
    # Sum over batch dimension
    return - sum_vec.sum() / bs
