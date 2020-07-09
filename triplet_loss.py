import torch

# loss function
def triplet_loss(anc, pos, neg, margin=1):
    ap_dist = torch.norm(anc - pos, dim=1)
    an_dist = torch.norm(anc - neg, dim=1)
    relu = torch.nn.ReLU()

    losses = relu(ap_dist - an_dist + margin)
    return torch.sum(losses)


# return the total count of valid triplets in a batch
def total_triplets_valid(anc, pos, neg, margin=1):
    ap_dist = torch.norm(anc - pos, dim=1)
    an_dist = torch.norm(anc - neg, dim=1)
    bits = ap_dist < an_dist
    return torch.sum(bits)


# return the indices of invalid triplets in a batch
def invalid_triplet_indices(anc, pos, neg, margin=1):
    ap_dist = torch.norm(anc - pos, dim=1)
    an_dist = torch.norm(anc - neg, dim=1)
    bits = ap_dist > an_dist
    return bits.nonzero()