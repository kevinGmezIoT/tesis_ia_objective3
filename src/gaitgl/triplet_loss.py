import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, batch_size, margin):
        super().__init__()
        self.P, self.M = batch_size
        self.margin = margin

    def forward(self, feature, label):
        n, m, d = feature.shape
        dist = self.pairwise_distance(feature)
        dist = dist.view(-1)

        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2)).bool().view(-1)
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2)).bool().view(-1)

        hp = torch.masked_select(dist, hp_mask)
        hn = torch.masked_select(dist, hn_mask)

        hard_pos = torch.max(hp.view(n, m, -1), 2)[0]
        hard_neg = torch.min(hn.view(n, m, -1), 2)[0]

        loss = F.relu(self.margin + hard_pos - hard_neg)
        return loss.mean()

    def pairwise_distance(self, x):
        x2 = (x**2).sum(2)
        dist2 = x2.unsqueeze(2) + x2.unsqueeze(1) - 2*torch.matmul(x, x.transpose(1,2))
        return torch.sqrt(F.relu(dist2) + 1e-12)
