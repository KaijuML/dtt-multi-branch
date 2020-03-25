import torch
import torch.nn as nn
from torch.nn.functional import relu


class PairwiseBilinear(nn.Module):
    """
    A bilinear module that deals with broadcasting for efficient memory usage.
    Input: tensors of sizes (N x L1 x D1) and (N x L2 x D2)
    Output: tensor of size (N x L1 x L2 x O)
    """

    def __init__(self, input1_size, input2_size, output_size, bias=True):
        super().__init__()

        self.input1_size = input1_size
        self.input2_size = input2_size
        self.output_size = output_size

        self.weight = nn.Parameter(torch.randn(input1_size, input2_size, output_size), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(output_size), requires_grad=True) if bias else 0

    def forward(self, input1, input2):
        input1_size = list(input1.size())
        input2_size = list(input2.size())
        output_size = [input1_size[0], input1_size[1], input2_size[1], self.output_size]

        # ((N x L1) x D1) * (D1 x (D2 x O)) -> (N x L1) x (D2 x O)
        intermediate = torch.mm(input1.view(-1, input1_size[-1]),
                                self.weight.view(-1, self.input2_size * self.output_size))
        # (N x L2 x D2) -> (N x D2 x L2)
        input2 = input2.transpose(1, 2)
        # (N x (L1 x O) x D2) * (N x D2 x L2) -> (N x (L1 x O) x L2)
        output = intermediate.view(input1_size[0], input1_size[1] * self.output_size, input2_size[2]).bmm(input2)
        # (N x (L1 x O) x L2) -> (N x L1 x L2 x O)
        output = output.view(input1_size[0], input1_size[1], self.output_size, input2_size[1]).transpose(2, 3)

        return output


class BiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, output_size):
        super().__init__()
        self.W_bilin = nn.Bilinear(input1_size + 1, input2_size + 1, output_size)

        self.W_bilin.weight.data.zero_()
        self.W_bilin.bias.data.zero_()

    def forward(self, input1, input2):
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], len(input1.size()) - 1)
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], len(input2.size()) - 1)
        return self.W_bilin(input1, input2)


class PairwiseBiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, output_size):
        super().__init__()
        self.W_bilin = PairwiseBilinear(input1_size + 1, input2_size + 1, output_size)

        self.W_bilin.weight.data.zero_()
        self.W_bilin.bias.data.zero_()

    def forward(self, input1, input2):
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], len(input1.size()) - 1)
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], len(input2.size()) - 1)
        return self.W_bilin(input1, input2)


class DeepBiaffineScorer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_func=relu, dropout=0,
                 pairwise=True):
        super().__init__()
        self.W1 = nn.Linear(input_size, hidden_size)
        self.W2 = nn.Linear(input_size, hidden_size)
        self.hidden_func = hidden_func
        if pairwise:
            self.scorer = PairwiseBiaffineScorer(hidden_size, hidden_size, output_size)
        else:
            self.scorer = BiaffineScorer(hidden_size, hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.scorer(self.dropout(self.hidden_func(self.W1(x))),
                           self.dropout(self.hidden_func(self.W2(x))))
