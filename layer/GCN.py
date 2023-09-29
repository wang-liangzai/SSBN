import torch.nn as nn
import torch.nn.functional as F
import torch
class GCN(nn.Module):

    def __init__(self, emb_dim=768, num_layers=1, gcn_dropout=0.7):             #此处dropout可以增大
        super(GCN, self).__init__()
        self.layers = num_layers
        self.emb_dim = emb_dim
        self.out_dim = emb_dim
        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.emb_dim if layer == 0 else self.out_dim
            self.W.append(nn.Linear(input_dim, input_dim))
        self.gcn_drop = nn.Dropout(gcn_dropout)


    def forward(self, adj, inputs):
        # gcn layer

        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        for l in range(self.layers):
            Ax = adj.bmm(inputs)  # 1,100,768
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](inputs)  # self loop
            AxW = AxW.cuda() / denom
            gAxW = F.relu(AxW)
            inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
        return inputs, mask

    #            print('Ax.is_cuda:', Ax.is_cuda)
    #            print(Ax)
    #            print(self.W[0])
    #            print(self.W[0](Ax.cpu()))