import torch
from torch.nn.parameter import Parameter
from torch import nn
import torch.nn.functional as F
from geffnet.conv2d_layers import Conv2dSame


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class EfficinetNet(nn.Module):
    def __init__(self, name='efficientnet_b0', pretrained='imagenet', out_features=81313, dropout=0.5, feature_dim=512):
        super().__init__()

        print("Model's name:", name)

        self.model = torch.hub.load('rwightman/gen-efficientnet-pytorch', name,
                                    pretrained=(pretrained == 'imagenet'))

        self.model.conv_stem = Conv2dSame(4, self.model.conv_stem.out_channels, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.last_linear = nn.Linear(in_features=self.model.classifier.in_features, out_features=out_features)
        self.last_linear2 = nn.Linear(in_features=self.model.classifier.in_features, out_features=out_features)
        self.pool = GeM()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cnt):
        # print("on model forward")
        # print("cnt", cnt)

        x = self.model.features(x)
        # print("x", x.shape)
        
        pooled = nn.Flatten()(self.pool(x))
        # print("pooled", pooled.shape)
        
        # separa os vetores de células por imagem
        pooled_split = torch.split(pooled, cnt.tolist())  # lista de tensores (n_células_i, features)
        # print("pooled_split", len(pooled_split), pooled_split[0].shape)

        # aplica max pooling por imagem (dim=0: entre as células)
        pooled_per_img = torch.stack([p.max(0)[0] for p in pooled_split])
        # print("pooled_per_img", pooled_per_img.shape)  # (batch_size, features)

        return self.last_linear(self.dropout(pooled)), self.last_linear2(self.dropout(pooled_per_img))
