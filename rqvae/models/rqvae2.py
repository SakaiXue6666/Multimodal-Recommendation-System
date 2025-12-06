import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 # num_emb_list=[256,256,256,256],
                 num_emb_list=None,
                 e_dim=64,
                 # layers=[512,256,128],
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 # sk_epsilons=[0,0,0.003,0.01]],
                 sk_epsilons=None,
                 sk_iters=100,
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    def forward(self, x, use_sk=True):
        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x,use_sk=use_sk)
        out = self.decoder(x_q)

        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quant_loss, xs=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon


class MultiModalFusion(nn.Module):
    """
    输入：
        - num: [B, num_dim]
        - cls: [B, cls_dim]
        - text: [B, 512]
        - img: [B, 512]
    输出：
        - fused embedding: [B, hidden_dim]
    """
    def __init__(self, num_dim, cls_dim, hidden_dim=256):
        super().__init__()
        self.cls_emb = nn.ModuleDict({
            k: nn.Embedding(cls_dim[k], hidden_dim) for k in cls_dim
        })
        cls_total_dim = len(cls_dim) * hidden_dim
        num_total_dim = sum(num_dim[k] for k in num_dim)

        self.num_cls_proj = nn.Linear(cls_total_dim + num_total_dim, hidden_dim)
        self.text_proj = nn.Linear(512, hidden_dim)
        self.img_proj = nn.Linear(512, hidden_dim)

        # 最终融合层
        self.fuse_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, batch):
        text = batch["text_emb"]
        img = batch["image_emb"]
        num = batch["num"]
        cls = batch["cls"]

        # ===== 数值模态 =====
        num_vecs = [v for v in num.values()]
        num_cat = torch.cat(num_vecs, dim=-1)

        # ===== 类别模态 =====
        cls_vecs = []
        for k, v in cls.items():
            emb = self.cls_emb[k](v)
            if emb.dim() == 3:  # e.g. [B, L, D]
                emb = emb.mean(dim=1)
            cls_vecs.append(emb)
        cls_cat = torch.cat(cls_vecs, dim=-1)

        # ===== 融合 =====
        num_cls_emb = self.num_cls_proj(torch.cat([num_cat, cls_cat], dim=-1))
        text_emb = self.text_proj(text)
        img_emb = self.img_proj(img)

        fused = torch.cat([num_cls_emb, text_emb, img_emb], dim=-1)
        fused = self.fuse_mlp(fused)
        return fused


class MultiModalRQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 # num_emb_list=[256,256,256,256],
                 num_emb_list=None,
                 e_dim=64,
                 # layers=[512,256,128],
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 # sk_epsilons=[0,0,0.003,0.01]],
                 sk_epsilons=None,
                 sk_iters=100,
                 num_dim=None,  # e.g. {'price': 1}
                 cls_dim=None,  # e.g. {'brand': n_brand, 'categories': n_cat}
        ):
        super(MultiModalRQVAE, self).__init__()
        self.fusion = MultiModalFusion(num_dim, cls_dim, in_dim)
        self.rqvae = RQVAE(in_dim, num_emb_list, e_dim, layers, dropout_prob, bn, loss_type, quant_loss_weight, beta, kmeans_init, kmeans_iters, sk_epsilons, sk_iters)

    def forward(self, batch, use_sk=True):
        """
        batch: Dataset 输出的结构:
        {
            'text_emb': [B, 512],
            'image_emb': [B, 512],
            'num': {'price': [B, 1]},
            'cls': {'brand': [B, 1], 'categories': [B, L]}
        }
        """
        fused = self.fusion(batch)
        out, rq_loss, indices = self.rqvae(fused, use_sk=use_sk)

        self.fused = fused
        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        fused = self.fusion(xs)
        return self.rqvae.get_indices(fused, use_sk=use_sk)

    def compute_loss(self, out, quant_loss, xs=None):
        return self.rqvae.compute_loss(out, quant_loss, self.fused)