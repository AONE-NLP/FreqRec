import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward, MultiHeadAttention
import pandas as pd
import numpy as np
from scipy.signal import czt

class FreqRecModel(SequentialRecModel):
    def __init__(self, args):
        super(FreqRecModel, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = FilterEncoder(args)
        self.apply(self.init_weights)
        self.fft_loss_type = args.fft_loss_type

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        #input_ids 256 50
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)  # 256 50 64
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output, sequence_emb

    @property
    def loss_fn(self):
        if self.fft_loss_type == 'l1':
            return F.l1_loss
        elif self.fft_loss_type == 'l2':
            return F.mse_loss
        elif self.fft_loss_type == 'SmoothL1Loss':
            return F.smooth_l1_loss
        elif self.fft_loss_type == 'mix_loss':
            return self.mix_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def mix_loss(self, pred, true,reduction=None):
        """混合L1和L2损失"""
        l1 = F.l1_loss(pred, true)
        l2 = F.mse_loss(pred, true)
        return 0.5 * l1 + 0.5 * l2

    def fft_loss(self, model_out, target):
        fft1 = torch.fft.fft(model_out.transpose(1, 2), norm='forward')
        fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
        fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
        fourier_loss = (self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none')
                        + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none'))
        return fourier_loss

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output, target = self.forward(input_ids)  # 256 50 64

        fourier_loss = self.fft_loss(seq_output, target)
        fourier_loss = torch.mean(fourier_loss)

        seq_output = seq_output[:, -1, :]  # 256 64
        item_emb = self.item_embeddings.weight  # 12102 64
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))  # 256 12102   answers 256
        loss = nn.CrossEntropyLoss()(logits, answers)

        if self.args.fourier_loss:
            loss_all = self.args.alpha_loss * loss + (1 - self.args.alpha_loss) * fourier_loss
        else:
            loss_all = loss
        # loss_all = loss
        return loss_all


class FilterEncoder(nn.Module):
    def __init__(self, args):
        super(FilterEncoder, self).__init__()
        self.args = args
        block = FilterBlock(args)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [hidden_states]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)  # hidden_states => torch.Size([256, 50, 64])
        return all_encoder_layers


class FilterBlock(nn.Module):
    def __init__(self, args):
        super(FilterBlock, self).__init__()
        self.layer = FilterLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


class FilterLayer(nn.Module):
    def __init__(self, args):
        super(FilterLayer, self).__init__()
        self.args = args

        self.filter_layer = Filter_Model(args)

        self.attention_layer = MultiHeadAttention(args)
        self.alpha = args.alpha
        self.use_norm = False

    def forward(self, input_tensor, attention_mask):
        if self.use_norm:
            mean_enc = input_tensor.mean(1, keepdim=True).detach()  # B x 1 x E
            x_enc = input_tensor - mean_enc
            std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
            input_tensor = x_enc / std_enc
        filter = self.filter_layer(input_tensor)
        att = self.attention_layer(input_tensor, attention_mask)
        hidden_states = self.alpha * filter + (1 - self.alpha) * att
        if self.use_norm:
            hidden_states = hidden_states * std_enc + mean_enc

        return hidden_states

class Filter_Model(nn.Module):
    def __init__(self, args):
        super(Filter_Model, self).__init__()
        self.hidden_size = args.hidden_size
        self.sparsity_threshold = 0.02
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.hidden_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.hidden_size, self.hidden_size))  #可视化
        self.i1 = nn.Parameter(self.scale * torch.randn(self.hidden_size, self.hidden_size)) #可视化
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.hidden_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.hidden_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.hidden_size, self.hidden_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.hidden_size, self.hidden_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.hidden_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.hidden_size))
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(self.hidden_size, eps=1e-12)
        self.gama = args.gama
        self.chux = args.chux


    def FFN(self, x, y):
        x = self.out_dropout(x)
        x = self.LayerNorm(x + y)
        return x

    # 频率时间学习器
    def MLP_temporal(self, x, B, S, H):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=1, norm='ortho')  # FFT 沿 L 维度

        y = self.FreMLP(B, S, H, x, self.r2, self.i2, self.rb2, self.ib2)

        # print("MLP_temporal")
        # pplt = self.visualize_freq(y)

        x = torch.fft.irfft(y, n=S, dim=1, norm="ortho")
        return x

    # 频率通道学习器
    def MLP_channel(self, x, B, S, H):
        x = x.permute(1, 0, 2)
        x = torch.fft.rfft(x, dim=1, norm='ortho')  # FFT 沿 N 维度
        y = self.FreMLP(B, S, H, x, self.r1, self.i1, self.rb1, self.ib1)

        # print("MLP_channel")
        # pplt = self.visualize_freq(y)

        x = torch.fft.irfft(y, n=B, dim=1, norm="ortho")
        x = x.permute(1, 0, 2)
        return x

    # 频域 MLP
    # dimension: 沿此维度进行 FFT，r: 权重实部，i: 权重大虚部
    # rb: 偏置实部，ib: 偏置虚部
    def FreMLP(self, B, S, H, x, r, i, rb, ib):

        o1_real = torch.nn.functional.relu(
            torch.einsum('bid,dd->bid', x.real, r) - \
            torch.einsum('bid,dd->bid', x.imag, i) + \
            rb
        )

        o1_imag = torch.nn.functional.relu(
            torch.einsum('bid,dd->bid', x.imag, r) + \
            torch.einsum('bid,dd->bid', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = torch.nn.functional.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x):
        if self.chux == "p":
            B, S, H = x.shape
            bias = x
            x_use = self.MLP_channel(x, B, S, H)
            # x = self.FFN(x_use,bias)
            # x = bias + x_use
            x_squence = self.MLP_temporal(x, B, S, H)
            x = ( 1 - self.gama) * x_use + self.gama * x_squence
            x = self.FFN(x,bias)
            return x
        elif self.chux == "c":
            B, S, H = x.shape
            bias = x
            x_use = self.MLP_channel(x, B, S, H)
            # x = self.FFN(x_use,bias)
            x = bias + x_use  #chu
            x = self.MLP_temporal(x, B, S, H)
            # x = ( 1 - self.gama) * x_use + self.gama * x_squence
            x = self.FFN(x,bias)
            return x



