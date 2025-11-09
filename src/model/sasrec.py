import torch
import torch.nn as nn
import copy
from model._abstract_model import SequentialRecModel
from model._modules import TransformerEncoder, LayerNorm
import torch.nn.functional as F

"""
[Paper]
Author: Wang-Cheng Kang et al. 
Title: "Self-Attentive Sequential Recommendation."
Conference: ICDM 2018

[Code Reference]
https://github.com/kang205/SASRec
https://github.com/Woeee/FMLP-Rec
"""

class SASRecModel(SequentialRecModel):
    def __init__(self, args):
        super(SASRecModel, self).__init__(args)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.item_encoder = TransformerEncoder(args)
        self.apply(self.init_weights)
        self.fft_loss_type = "l1"
        self.args.fourier_loss = False

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output,sequence_emb

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

        seq_out,target = self.forward(input_ids)

        fourier_loss = self.fft_loss(seq_out, target)
        fourier_loss = torch.mean(fourier_loss)

        seq_out = seq_out[:, -1, :]
        pos_ids, neg_ids = answers, neg_answers

        # [batch seq_len hidden_size]
        pos_emb = self.item_embeddings(pos_ids)
        neg_emb = self.item_embeddings(neg_ids)

        # [batch hidden_size]
        seq_emb = seq_out # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos_emb * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg_emb * seq_emb, -1)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=seq_out.device), torch.zeros(neg_logits.shape, device=seq_out.device)
        indices = (pos_ids != 0).nonzero().reshape(-1)
        bce_criterion = torch.nn.BCEWithLogitsLoss()
        loss = bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += bce_criterion(neg_logits[indices], neg_labels[indices])

        if self.args.fourier_loss:
            loss_all = self.args.alpha_loss * loss + (1 - self.args.alpha_loss) * fourier_loss
        else:
            loss_all = loss

        return loss_all
