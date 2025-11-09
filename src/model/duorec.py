import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, TransformerEncoder
import torch.nn.functional as F

"""
[Paper]
Author: Ruihong Qiu et al.
Title: "Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation."
Conference: WSDM 2022

[Code Reference]
https://github.com/RuihongQiu/DuoRec
"""

class DuoRecModel(SequentialRecModel):
    def __init__(self, args):
        super(DuoRecModel, self).__init__(args)
        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = TransformerEncoder(args)
        self.batch_size = args.batch_size
        self.gamma = 1e-10

        self.aug_nce_fct = nn.CrossEntropyLoss()
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.tau = args.tau
        self.ssl = args.ssl
        self.sim = args.sim
        self.lmd_sem = args.lmd_sem
        self.lmd = args.lmd

        self.apply(self.init_weights)
        self.fft_loss_type = "l1"

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)
        z = z[:, -1, :]
    
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp
    
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
    
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)
    
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

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

    def mix_loss(self, pred, true, reduction=None):
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

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_out,target = self.forward(input_ids)
        # fourier_loss = self.fft_loss(seq_out, target)
        # fourier_loss = torch.mean(fourier_loss)

        seq_output = seq_out[:, -1, :]

        # cross-entropy loss
        test_item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)
        # if self.args.fourier_loss:
        #     loss = self.args.alpha_loss * loss + (1 - self.args.alpha_loss) * fourier_loss

        # Unsupervised NCE: original vs dropout
        if self.ssl in ['us', 'un']:
            aug_seq_output = self.forward(input_ids)
            nce_logits, nce_labels = self.info_nce(seq_output, aug_seq_output, temp=self.tau,
                                                batch_size=input_ids.shape[0], sim=self.sim)
            loss += self.lmd * self.aug_nce_fct(nce_logits, nce_labels)

        # Supervised NCE: original vs semantic augmentation
        if self.ssl in ['us', 'su']:
            sem_aug = same_target
            sem_aug_seq_output ,_= self.forward(sem_aug)
            sem_nce_logits, sem_nce_labels = self.info_nce(seq_output, sem_aug_seq_output, temp=self.tau,
                                                        batch_size=input_ids.shape[0], sim=self.sim)
            loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)
        
        # Unsupervised + Supervised NCE: dropout vs semantic augmentation
        if self.ssl == 'us_x':
            # unsupervised
            aug_seq_output,_ = self.forward(input_ids)
            # supervised
            sem_aug = same_target
            sem_aug_seq_output,_ = self.forward(sem_aug)

            sem_nce_logits, sem_nce_labels = self.info_nce(aug_seq_output, sem_aug_seq_output, temp=self.tau,
                                                        batch_size=input_ids.shape[0], sim=self.sim)

            loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)

        return loss
