import torch
from torch import nn
from .modeling_unimo import UnimoModel
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


def js_div(p_output, q_output, labels, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    margin = 0.5
    labels = labels.unsqueeze(1)
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)

    # 0 is Sarcasm and 1 is Non-Sarcasm
    # KLDivLoss()
    new_q_output = labels * q_output + (1 - labels) * (1 - q_output)
    # new_p_output = labels * p_output + (1 - labels) * (1 - p_output)
    # log_mean_output = ((p_output + q_output) / 2).log()
    regularizer = (1 / (torch.norm(0.5 - q_output) + margin) + (1 / torch.norm(0.5 - p_output)) + margin) / 2

    return KLDivLoss(p_output.log(), new_q_output) + 0.5 * regularizer


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def cal_raido(self, labels):
        length = labels.shape[0]
        num_pos = torch.eq(labels.squeeze(), 1).sum().item()
        num_neg = length - num_pos
        radio1 = torch.ones(length) * (num_pos / length)
        radio2 = torch.ones(length) * (num_neg / length)
        radio_label = torch.where(labels.cpu().squeeze() == 1, radio2, radio1)
        return radio_label

    def forward(self, features, labels=None, mask=None, similary=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')

            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # radio_label = self.cal_raido(labels)
        # radio_label = radio_label.to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # (bs, hidden_dim)
        if self.contrast_mode == 'one':
            # (bs, hidden_dim)
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        mean_log_prob_pos = mean_log_prob_pos * (1 - similary)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class UnimoModelF(nn.Module):
    def __init__(self, args, vision_config, text_config):
        super(UnimoModelF, self).__init__()
        self.args = args
        self.vision_config = vision_config
        self.text_config = text_config
        self.model = UnimoModel(args, vision_config, text_config)
        self.fc = nn.Linear(self.text_config.hidden_size, 3)

        self.CE_Loss = CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, labels, images):
        output, js_loss = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            pixel_values=images,
                            return_dict=True
                            )
        pool_out = output.pooler_output
        # 分类头   (bsz, 768)  ->   (bsz, 3)
        final_output = self.fc(pool_out)

        loss = self.CE_Loss(final_output, labels.long()) + js_loss

        return (loss, final_output)
