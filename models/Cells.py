import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .SelfAttention import SelfAttention
from .Router import Router
from models.Refinement import Refinement

from transformers import BertConfig, CLIPConfig
import numpy as np

from models.XModules import CrossModalAlignment, AttentionFiltration, SoftContrastiveLoss, SelfEncoder


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class RectifiedIdentityCell(nn.Module):
    def __init__(self, args, num_out_path):
        super(RectifiedIdentityCell, self).__init__()
        self.keep_mapping = nn.ReLU()
        self.router = Router(num_out_path, args.embed_size, args.hid_router)

    def forward(self, x):
        path_prob = self.router(x)     # (bsz, L, 768) -> (bsz, 4)
        emb = self.keep_mapping(x)

        return emb, path_prob

class IntraModelReasoningCell(nn.Module):
    def __init__(self, args, num_out_path):
        super(IntraModelReasoningCell, self).__init__()
        self.args = args
        self.router = Router(num_out_path, args.embed_size, args.hid_router)
        self.sa = SelfAttention(args.embed_size, args.hid_IMRC, args.num_head_IMRC)

    def forward(self, inp):
        path_prob = self.router(inp)
        if inp.dim() == 4:
            n_img, n_stc, n_local, dim = inp.size()
            x = inp.view(-1, n_local, dim)
        else:
            x = inp

        sa_emb = self.sa(x)
        if inp.dim() == 4:
            sa_emb = sa_emb.view(n_img, n_stc, n_local, -1)
        return sa_emb, path_prob

# class IntraModelReasoningCell(nn.Module):
#     def __init__(self, args, num_out_path):
#         super(IntraModelReasoningCell, self).__init__()
#         self.args = args
#         self.router = Router(num_out_path, args.embed_size, args.hid_router)
#         self.sa = SelfEncoder(BertConfig.from_pretrained(args.bert_name), BertConfig.from_pretrained(args.bert_name).hidden_size, head=64, drop=0.0)
#
#     def forward(self, inp):
#         path_prob = self.router(inp)
#         sa_emb = self.sa(inp)
#
#         return sa_emb, path_prob


class CrossModalRefinementCell(nn.Module):
    def __init__(self, args, num_out_path):
        super(CrossModalRefinementCell, self).__init__()
        self.refine = Refinement(args, args.embed_size, args.raw_feature_norm_CMRC, args.lambda_softmax_CMRC)
        self.router = Router(num_out_path, args.embed_size, args.hid_router)

    def forward(self, text, image):

        path_prob = self.router(text)
        rf_pairs_emb = self.refine(text, image)

        return rf_pairs_emb, path_prob


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class GlobalLocalGuidanceCell(nn.Module):
    def __init__(self, args, num_out_path):
        super(GlobalLocalGuidanceCell, self).__init__()
        self.args = args
        self.router = Router(num_out_path, args.embed_size, args.hid_router)
        self.fc_1 = nn.Linear(args.embed_size, args.embed_size)
        self.fc_2 = nn.Linear(args.embed_size, args.embed_size)
        self.image_cls_pool = BertPooler(BertConfig.from_pretrained(args.bert_name))

    def regulate(self, l_emb, g_emb_expand):
        l_emb_mid = self.fc_1(l_emb)
        x = l_emb_mid * g_emb_expand
        x = F.normalize(x, dim=-2)
        ref_l_emb = (1 + x) * l_emb
        return ref_l_emb

    def forward(self, text, image):

        path_prob = self.router(text)
        global_image = self.image_cls_pool(image).unsqueeze(-2)    # (bsz, 1, 768)
        ref_emb = self.regulate(text, global_image)

        return ref_emb, path_prob



class GlobalLocalAlignmentCell(nn.Module):
    def __init__(self, args, num_out_path):
        super(GlobalLocalAlignmentCell, self).__init__()
        self.args = args
        self.router = Router(num_out_path, args.embed_size, args.hid_router)
        self.CrossModalAlignment = CrossModalAlignment(BertConfig.from_pretrained(args.bert_name), args)
        self.SAF_module = AttentionFiltration(BertConfig.from_pretrained(args.bert_name).hidden_size)
        self.text_cls_pool = BertPooler(BertConfig.from_pretrained(args.bert_name))
        self.image_cls_pool = BertPooler(CLIPConfig.from_pretrained(args.vit_name).vision_config)
        self.fc_sim_tranloc = nn.Linear(768, 768)
        self.fc_sim_tranglo = nn.Linear(768, 768)
        self.fc_1 = nn.Linear(768, 768)
        self.fc_2 = nn.Linear(768, 768)

    def alignment(self, text, image):
        # 局部相似性表征
        text_aware_image, _ = self.CrossModalAlignment(text, image)  # (32, 128, 768)

        sim_local = torch.pow(torch.sub(text, text_aware_image), 2)
        sim_local = l2norm(self.fc_sim_tranloc(sim_local), dim=-1)
        sim_local = self.fc_1(sim_local)

        # 全局相似性表征   （32, 768）
        text_cls_output = self.text_cls_pool(text)
        image_cls_output = self.image_cls_pool(image)
        sim_global = torch.pow(torch.sub(text_cls_output, image_cls_output), 2)
        sim_global = l2norm(self.fc_sim_tranglo(sim_global), dim=-1)
        sim_global = self.fc_2(sim_global)

        # concat the global and local alignments
        sim_emb = torch.cat([sim_global.unsqueeze(1), sim_local], 1)  # (bsz, L+1, 768)

        # 相似图推理
        sim_emb = self.SAF_module(sim_emb)  # (bsz, 768)

        return sim_emb

    def forward(self, text, image):

        path_prob = self.router(text)

        sim_emb = self.alignment(text, image)
        sim_emb = sim_emb.unsqueeze(-2).expand(-1, text.size(1), -1)

        return sim_emb, path_prob



class GlobalEnhancedSemanticCell(nn.Module):
    def __init__(self, args, num_out_path):
        super(GlobalEnhancedSemanticCell, self).__init__()
        self.args = args
        self.router = Router(num_out_path, args.embed_size, args.hid_router)

        # # MildTriple Loss
        # self.mild_loss = SoftContrastiveLoss(alpha=args.beta, margin=args.mild_margin, max_violation=True,
        #                                      threshold_hetero=args.hetero, threshold_homo=args.homo)
        # self.fc_1 = nn.Linear(768, 768)
        # self.fc_2 = nn.Linear(768, 768)
        self.text_cls_pool = BertPooler(BertConfig.from_pretrained(args.bert_name))
        self.image_cls_pool = BertPooler(BertConfig.from_pretrained(args.bert_name))

        self.fc_mlp = nn.Sequential(nn.Linear(768, 768),
                                 nn.Tanh(),
                                 nn.Linear(768, 768))

    def global_gate_fusion(self, text, image):
        text_cls = self.text_cls_pool(text)  # (bsz, 768)
        image_cls = self.image_cls_pool(image)  # (bsz, 768)

        # # MildTriple Loss
        # mild_loss = self.mild_loss(torch.nn.functional.normalize(self.fc_1(text_cls)),
        #                            torch.nn.functional.normalize(self.fc_2(image_cls)))

        # 门控机制   全局信息对齐、融合
        gate_all = self.fc_mlp(text_cls + image_cls)  # (bsz, 768)
        gate = torch.softmax(gate_all, dim=-1)  # (bsz, 768)
        gate_out = gate * text_cls + (1 - gate) * image_cls  # (bsz, 768)
        gate_out = gate_out.unsqueeze(-2).expand(-1, text.size(1), -1)

        return gate_out

    def forward(self, text, image):

        path_prob = self.router(text)
        gate_out = self.global_gate_fusion(text, image)

        return gate_out, path_prob



class  ContextRichCrossModalCell(nn.Module):
    def __init__(self, args, num_out_path):
        super(ContextRichCrossModalCell, self).__init__()
        self.args = args
        self.router = Router(num_out_path, args.embed_size, args.hid_router)

        self.CrossModalAlignment = CrossModalAlignment(BertConfig.from_pretrained(args.bert_name), args)
        self.fc_mlp_1 = nn.Sequential(nn.Linear(768, 768),
                                 nn.Tanh())
        self.fc_mlp_2 = nn.Sequential(nn.Linear(768, 768),
                                 nn.Tanh())
        self.fc_1 = nn.Linear(768, 768)
        self.fc_2 = nn.Linear(768, 768)

    def alignment(self, text, image):

        text_aware_image, _ = self.CrossModalAlignment(text, image)   # (32, 128, 768)
        Q_state = self.fc_mlp_1(text_aware_image)    # (32, 128, 768)
        K_state = self.fc_mlp_2(text)  # (32, 128, 768)
        Q = self.fc_1(Q_state)
        K = self.fc_2(K_state)

        scores = torch.matmul(Q, K.transpose(-1, -2))   # (bsz, 128, 128)
        scores = nn.Softmax(dim=-1)(scores)   # (bsz, 128, 128)
        output = Q_state + torch.bmm(scores, K_state)   # (32, 128, 768)

        return output

    def forward(self, text, image):

        path_prob = self.router(text)
        output = self.alignment(text, image)

        return output, path_prob




class CrossModalFusionCell(nn.Module):
    def __init__(self, args, num_out_path):
        super(CrossModalFusionCell, self).__init__()
        self.args = args
        self.router = Router(num_out_path, args.embed_size, args.hid_router)
        self.fc_1 = nn.Linear(768, 768)
        self.fc_2 = nn.Linear(768, 768)
        self.fc_gate = nn.Linear(768, 1)
        self.gate_act = nn.Tanh()
        self.gate_layer_norm = nn.LayerNorm(768)
        self.cls_pool = BertPooler(BertConfig.from_pretrained(args.bert_name))

    def glo_to_loc(self, text, image):

        text_cls = self.cls_pool(text).unsqueeze(-2)  # (bsz, 1, 768)
        text_cls = self.fc_1(text_cls)  # (bsz, 1, 768)
        text_cls_ori = text_cls
        image_tokens = self.fc_2(image)  # (bsz, 50, 768)

        cross_modal_score = torch.matmul(text_cls, image_tokens.transpose(-1, -2))  # (bsz, 1, 50)
        cross_modal_probs = nn.Softmax(dim=-1)(cross_modal_score)  # (bsz, 1, 50)
        context = torch.matmul(cross_modal_probs, image_tokens)  # (bsz, 1, 768)
        gate_score = self.gate_act(self.fc_gate(text_cls_ori))  # (bsz, 1, 1)
        final_context = self.gate_layer_norm((text_cls_ori * gate_score) + context)  # (bsz, 1, 768)
        final_context = final_context.expand(-1, text.size(1), -1) + text  # (bsz, 128, 768)

        return final_context

    def forward(self, text, image):

        path_prob = self.router(text)
        final_context = self.glo_to_loc(text, image)

        return final_context, path_prob



class TextBasedGlobalLocalCell(nn.Module):
    def __init__(self, args):
        super(TextBasedGlobalLocalCell, self).__init__()
        self.args = args
        self.fc_query = nn.Linear(self.args.model.input_hidden_dim, self.args.model.TGLU_hidden_dim)
        self.fc_key = nn.Linear(self.args.model.input_hidden_dim, self.args.model.TGLU_hidden_dim)
        self.fc_value = nn.Linear(self.args.model.input_hidden_dim, self.args.model.TGLU_hidden_dim)
        self.fc_cls = nn.Linear(self.args.model.input_hidden_dim, self.args.model.TGLU_hidden_dim)
        self.layer_norm = nn.LayerNorm(self.args.model.TGLU_hidden_dim)

    def forward(self,
                entity_text_cls,
                entity_text_tokens,
                mention_text_cls,
                mention_text_tokens):
        """

        :param entity_text_cls:     [num_entity, dim]
        :param entity_text_tokens:  [num_entity, max_seq_len, dim]
        :param mention_text_cls:    [batch_size, dim]
        :param mention_text_tokens: [batch_size, max_sqe_len, dim]
        :return:
        """

        entity_cls_fc = self.fc_cls(entity_text_cls)  # [num_entity, dim]
        entity_cls_fc = entity_cls_fc.unsqueeze(dim=1)  # [num_entity, 1, dim]

        query = self.fc_query(entity_text_tokens)  # [num_entity, max_seq_len, dim]
        key = self.fc_key(mention_text_tokens)  # [batch_size, max_sqe_len, dim]
        value = self.fc_value(mention_text_tokens)  # [batch_size, max_sqe_len, dim]

        query = query.unsqueeze(dim=1)  # [num_entity, 1, max_seq_len, dim]
        key = key.unsqueeze(dim=0)  # [1, batch_size, max_sqe_len, dim]
        value = value.unsqueeze(dim=0)  # [1, batch_size, max_sqe_len, dim]

        attention_scores = torch.matmul(query,
                                        key.transpose(-1, -2))  # [num_entity, batch_size, max_seq_len, max_seq_len]

        attention_scores = attention_scores / math.sqrt(self.args.model.TGLU_hidden_dim)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [num_entity, batch_size, max_seq_len, max_seq_len]

        context = torch.matmul(attention_probs, value)  # [num_entity, batch_size, max_seq_len, dim]
        context = torch.mean(context, dim=-2)  # [num_entity, batch_size, dim]
        context = self.layer_norm(context)

        g2l_matching_score = torch.sum(entity_cls_fc * context, dim=-1)  # [num_entity, batch_size]
        g2l_matching_score = g2l_matching_score.transpose(0, 1)  # [batch_size, num_entity]
        g2g_matching_score = torch.matmul(mention_text_cls, entity_text_cls.transpose(-1, -2))

        matching_score = (g2l_matching_score + g2g_matching_score) / 2
        return matching_score


class DUAL(nn.Module):
    def __init__(self, args):
        super(DUAL, self).__init__()
        self.args = args
        self.cls_fc = nn.Linear(self.args.model.dv, self.args.model.IDLU_hidden_dim)
        self.tokens_fc = nn.Linear(self.args.model.dv, self.args.model.IDLU_hidden_dim)
        self.fc = nn.Linear(self.args.model.IDLU_hidden_dim, self.args.model.IDLU_hidden_dim)
        self.gate_fc = nn.Linear(self.args.model.IDLU_hidden_dim, 1)
        self.activation = nn.Tanh()
        self.add_layer_norm = nn.LayerNorm(self.args.model.IDLU_hidden_dim)
        self.layer_norm = nn.LayerNorm(self.args.model.IDLU_hidden_dim)

    def forward(self,
                query_cls,
                key_cls,
                value_tokens):
        """

        :param query_cls:       [a, dim]
        :param key_cls:         [b, dim]
        :param value_tokens:    [b, num_patch, dim]
        :return:
        """
        query_cls = self.cls_fc(query_cls)
        key_cls = self.cls_fc(key_cls)
        value_tokens = self.tokens_fc(value_tokens)

        value_pooled = torch.mean(value_tokens, dim=-2)  # [b, dim]
        value = value_pooled.unsqueeze(dim=0)  # [1, b, dim]
        query = query_cls.unsqueeze(dim=1)  # [a, 1, dim]
        context = self.add_layer_norm(value + query)
        context = self.fc(context)  # [a, b, dim]

        gate_value = self.activation(self.gate_fc(context))  # [a, b, 1]
        aggregated_value = (context * gate_value) + key_cls.unsqueeze(dim=0)  # [a, b, dim]
        aggregated_value = self.layer_norm(aggregated_value)  # [a, b, dim]

        query_cls = self.layer_norm(query)  # [a, 1, dim]
        score = torch.sum(aggregated_value * query_cls, dim=-1)
        return score


class VisionBasedDualUnit(nn.Module):
    def __init__(self, args):
        super(VisionBasedDualUnit, self).__init__()
        self.args = args

        self.dual_ent2men = DUAL(self.args)     # dual function from entity to mention
        self.dual_men2ent = DUAL(self.args)     # dual function from mention to entity

    def forward(self,
                entity_image_cls, entity_image_tokens,
                mention_image_cls, mention_image_tokens):
        """
        :param entity_image_cls:        [num_entity, dim]
        :param entity_image_tokens:     [num_entity, num_patch, dim]
        :param mention_image_cls:       [batch_size, dim]
        :param mention_image_tokens:    [batch_size, num_patch, dim]
        :return:
        """

        entity_to_mention_score = self.dual_ent2men(entity_image_cls, mention_image_cls, mention_image_tokens)
        mention_to_entity_score = self.dual_men2ent(mention_image_cls, entity_image_cls, entity_image_tokens)

        dual_score = (entity_to_mention_score.transpose(0, 1) + mention_to_entity_score) / 2  # [batch_size, num_entity]
        return dual_score



class CrossModalFusionUnit(nn.Module):
    def __init__(self, args):
        super(CrossModalFusionUnit, self).__init__()
        self.args = args
        self.text_fc = nn.Linear(self.args.model.input_hidden_dim, self.args.model.CMFU_hidden_dim)
        self.image_fc = nn.Linear(self.args.model.dv, self.args.model.CMFU_hidden_dim)
        self.gate_fc = nn.Linear(self.args.model.CMFU_hidden_dim, 1)
        self.gate_act = nn.Tanh()
        self.gate_layer_norm = nn.LayerNorm(self.args.model.CMFU_hidden_dim)
        self.context_layer_norm = nn.LayerNorm(self.args.model.CMFU_hidden_dim)

    def forward(self, entity_text_cls, entity_image_tokens,
                mention_text_cls, mention_image_tokens):
        """
        :param entity_text_cls:         [num_entity, dim]
        :param entity_image_tokens:     [num_entity, num_patch, dim]
        :param mention_text_cls:        [batch_size, dim]
        :param mention_image_tokens:    [batch_size, num_patch, dim]
        :return:
        """
        entity_text_cls = self.text_fc(entity_text_cls)  # [num_entity, dim]
        entity_text_cls_ori = entity_text_cls
        mention_text_cls = self.text_fc(mention_text_cls)  # [batch_size, dim]
        mention_text_cls_ori = mention_text_cls

        entity_image_tokens = self.image_fc(entity_image_tokens)  # [num_entity, num_patch, dim]
        mention_image_tokens = self.image_fc(mention_image_tokens)  # [batch_size, num_patch, dim]

        entity_text_cls = entity_text_cls.unsqueeze(dim=1)  # [num_entity, 1, dim]
        entity_cross_modal_score = torch.matmul(entity_text_cls, entity_image_tokens.transpose(-1, -2))
        entity_cross_modal_probs = nn.Softmax(dim=-1)(entity_cross_modal_score)  # [num_entity, 1, num_patch]
        entity_context = torch.matmul(entity_cross_modal_probs, entity_image_tokens).squeeze()  # [num_entity, 1, dim]
        entity_gate_score = self.gate_act(self.gate_fc(entity_text_cls_ori))
        entity_context = self.gate_layer_norm((entity_text_cls_ori * entity_gate_score) + entity_context)

        mention_text_cls = mention_text_cls.unsqueeze(dim=1)  # [batch_size, 1, dim]
        mention_cross_modal_score = torch.matmul(mention_text_cls, mention_image_tokens.transpose(-1, -2))
        mention_cross_modal_probs = nn.Softmax(dim=-1)(mention_cross_modal_score)
        mention_context = torch.matmul(mention_cross_modal_probs, mention_image_tokens).squeeze()
        mention_gate_score = self.gate_act(self.gate_fc(mention_text_cls_ori))
        mention_context = self.gate_layer_norm((mention_text_cls_ori * mention_gate_score) + mention_context)

        score = torch.matmul(mention_context, entity_context.transpose(-1, -2))
        return score