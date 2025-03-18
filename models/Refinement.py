import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import math
from transformers import BertConfig



def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def func_attention(query, context, raw_feature_norm, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)   #(n, d, qL)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)   #(n, cL, qL)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = F.softmax(attn, dim=-1)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        # attn = l2norm(attn, 2)
        attn = F.normalize(attn, dim=2)
    # elif raw_feature_norm == "l1norm":
    #     attn = l1norm_d(attn, 2)
    # elif raw_feature_norm == "clipped_l1norm":
    #     attn = nn.LeakyReLU(0.1)(attn)
    #     attn = l1norm_d(attn, 2)
    elif raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif raw_feature_norm == "no_norm":
        pass
    else:
        # raise ValueError("unknown first norm type:", args.raw_feature_norm)
        raise ValueError("unknown first norm type:")
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous() #(n, qL, cL)
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)    #(n*qL, cL)
    attn = F.softmax(attn*smooth, dim=-1)                #(n*qL, cL)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)   #(n, qL, cL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()    #(n, cL, qL)

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)   #(n, d, cL)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)    #(n, d, qL)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)    #(n, qL, d)

    return weightedContext, attnT



class CrossModalAlignment(nn.Module):
    def __init__(self, config):
        super(CrossModalAlignment, self).__init__()
        self.config = config
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.fc_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc_2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, text_emb, image_emb):
        """
        inputs :
            text_emb : input feature maps( B X 128 X 768 )
            image_emb : input feature maps( B X 50 X 768 )
        returns :
            out : ( B X 128 X 768 )
        """
        query_layer = self.query(text_emb)       # (bsz, 128, 768)
        key_layer = self.key(image_emb)          # (bsz, 50, 768)
        value_layer = self.value(image_emb)      # (bsz, 50, 768)

        # (bsz, 128, 50)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2))
        attn_score = attention_scores / math.sqrt(self.config.hidden_size)

        # Softmax attention   (bsz, 128, 768)
        attn_score = torch.softmax(100 * attn_score, dim=-1)
        text_img_rep_init = torch.bmm(attn_score, value_layer)

        return text_img_rep_init


class Refinement(nn.Module):
    def __init__(self, args, embed_size, raw_feature_norm, lambda_softmax):
        super(Refinement, self).__init__()
        self.raw_feature_norm = raw_feature_norm
        self.lambda_softmax = lambda_softmax

        self.fc_scale = nn.Linear(embed_size, embed_size)
        self.fc_shift = nn.Linear(embed_size, embed_size)
        self.fc_1 = nn.Linear(embed_size, embed_size)
        self.fc_2 = nn.Linear(embed_size, embed_size)

        self.CrossModalAlignment = CrossModalAlignment(BertConfig.from_pretrained(args.bert_name))

    def refine(self, query, weiContext):
        scaling = torch.tanh(self.fc_scale(weiContext))
        shifting = self.fc_shift(weiContext)  
        modu_res = self.fc_2(F.relu(self.fc_1(query * scaling + shifting))) 
        ref_q = modu_res + query

        return ref_q

    def forward(self, text, image):
        '''
        Args:
            text: (bsz, 128, 768)
            image: (bsz, 50, 768)

        Returns: (bsz, 128, 768)
        '''

        # weiContext, attn = func_attention(text, image, self.raw_feature_norm, smooth=self.lambda_softmax)
        weiContext = self.CrossModalAlignment(text, image)
        ref_wrd = self.refine(text, weiContext)   # (bsz, 128, 768)

        return ref_wrd

    

