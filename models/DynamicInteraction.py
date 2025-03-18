import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pickle

from models.Cells import RectifiedIdentityCell, IntraModelReasoningCell, GlobalLocalGuidanceCell, CrossModalRefinementCell, GlobalLocalAlignmentCell, ContextRichCrossModalCell, GlobalEnhancedSemanticCell

def unsqueeze2d(x):
    return x.unsqueeze(-1).unsqueeze(-1)

def unsqueeze3d(x):
    return x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

def clones(module, N):
    '''Produce N identical layers.'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class DynamicInteraction_Layer0(nn.Module):
    def __init__(self, args, num_cell, num_out_path):
        super(DynamicInteraction_Layer0, self).__init__()
        self.args = args
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell
        self.num_out_path = num_out_path
        self.ric = RectifiedIdentityCell(args, num_out_path)
        self.imrc = IntraModelReasoningCell(args, num_out_path)
        # self.glgc = GlobalLocalGuidanceCell(args, num_out_path)
        # self.cmfc = CrossModalFusionCell(args, num_out_path)
        self.glac = GlobalLocalAlignmentCell(args, num_out_path)
        self.cmrc = CrossModalRefinementCell(args, num_out_path)
        self.crcmc = ContextRichCrossModalCell(args, num_out_path)
        self.gesc = GlobalEnhancedSemanticCell(args, num_out_path)

    def forward(self, text, image):

        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        emb_lst[0], path_prob[0] = self.ric(text)
        # emb_lst[1], path_prob[1] = self.glgc(text, image)
        emb_lst[1], path_prob[1] = self.glac(text, image)
        emb_lst[2], path_prob[2] = self.imrc(text)
        emb_lst[3], path_prob[3] = self.cmrc(text, image)

        emb_lst[4], path_prob[4] = self.crcmc(text, image)
        emb_lst[5], path_prob[5] = self.gesc(text, image)

        gate_mask = (sum(path_prob) < self.threshold).float() 
        all_path_prob = torch.stack(path_prob, dim=2)  
        all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)     # (bsz, 4, 4)
        path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]

        aggr_res_lst = []
        for i in range(self.num_out_path):
            skip_emb = unsqueeze2d(gate_mask[:, i]) * emb_lst[0]    # (bsz, L, 768)
            res = 0
            for j in range(self.num_cell):
                cur_path = unsqueeze2d(path_prob[j][:, i])    # (bsz, 1, 1)
                if emb_lst[j].dim() == 3:
                    cur_emb = emb_lst[j]       # (bsz, L, 768)
                else:  
                    cur_emb = emb_lst[j]
                res = res + cur_path * cur_emb
            res = res + skip_emb     # (32, L, 768)
            aggr_res_lst.append(res)     # list:4

        return aggr_res_lst, all_path_prob


class DynamicInteraction_Layer(nn.Module):
    def __init__(self, args, num_cell, num_out_path):
        super(DynamicInteraction_Layer, self).__init__()
        self.args = args
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell
        self.num_out_path = num_out_path

        self.ric = RectifiedIdentityCell(args, num_out_path)
        # self.glgc = GlobalLocalGuidanceCell(args, num_out_path)
        # self.cmfc = CrossModalFusionCell(args, num_out_path)
        self.glac = GlobalLocalAlignmentCell(args, num_out_path)
        self.imrc = IntraModelReasoningCell(args, num_out_path)
        self.cmrc = CrossModalRefinementCell(args, num_out_path)
        self.crcmc = ContextRichCrossModalCell(args, num_out_path)
        self.gesc = GlobalEnhancedSemanticCell(args, num_out_path)

    def forward(self, ref_wrd, text, image):

        # assert len(ref_wrd) == self.num_cell and ref_wrd[0].dim() == 4
        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        emb_lst[0], path_prob[0] = self.ric(ref_wrd[0])
        # emb_lst[1], path_prob[1] = self.glgc(ref_wrd[1], image)
        emb_lst[1], path_prob[1] = self.glac(ref_wrd[1], image)
        emb_lst[2], path_prob[2] = self.imrc(ref_wrd[2])
        emb_lst[3], path_prob[3] = self.cmrc(ref_wrd[3], image)

        emb_lst[4], path_prob[4] = self.crcmc(ref_wrd[4], image)
        emb_lst[5], path_prob[5] = self.gesc(ref_wrd[5], image)
        
        if self.num_out_path == 1:
            aggr_res_lst = []
            gate_mask_lst = []
            res = 0
            for j in range(self.num_cell):
                gate_mask = (path_prob[j] < self.threshold / self.num_cell).float() 
                gate_mask_lst.append(gate_mask)
                skip_emb = gate_mask.unsqueeze(-1) * ref_wrd[j]
                res += path_prob[j].unsqueeze(-1) * emb_lst[j]
                res += skip_emb

            res = res / (sum(gate_mask_lst) + sum(path_prob)).unsqueeze(-1)
            all_path_prob = torch.stack(path_prob, dim=2)
            aggr_res_lst.append(res)
        else:
            gate_mask = (sum(path_prob) < self.threshold).float()     # (bsz, 4)
            all_path_prob = torch.stack(path_prob, dim=2)
            all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
            path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]

            aggr_res_lst = []
            for i in range(self.num_out_path):
                skip_emb = unsqueeze2d(gate_mask[:, i]) * emb_lst[0]
                res = 0
                for j in range(self.num_cell):
                    cur_path = unsqueeze2d(path_prob[j][:, i])
                    res = res + cur_path * emb_lst[j]
                res = res + skip_emb
                aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob





class Reversed_DynamicInteraction_Layer0(nn.Module):
    def __init__(self, args, num_cell, num_out_path):
        super(Reversed_DynamicInteraction_Layer0, self).__init__()
        self.args = args
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell
        self.num_out_path = num_out_path
        self.ric = RectifiedIdentityCell(args, num_out_path)
        self.imrc = IntraModelReasoningCell(args, num_out_path)
        # self.glgc = GlobalLocalGuidanceCell(args, num_out_path)
        # self.cmfc = CrossModalFusionCell(args, num_out_path)
        self.glac = GlobalLocalAlignmentCell(args, num_out_path)
        self.cmrc = CrossModalRefinementCell(args, num_out_path)
        self.crcmc = ContextRichCrossModalCell(args, num_out_path)
        self.gesc = GlobalEnhancedSemanticCell(args, num_out_path)

    def forward(self, text, image):

        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        emb_lst[0], path_prob[0] = self.ric(image)
        # emb_lst[1], path_prob[1] = self.glgc(image, text)
        emb_lst[1], path_prob[1] = self.glac(image, text)
        emb_lst[2], path_prob[2] = self.imrc(image)
        emb_lst[3], path_prob[3] = self.cmrc(image, text)

        emb_lst[4], path_prob[4] = self.crcmc(image, text)
        emb_lst[5], path_prob[5] = self.gesc(image, text)

        gate_mask = (sum(path_prob) < self.threshold).float()
        all_path_prob = torch.stack(path_prob, dim=2)
        all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)  # (bsz, 4, 4)
        path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]

        aggr_res_lst = []
        for i in range(self.num_out_path):
            skip_emb = unsqueeze2d(gate_mask[:, i]) * emb_lst[0]  # (bsz, L, 768)
            res = 0
            for j in range(self.num_cell):
                cur_path = unsqueeze2d(path_prob[j][:, i])  # (bsz, 1, 1)
                if emb_lst[j].dim() == 3:
                    cur_emb = emb_lst[j]  # (bsz, L, 768)
                else:
                    cur_emb = emb_lst[j]
                res = res + cur_path * cur_emb
            res = res + skip_emb  # (32, L, 768)
            aggr_res_lst.append(res)  # list:4

        return aggr_res_lst, all_path_prob


class Reversed_DynamicInteraction_Layer(nn.Module):
    def __init__(self, args, num_cell, num_out_path):
        super(Reversed_DynamicInteraction_Layer, self).__init__()
        self.args = args
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell
        self.num_out_path = num_out_path

        self.ric = RectifiedIdentityCell(args, num_out_path)
        # self.glgc = GlobalLocalGuidanceCell(args, num_out_path)
        # self.cmfc = CrossModalFusionCell(args, num_out_path)
        self.glac = GlobalLocalAlignmentCell(args, num_out_path)
        self.imrc = IntraModelReasoningCell(args, num_out_path)
        self.cmrc = CrossModalRefinementCell(args, num_out_path)
        self.crcmc = ContextRichCrossModalCell(args, num_out_path)
        self.gesc = GlobalEnhancedSemanticCell(args, num_out_path)

    def forward(self, ref_wrd, text, image):

        # assert len(ref_wrd) == self.num_cell and ref_wrd[0].dim() == 4
        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        emb_lst[0], path_prob[0] = self.ric(ref_wrd[0])
        # emb_lst[1], path_prob[1] = self.glgc(ref_wrd[1], text)
        emb_lst[1], path_prob[1] = self.glac(ref_wrd[1], text)
        emb_lst[2], path_prob[2] = self.imrc(ref_wrd[2])
        emb_lst[3], path_prob[3] = self.cmrc(ref_wrd[3], text)

        emb_lst[4], path_prob[4] = self.crcmc(ref_wrd[4], text)
        emb_lst[5], path_prob[5] = self.gesc(ref_wrd[5], text)

        if self.num_out_path == 1:
            aggr_res_lst = []
            gate_mask_lst = []
            res = 0
            for j in range(self.num_cell):
                gate_mask = (path_prob[j] < self.threshold / self.num_cell).float()
                gate_mask_lst.append(gate_mask)
                skip_emb = gate_mask.unsqueeze(-1) * ref_wrd[j]
                res += path_prob[j].unsqueeze(-1) * emb_lst[j]
                res += skip_emb

            res = res / (sum(gate_mask_lst) + sum(path_prob)).unsqueeze(-1)
            all_path_prob = torch.stack(path_prob, dim=2)
            aggr_res_lst.append(res)
        else:
            gate_mask = (sum(path_prob) < self.threshold).float()  # (bsz, 4)
            all_path_prob = torch.stack(path_prob, dim=2)
            all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
            path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]

            aggr_res_lst = []
            for i in range(self.num_out_path):
                skip_emb = unsqueeze2d(gate_mask[:, i]) * emb_lst[0]
                res = 0
                for j in range(self.num_cell):
                    cur_path = unsqueeze2d(path_prob[j][:, i])
                    res = res + cur_path * emb_lst[j]
                res = res + skip_emb
                aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob


