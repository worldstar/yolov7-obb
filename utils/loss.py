# Loss functions

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import bbox_iou, bbox_alpha_iou, box_iou, box_giou, box_diou, box_ciou, xywh2xyxy
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class SigmoidBin(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, bin_count=10, min=0.0, max=1.0, reg_scale = 2.0, use_loss_regression=True, use_fw_regression=True, BCE_weight=1.0, smooth_eps=0.0):
        super(SigmoidBin, self).__init__()
        
        self.bin_count = bin_count
        self.length = bin_count + 1
        self.min = min
        self.max = max
        self.scale = float(max - min)
        self.shift = self.scale / 2.0

        self.use_loss_regression = use_loss_regression
        self.use_fw_regression = use_fw_regression
        self.reg_scale = reg_scale
        self.BCE_weight = BCE_weight

        start = min + (self.scale/2.0) / self.bin_count
        end = max - (self.scale/2.0) / self.bin_count
        step = self.scale / self.bin_count
        self.step = step
        #print(f" start = {start}, end = {end}, step = {step} ")

        bins = torch.range(start, end + 0.0001, step).float() 
        self.register_buffer('bins', bins) 
               

        self.cp = 1.0 - 0.5 * smooth_eps
        self.cn = 0.5 * smooth_eps

        self.BCEbins = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([BCE_weight]))
        self.MSELoss = nn.MSELoss()

    def get_length(self):
        return self.length

    def forward(self, pred):
        assert pred.shape[-1] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (pred.shape[-1], self.length)

        pred_reg = (pred[..., 0] * self.reg_scale - self.reg_scale/2.0) * self.step
        pred_bin = pred[..., 1:(1+self.bin_count)]

        _, bin_idx = torch.max(pred_bin, dim=-1)
        bin_bias = self.bins[bin_idx]

        if self.use_fw_regression:
            result = pred_reg + bin_bias
        else:
            result = bin_bias
        result = result.clamp(min=self.min, max=self.max)

        return result


    def training_loss(self, pred, target):
        assert pred.shape[-1] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (pred.shape[-1], self.length)
        assert pred.shape[0] == target.shape[0], 'pred.shape=%d is not equal to the target.shape=%d' % (pred.shape[0], target.shape[0])
        device = pred.device

        pred_reg = (pred[..., 0].sigmoid() * self.reg_scale - self.reg_scale/2.0) * self.step
        pred_bin = pred[..., 1:(1+self.bin_count)]

        diff_bin_target = torch.abs(target[..., None] - self.bins)
        _, bin_idx = torch.min(diff_bin_target, dim=-1)
    
        bin_bias = self.bins[bin_idx]
        bin_bias.requires_grad = False
        result = pred_reg + bin_bias

        target_bins = torch.full_like(pred_bin, self.cn, device=device)  # targets
        n = pred.shape[0] 
        target_bins[range(n), bin_idx] = self.cp

        loss_bin = self.BCEbins(pred_bin, target_bins) # BCE

        if self.use_loss_regression:
            loss_regression = self.MSELoss(result, target)  # MSE        
            loss = loss_bin + loss_regression
        else:
            loss = loss_bin

        out_result = result.clamp(min=self.min, max=self.max)

        return loss, out_result


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class RankSort(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, delta_RS=0.50, eps=1e-10): 

        classification_grads=torch.zeros(logits.shape).cuda()
        
        #Filter fg logits
        fg_labels = (targets > 0.)
        fg_logits = logits[fg_labels]
        fg_targets = targets[fg_labels]
        fg_num = len(fg_logits)

        #Do not use bg with scores less than minimum fg logit
        #since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits)-delta_RS
        relevant_bg_labels=((targets==0) & (logits>=threshold_logit))
        
        relevant_bg_logits = logits[relevant_bg_labels] 
        relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
        sorting_error=torch.zeros(fg_num).cuda()
        ranking_error=torch.zeros(fg_num).cuda()
        fg_grad=torch.zeros(fg_num).cuda()
        
        #sort the fg logits
        order=torch.argsort(fg_logits)
        #Loops over each positive following the order
        for ii in order:
            # Difference Transforms (x_ij)
            fg_relations=fg_logits-fg_logits[ii] 
            bg_relations=relevant_bg_logits-fg_logits[ii]

            if delta_RS > 0:
                fg_relations=torch.clamp(fg_relations/(2*delta_RS)+0.5,min=0,max=1)
                bg_relations=torch.clamp(bg_relations/(2*delta_RS)+0.5,min=0,max=1)
            else:
                fg_relations = (fg_relations >= 0).float()
                bg_relations = (bg_relations >= 0).float()

            # Rank of ii among pos and false positive number (bg with larger scores)
            rank_pos=torch.sum(fg_relations)
            FP_num=torch.sum(bg_relations)

            # Rank of ii among all examples
            rank=rank_pos+FP_num
                            
            # Ranking error of example ii. target_ranking_error is always 0. (Eq. 7)
            ranking_error[ii]=FP_num/rank      

            # Current sorting error of example ii. (Eq. 7)
            current_sorting_error = torch.sum(fg_relations*(1-fg_targets))/rank_pos

            #Find examples in the target sorted order for example ii         
            iou_relations = (fg_targets >= fg_targets[ii])
            target_sorted_order = iou_relations * fg_relations

            #The rank of ii among positives in sorted order
            rank_pos_target = torch.sum(target_sorted_order)

            #Compute target sorting error. (Eq. 8)
            #Since target ranking error is 0, this is also total target error 
            target_sorting_error= torch.sum(target_sorted_order*(1-fg_targets))/rank_pos_target

            #Compute sorting error on example ii
            sorting_error[ii] = current_sorting_error - target_sorting_error
  
            #Identity Update for Ranking Error 
            if FP_num > eps:
                #For ii the update is the ranking error
                fg_grad[ii] -= ranking_error[ii]
                #For negatives, distribute error via ranking pmf (i.e. bg_relations/FP_num)
                relevant_bg_grad += (bg_relations*(ranking_error[ii]/FP_num))

            #Find the positives that are misranked (the cause of the error)
            #These are the ones with smaller IoU but larger logits
            missorted_examples = (~ iou_relations) * fg_relations

            #Denominotor of sorting pmf 
            sorting_pmf_denom = torch.sum(missorted_examples)

            #Identity Update for Sorting Error 
            if sorting_pmf_denom > eps:
                #For ii the update is the sorting error
                fg_grad[ii] -= sorting_error[ii]
                #For positives, distribute error via sorting pmf (i.e. missorted_examples/sorting_pmf_denom)
                fg_grad += (missorted_examples*(sorting_error[ii]/sorting_pmf_denom))

        #Normalize gradients by number of positives 
        classification_grads[fg_labels]= (fg_grad/fg_num)
        classification_grads[relevant_bg_labels]= (relevant_bg_grad/fg_num)

        ctx.save_for_backward(classification_grads)

        return ranking_error.mean(), sorting_error.mean()

    @staticmethod
    def backward(ctx, out_grad1, out_grad2):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None, None

class aLRPLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, regression_losses, delta=1., eps=1e-5): 
        classification_grads=torch.zeros(logits.shape).cuda()
        
        #Filter fg logits
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)

        #Do not use bg with scores less than minimum fg logit
        #since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits)-delta

        #Get valid bg logits
        relevant_bg_labels=((targets==0)&(logits>=threshold_logit))
        relevant_bg_logits=logits[relevant_bg_labels] 
        relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
        rank=torch.zeros(fg_num).cuda()
        prec=torch.zeros(fg_num).cuda()
        fg_grad=torch.zeros(fg_num).cuda()
        
        max_prec=0                                           
        #sort the fg logits
        order=torch.argsort(fg_logits)
        #Loops over each positive following the order
        for ii in order:
            #x_ij s as score differences with fgs
            fg_relations=fg_logits-fg_logits[ii] 
            #Apply piecewise linear function and determine relations with fgs
            fg_relations=torch.clamp(fg_relations/(2*delta)+0.5,min=0,max=1)
            #Discard i=j in the summation in rank_pos
            fg_relations[ii]=0

            #x_ij s as score differences with bgs
            bg_relations=relevant_bg_logits-fg_logits[ii]
            #Apply piecewise linear function and determine relations with bgs
            bg_relations=torch.clamp(bg_relations/(2*delta)+0.5,min=0,max=1)

            #Compute the rank of the example within fgs and number of bgs with larger scores
            rank_pos=1+torch.sum(fg_relations)
            FP_num=torch.sum(bg_relations)
            #Store the total since it is normalizer also for aLRP Regression error
            rank[ii]=rank_pos+FP_num
                            
            #Compute precision for this example to compute classification loss 
            prec[ii]=rank_pos/rank[ii]                
            #For stability, set eps to a infinitesmall value (e.g. 1e-6), then compute grads
            if FP_num > eps:   
                fg_grad[ii] = -(torch.sum(fg_relations*regression_losses)+FP_num)/rank[ii]
                relevant_bg_grad += (bg_relations*(-fg_grad[ii]/FP_num))   
                    
        #aLRP with grad formulation fg gradient
        classification_grads[fg_labels]= fg_grad
        #aLRP with grad formulation bg gradient
        classification_grads[relevant_bg_labels]= relevant_bg_grad 
 
        classification_grads /= (fg_num)
    
        cls_loss=1-prec.mean()
        ctx.save_for_backward(classification_grads)

        return cls_loss, rank, order

    @staticmethod
    def backward(ctx, out_grad1, out_grad2, out_grad3):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None, None, None
    
    
class APLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, delta=1.): 
        classification_grads=torch.zeros(logits.shape).cuda()
        
        #Filter fg logits
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)

        #Do not use bg with scores less than minimum fg logit
        #since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits)-delta

        #Get valid bg logits
        relevant_bg_labels=((targets==0)&(logits>=threshold_logit))
        relevant_bg_logits=logits[relevant_bg_labels] 
        relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
        rank=torch.zeros(fg_num).cuda()
        prec=torch.zeros(fg_num).cuda()
        fg_grad=torch.zeros(fg_num).cuda()
        
        max_prec=0                                           
        #sort the fg logits
        order=torch.argsort(fg_logits)
        #Loops over each positive following the order
        for ii in order:
            #x_ij s as score differences with fgs
            fg_relations=fg_logits-fg_logits[ii] 
            #Apply piecewise linear function and determine relations with fgs
            fg_relations=torch.clamp(fg_relations/(2*delta)+0.5,min=0,max=1)
            #Discard i=j in the summation in rank_pos
            fg_relations[ii]=0

            #x_ij s as score differences with bgs
            bg_relations=relevant_bg_logits-fg_logits[ii]
            #Apply piecewise linear function and determine relations with bgs
            bg_relations=torch.clamp(bg_relations/(2*delta)+0.5,min=0,max=1)

            #Compute the rank of the example within fgs and number of bgs with larger scores
            rank_pos=1+torch.sum(fg_relations)
            FP_num=torch.sum(bg_relations)
            #Store the total since it is normalizer also for aLRP Regression error
            rank[ii]=rank_pos+FP_num
                            
            #Compute precision for this example 
            current_prec=rank_pos/rank[ii]
            
            #Compute interpolated AP and store gradients for relevant bg examples
            if (max_prec<=current_prec):
                max_prec=current_prec
                relevant_bg_grad += (bg_relations/rank[ii])
            else:
                relevant_bg_grad += (bg_relations/rank[ii])*(((1-max_prec)/(1-current_prec)))
            
            #Store fg gradients
            fg_grad[ii]=-(1-max_prec)
            prec[ii]=max_prec 

        #aLRP with grad formulation fg gradient
        classification_grads[fg_labels]= fg_grad
        #aLRP with grad formulation bg gradient
        classification_grads[relevant_bg_labels]= relevant_bg_grad 
 
        classification_grads /= fg_num
    
        cls_loss=1-prec.mean()
        ctx.save_for_backward(classification_grads)

        return cls_loss

    @staticmethod
    def backward(ctx, out_grad1):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEtheta = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['theta_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
            BCEtheta = FocalLoss(BCEtheta, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.stride = det.stride # tensor([8., 16., 32., ...])
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        # self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.ssi = list(self.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.BCEtheta = BCEtheta
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        ltheta = torch.zeros(1, device=device)
        # tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        tcls, tbox, indices, anchors, tgaussian_theta = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                #tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification (Supasin W. 112.01.11)
                class_index = 5 + self.nc
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t = torch.full_like(ps[:, 5:class_index], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    # t[t==self.cp] = iou.detach().clamp(0).type(t.dtype)
                    # lcls += self.BCEcls(ps[:, 5:], t)  # BCE
                    lcls += self.BCEcls(ps[:, 5:class_index], t)  # BCE

                # theta Classification by Circular Smooth Label
                t_theta = tgaussian_theta[i].type(ps.dtype) # target theta_gaussian_labels
                ltheta += self.BCEtheta(ps[:, class_index:], t_theta)
                
                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        ltheta *= self.hyp['theta']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + ltheta
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        """
        Args:
            p (list[P3_out,...]): torch.Size(b, self.na, h_i, w_i, self.no), self.na means the number of anchors scales
            targets (tensor): (n_gt_all_batch, [img_index clsid cx cy l s theta gaussian_θ_labels]) pixel
        Return：non-normalized data
            tcls (list[P3_out,...]): len=self.na, tensor.size(n_filter2)
            tbox (list[P3_out,...]): len=self.na, tensor.size(n_filter2, 4) featuremap pixel
            indices (list[P3_out,...]): len=self.na, tensor.size(4, n_filter2) [b, a, gj, gi]
            anch (list[P3_out,...]): len=self.na, tensor.size(n_filter2, 2)
            tgaussian_theta (list[P3_out,...]): len=self.na, tensor.size(n_filter2, hyp['cls_theta'])
            # ttheta (list[P3_out,...]): len=self.na, tensor.size(n_filter2)
        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # ttheta, tgaussian_theta = [], []
        tgaussian_theta = []
        # gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        feature_wh = torch.ones(2, device=targets.device)  # feature_wh
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # targets (tensor): (n_gt_all_batch, c) -> (na, n_gt_all_batch, c) -> (na, n_gt_all_batch, c+1)
        # targets (tensor): (na, n_gt_all_batch, [img_index, clsid, cx, cy, l, s, theta, gaussian_θ_labels, anchor_index]])
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0], # tensor: (5, 2)
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i] 
            # gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain=[1, 1, w, h, w, h, 1, 1]
            feature_wh[0:2] = torch.tensor(p[i].shape)[[3, 2]]  # xyxy gain=[w_f, h_f]

            # Match targets to anchors
            # t = targets * gain # xywh featuremap pixel
            t = targets.clone() # (na, n_gt_all_batch, c+1)
            t[:, :, 2:6] /= self.stride[i] # xyls featuremap pixel
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # edge_ls ratio, torch.size(na, n_gt_all_batch, 2)
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare, torch.size(na, n_gt_all_batch)
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter; Tensor.size(n_filter1, c+1)

                # Offsets
                gxy = t[:, 2:4]  # grid xy; (n_filter1, 2)
                # gxi = gain[[2, 3]] - gxy  # inverse
                gxi = feature_wh[[0, 1]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1)).T
                l, m = ((gxi % 1. < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m)) # (5, n_filter1)
                t = t.repeat((5, 1, 1))[j] # (n_filter1, c+1) -> (5, n_filter1, c+1) -> (n_filter2, c+1)
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j] # (5, n_filter1, 2) -> (n_filter2, 2)
            else:
                t = targets[0] # (n_gt_all_batch, c+1)
                offsets = 0

            # Define, t (tensor): (n_filter2, [img_index, clsid, cx, cy, l, s, theta, gaussian_θ_labels, anchor_index])
            b, c = t[:, :2].long().T  # image, class; (n_filter2)
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            # theta = t[:, 6]
            gaussian_theta_labels = t[:, 7:-1]
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, -1].long()  # anchor indices 取整
            # indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            indices.append((b, a, gj.clamp_(0, feature_wh[1] - 1), gi.clamp_(0, feature_wh[0] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            # ttheta.append(theta) # theta, θ∈[-pi/2, pi/2)
            tgaussian_theta.append(gaussian_theta_labels)

        # return tcls, tbox, indices, anch
        return tcls, tbox, indices, anch, tgaussian_theta #, ttheta
