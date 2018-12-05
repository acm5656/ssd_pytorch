import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax
def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
class LossFun(nn.Module):
    def __init__(self):
        super(LossFun,self).__init__()
    def forward(self, prediction,targets,priors_boxes):
        loc_data , conf_data = prediction
        loc_data = torch.cat([o.view(o.size(0),-1,4) for o in loc_data] ,1)
        conf_data = torch.cat([o.view(o.size(0),-1,21) for o in conf_data],1)
        priors_boxes = torch.cat([o.view(-1,4) for o in priors_boxes],0)
        # batch_size
        batch_num = loc_data.size(0)
        # default_box数量
        box_num = loc_data.size(1)
        # 存储targets根据每一个prior_box变换后的数据
        target_loc = torch.Tensor(batch_num,box_num,4)
        target_loc.requires_grad_(requires_grad=False)
        # 存储每一个default_box预测的种类
        target_conf = torch.LongTensor(batch_num,box_num)
        target_conf.requires_grad_(requires_grad=False)
        for batch_id in range(batch_num):
            target_truths = targets[batch_id][:,:-1].data
            target_labels = targets[batch_id][:,-1].data
            utils.match(0.5,target_truths,priors_boxes,target_labels,target_loc,target_conf,batch_id)
        pos = target_conf > 0
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # 相当于论文中L1损失函数乘xij的操作
        pre_loc_xij = loc_data[pos_idx].view(-1,4)
        tar_loc_xij = target_loc[pos_idx].view(-1,4)
        loss_loc = F.smooth_l1_loss(pre_loc_xij,tar_loc_xij,size_average=False)

        batch_conf = conf_data.view(-1,21)

        loss_c = utils.log_sum_exp(batch_conf) - batch_conf.gather(1, target_conf.view(-1, 1))

        loss_c = loss_c.view(batch_num, -1)
        loss_c[pos] = 0

        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(3*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        test = pos_idx+neg_idx
        test1 = (pos_idx+neg_idx).gt(0)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, 21)
        targets_weighted = target_conf[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum().double()
        loss_l = loss_loc.double()
        loss_c = loss_c.double()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
