import Config
from itertools import product as product
from math import sqrt as sqrt
import torch
def default_prior_box():
    mean_layer = []
    for k,f in enumerate(Config.feature_map):
        mean = []
        for i,j in product(range(f),repeat=2):
            f_k = Config.image_size/Config.steps[k]
            cx = (j+0.5)/f_k
            cy = (i+0.5)/f_k

            s_k = Config.sk[k]/Config.image_size
            mean += [cx,cy,s_k,s_k]

            s_k_prime = sqrt(s_k * Config.sk[k+1]/Config.image_size)
            mean += [cx,cy,s_k_prime,s_k_prime]
            for ar in Config.aspect_ratios[k]:
                mean += [cx, cy, s_k * sqrt(ar), s_k/sqrt(ar)]
                mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        mean = torch.Tensor(mean).view( Config.feature_map[k],Config.feature_map[k],-1).contiguous()
        mean.clamp_(max=1, min=0)
        mean_layer.append(mean)

    return mean_layer

def change_prior_box(box):
    return torch.cat((box[:, :2] - box[:, 2:]/2,     # xmin, ymin
                     box[:, :2] + box[:, 2:]/2), 1)  # xmax, ymax

# 计算两个box的交集
def insersect(box1,box2):
    label_num = box1.size(0)
    box_num = box2.size(0)
    max_xy = torch.min(
        box1[:,2:].unsqueeze(1).expand(label_num,box_num,2),
        box2[:,2:].unsqueeze(0).expand(label_num,box_num,2)
    )
    min_xy = torch.max(
        box1[:,:2].unsqueeze(1).expand(label_num,box_num,2),
        box2[:,:2].unsqueeze(0).expand(label_num,box_num,2)
    )
    inter = torch.clamp((max_xy-min_xy),min=0)
    return inter[:,:,0]*inter[:,:,1]

# 计算jaccard
def jaccard(box1,box2):
    inter = insersect(box1,box2)
    area_1 = (
        (box1[:,2]-box1[:,0])
        *
        (box1[:,3]-box1[:,1])
    ).unsqueeze(1).expand_as(inter)
    area_2 = (
        (box2[:,2]-box2[:,0])
        *
        (box2[:,3]-box2[:,1])
    ).unsqueeze(0).expand_as(inter)
    union = area_1+area_2 - inter
    return inter/union

def encode(match_boxes,prior_box,variances):
    g_cxcy = (match_boxes[:, :2] + match_boxes[:, 2:])/2 - prior_box[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * prior_box[:, 2:])
    # match wh / prior wh
    g_wh = (match_boxes[:, 2:] - match_boxes[:, :2]) / prior_box[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

# 计算每一个box对应的类别和与prior_box变换后的数值
def match(threshold,target_truth,prior_box,target_lable,target_loc,target_conf,batch_id):
    # [label_num,box_num]
    overlaps = jaccard(target_truth,change_prior_box(prior_box))
    # 每一个类别中最大的overlap [label_num,1]
    best_prior_overlap,best_prior_idx = overlaps.max(1,keepdim = True)
    # 每个default box中最优的label [1,box_number]
    best_label_overlap,best_label_idx = overlaps.max(0,keepdim = True)
    best_prior_overlap.squeeze_(1)
    best_prior_idx.squeeze_(1)
    best_label_overlap.squeeze_(0)
    best_label_idx.squeeze_(0)
    best_label_overlap.index_fill_(0,best_prior_idx,2)
    for j in range(best_prior_idx.size(0)):
        best_label_idx[best_prior_idx[j]] = j
    match_boxes = target_truth[best_label_idx]
    conf = target_lable[best_label_idx]+1
    conf[best_label_overlap<threshold] = 0
    loc = encode(match_boxes,prior_box,[0.1,0.2])
    target_loc[batch_id] = loc
    target_conf[batch_id] = conf


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    result = torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max

if __name__ == '__main__':
    mean = default_prior_box()
    print(mean)