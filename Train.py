import torch
torch.manual_seed(1)
import ssd_net

import torch.nn as nn
import cv2
import utils
import loss_function
import voc0712
import augmentations
import Config
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets
def xavier(param):
    nn.init.xavier_uniform_(param)
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
def train():
    dataset = voc0712.VOCDetection(root=Config.dataset_root,
                           transform=augmentations.SSDAugmentation(Config.image_size,
                                                     Config.MEANS))
    data_loader = data.DataLoader(dataset, Config.batch_size,
                                  num_workers=Config.data_load_number_worker,
                                  shuffle=False, collate_fn=detection_collate,
                                  pin_memory=True)

    net = ssd_net.SSD()
    vgg_weights = torch.load('./vgg16_reducedfc.pth')
    net.apply(weights_init)
    net.vgg.load_state_dict(vgg_weights)
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=Config.lr, momentum=Config.momentum,
                          weight_decay=Config.weight_decacy)
    for epoch in range(1000):
        for step,(img,target) in enumerate(data_loader):
            img = torch.Tensor(img)
            loc_pre,conf_pre = net(img)
            priors = utils.default_prior_box()
            loss_fun = loss_function.LossFun()
            loss_l,loss_c = loss_fun((loc_pre,conf_pre),target,priors)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            print('epoch : ',epoch,' step : ',step,' loss : ',loss.data[0])
if __name__ == '__main__':
    train()



