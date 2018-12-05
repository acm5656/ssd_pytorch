import torch
import torch.nn as nn
import l2norm
base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
class SSD(nn.Module):
    def __init__(self):
        super(SSD,self).__init__()
        #vgg-16模型
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        )
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv4_3 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv5_3 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=6,dilation=6),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv8_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=256,kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv8_2 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv9_1 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=128,kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv9_2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv10_1 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv10_2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv11_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv11_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        #特征层位置输出
        self.feature_map_loc_1 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=4*4,kernel_size=3,stride=1,padding=1)
        )
        self.feature_map_loc_2 = nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=6*4,kernel_size=3,stride=1,padding=1)
        )
        self.feature_map_loc_3 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=6*4,kernel_size=3,stride=1,padding=1)
        )
        self.feature_map_loc_4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=6*4,kernel_size=3,stride=1,padding=1)
        )
        self.feature_map_loc_5 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=4*4,kernel_size=3,stride=1,padding=1)
        )
        self.feature_map_loc_6 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=4*4,kernel_size=3,stride=1,padding=1)
        )
        #特征层类别输出
        self.feature_map_conf_1 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=4*21,kernel_size=3,stride=1,padding=1)
        )
        self.feature_map_conf_2 = nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=6*21,kernel_size=3,stride=1,padding=1)
        )
        self.feature_map_conf_3 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=6*21,kernel_size=3,stride=1,padding=1)
        )
        self.feature_map_conf_4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=6*21,kernel_size=3,stride=1,padding=1)
        )
        self.feature_map_conf_5 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=4*21,kernel_size=3,stride=1,padding=1)
        )
        self.feature_map_conf_6 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=4*21,kernel_size=3,stride=1,padding=1)
        )
        self.vgg = [
            self.conv1_1[0],self.conv1_1[1],self.conv1_2[0],self.conv1_2[1],
            self.maxpool1,
            self.conv2_1[0],self.conv2_1[1],self.conv2_2[0],self.conv2_2[1],
            self.maxpool2,
            self.conv3_1[0],self.conv3_1[1],self.conv3_2[0],self.conv3_2[1],self.conv3_3[0],self.conv3_3[1],
            self.maxpool3,
            self.conv4_1[0],self.conv4_1[1],self.conv4_2[0],self.conv4_2[1],self.conv4_3[0],self.conv4_3[1],
            self.maxpool4,
            self.conv5_1[0],self.conv5_1[1],self.conv5_2[0],self.conv5_2[1],self.conv5_3[0],self.conv5_3[1],
            self.maxpool5,
            self.conv6[0],self.conv6[1],
            self.conv7[0],self.conv7[1]
        ]
        self.vgg = nn.ModuleList(self.vgg)
    #正向传播过程
    def forward(self, image):
        out = self.conv1_1(image)
        out = self.conv1_2(out)
        out = self.maxpool1(out)
        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.maxpool2(out)
        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.conv3_3(out)
        out = self.maxpool3(out)
        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.conv4_3(out)
        my_L2Norm = l2norm.L2Norm(512, 20)
        feature_map_1 = out
        feature_map_1 = my_L2Norm(feature_map_1)
        loc_1 = self.feature_map_loc_1(feature_map_1).permute((0,2,3,1)).contiguous()
        conf_1 = self.feature_map_conf_1(feature_map_1).permute((0,2,3,1)).contiguous()
        out = self.maxpool4(out)
        out = self.conv5_1(out)
        out = self.conv5_2(out)
        out = self.conv5_3(out)
        out = self.maxpool5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        feature_map_2 = out
        loc_2 = self.feature_map_loc_2(feature_map_2).permute((0,2,3,1)).contiguous()
        conf_2 = self.feature_map_conf_2(feature_map_2).permute((0,2,3,1)).contiguous()
        out = self.conv8_1(out)
        out = self.conv8_2(out)
        feature_map_3 = out
        loc_3 = self.feature_map_loc_3(feature_map_3).permute((0,2,3,1)).contiguous()
        conf_3 = self.feature_map_conf_3(feature_map_3).permute((0,2,3,1)).contiguous()
        out = self.conv9_1(out)
        out = self.conv9_2(out)
        feature_map_4 = out
        loc_4 = self.feature_map_loc_4(feature_map_4).permute((0,2,3,1)).contiguous()
        conf_4 = self.feature_map_conf_4(feature_map_4).permute((0,2,3,1)).contiguous()
        out = self.conv10_1(out)
        out = self.conv10_2(out)
        feature_map_5 = out
        loc_5 = self.feature_map_loc_5(feature_map_5).permute((0,2,3,1)).contiguous()
        conf_5 = self.feature_map_conf_5(feature_map_5).permute((0,2,3,1)).contiguous()
        out = self.conv11_1(out)
        out = self.conv11_2(out)
        feature_map_6 = out
        loc_6 = self.feature_map_loc_6(feature_map_6).permute((0,2,3,1)).contiguous()
        conf_6 = self.feature_map_conf_6(feature_map_6).permute((0,2,3,1)).contiguous()
        loc_list = [loc_1,loc_2,loc_3,loc_4,loc_5,loc_6]
        conf_list = [conf_1,conf_2,conf_3,conf_4,conf_5,conf_6]
        return loc_list,conf_list
