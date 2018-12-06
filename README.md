# ssd_pytorch  
首先说下实现原因，大佬写的代码太难读了，对于新手太不友好，而且还是英文的，让我这英语渣渣看起来头疼，为了学习和帮助其他像我一样的人，复现了一下代码，并简化部分功能  
  
首先发下代码的参考连接 [ssd-pytorch](https://github.com/amdegroot/ssd.pytorch)  
论文地址就不发了，可以自行百度，我的博客ssd解读地址在[这里](https://www.cnblogs.com/cmai/p/10076050.html)，可以进行参考学习  
代码解读地址在[这里](https://www.cnblogs.com/cmai/p/10080005.html)  
  
运行环境：  
>1.python 3.6  
>2.pytorch 0.4.1  
>3.python-opencv  
  
## 说明  
预训练的权重文件[vgg_16](https://pan.baidu.com/s/1t_kd5YfdFHlzIiLWlYNjIQ ) 提取码：rdbn  
具体的配置文件请看Config.py文件  
训练运行python3.6 Train.py  
测试运行python3.6 Test.py  
  
## 运行结果  
![demo](https://github.com/acm5656/ssd_pytorch/blob/master/result.jpg)
  
## 注：  
>1.目前的版本是作者学习ssd模型并参考他人代码尝试复现的结果，所以代码鲁棒性较低，望见谅  
>2.目前版本只兼容voc的数据集  
>3.如果有问题，可以发邮件到807603148@qq.com，作者尽量答复  
>4.如果帮到你啦，麻烦给个star，谢谢啦  


