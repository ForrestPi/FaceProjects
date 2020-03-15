# Build Your Own Face Recognition Model

训练你自己的人脸识别模型！

人脸识别从原始的 Softmax Embbedding，经过2015年 Facenet 领衔的 triple loss metric learning，然后是 additional margin metric learning。这次的系列博客实现的是2018年提出的 ArcFace 。


### 依赖
```py
Python >= 3.6
pytorch >= 1.0
torchvision
imutils
pillow == 6.2.0
tqdm
```

### 数据准备

+ 下载WebFace（百度一下）以及干净的图片列表用于训练
+ 下载LFW用于测试
+ 删除WebFace中的脏数据，使用`utils.py`

### 配置参数

见`config.py`

### 训练

支持简单的单机多GPU训练

```python
export CUDA_VISIBLE_DEVICES=0,1
python train.py
```

### 测试

```py
python test.py
```

### 博客


+ 001 [数据准备](./blog/data.md)
+ 002 [模型架构](./blog/model.md)
+ 003 [损失函数](./blog/loss.md)
+ 004 [度量函数](./blog/metric.md)
+ 005 [训练](./blog/train.md)
+ 006 [测试](./blog/test.md)

### 参考


+ [insightFace](https://github.com/deepinsight/insightface/tree/master/recognition)
+ [insightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)
+ [arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch)
