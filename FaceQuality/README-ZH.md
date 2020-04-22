# QualityFace


人脸质量模型，可用于评估一张图片是否适合用作人脸识别系统的输入。

请注意，`qualityface`仅是一个辅助模型，其输入必须刚好是 **_128x128x3_**。

## 快速开始

+ Python3.6 (test in 3.6, 3.7)
+ Pytorch (test in 1.3.1, 1.4.0)
+ opencv4 (test in 4.1.1)
+ pillow (test in 6.2.0)
+ numpy
+ siriusbackbone

install from PyPI:
```sh
pip install qualityface
```

## 用法

```sh
import qualityface
path = 'path/to/your/img'
score = qualityface.estimate(path)
print(score)
```

## 结果
<img src="test/crop1.jpeg">
<img src="test/crop2.jpeg">
<img src="test/crop3.jpeg">
<img src="test/crop4.jpeg">
<img src="test/crop5.jpeg">

```sh
Test: crop1.jpeg, score: 0.78
Test: crop2.jpeg, score: 0.7
Test: crop3.jpeg, score: 0.74
Test: crop4.jpeg, score: 0.92
Test: crop5.jpeg, score: 0.93
```

这个模型对于1）遮挡、2）非正脸、3）模糊、4）阴影、5）正常 都有比较好的判别度。

