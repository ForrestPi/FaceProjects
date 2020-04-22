# QualityFace


Quality Face model which decides how suitable of an input face for face recognition system

Note that `qualityface` is a **helper model** for a face recognition system, its input shape should be **_exactly `128x128x3`_**.

## Getting Started

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

## Usage

```sh
import qualityface
path = 'path/to/your/img'
score = qualityface.estimate(path)
print(score)
```

## Results
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

