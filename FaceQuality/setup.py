import setuptools

setuptools.setup(
    name = "qualityface",
    version = "1.0.3",
    author = "sirius demon",
    author_email = "mory2016@126.com",
    description="Quality face in Pytorch",
    long_description="Quality Face model which decides how suitable of an input face for face recognition system",
    long_description_content_type='text/markdown',
    url = "https://github.com/siriusdemon/pytorch-QualityFace",
    packages=setuptools.find_packages(),
    package_data = {
        'qualityface': ['checkpoints/last.pth'],
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires = [
        'opencv-python',
        'numpy',
        'siriusbackbone',
        'pillow',
    ]
)
