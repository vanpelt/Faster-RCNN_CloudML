# Faster-RCNN for Cloud ML

This is a fork of [@smallcorgi](smallcorgi/Faster-RCNN_TF)'s experimental Tensorflow implementation of Faster RCNN made to run on [Google Cloud Machine Learning](https://cloud.google.com/products/machine-learning).  For details about R-CNN please refer to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497v3.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.


## Software Requirements

### Cloud

1. Google Cloud Project and local tools described [here](https://cloud.google.com/ml/docs/how-tos/getting-set-up)

### Local

VirtualEnv, Install tensorflow

1. Requirements for Tensorflow (see: [Tensorflow](https://www.tensorflow.org/))

2. CUDA and cudnn if you want to train locally

3. Build and install the python requirements locally

```Shell
cd lib
python setup.py build_ext --inplace
```

## Hardware Requirements

1. For training the end-to-end version of Faster R-CNN with VGG16, 3G of GPU memory is sufficient (using CUDNN)

## Installation (sufficient for the demo)

1. Clone the Faster R-CNN repository
  ```Shell
  git clone --recursive https://github.com/smallcorgi/Faster-RCNN_TF.git
  ```

2. Build the Cython modules
    ```Shell
    cd $FRCN_ROOT/lib
    python setup.py build_ext --inplace
    ```

## Demo

TBD Monkies...

## References

[Faster-RCNN_TF](smallcorgi/Faster-RCNN_TF)

[Faster R-CNN caffe version](https://github.com/rbgirshick/py-faster-rcnn)

[A tensorflow implementation of SubCNN (working progress)](https://github.com/yuxng/SubCNN_TF)