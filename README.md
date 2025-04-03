# YOLOFT: [An Extremely Small Video Object Detection Benchmark](https://gjhhust.github.io/XS-VID/) Baseline

## :loudspeaker: Introduction
This is the official implementation of the baseline model for [XS-VID](https://gjhhust.github.io/XS-VID/) benchmark.

[news]: We will soon be releasing XS-VIDv2, incorporating many new videos and scenarios, significantly expanding our dataset! Please stay tuned!

## :ferris_wheel: Dependencies
 - CUDA 11.7
 - Python 3.8
 - PyTorch 1.12.1(cu116)
 - TorchVision 0.13.1(cu116)
 - numpy 1.24.4

## :open_file_folder: Datasets
Our work is based on the large-scale extremely small video object detection benchmark **XS-VID**. Download the dataset(s) from corresponding links below.
- [Google drive]：[annotations](https://drive.google.com/file/d/1-MF_H6OnLL-6ZAHwmwTOdxIeKY9zqGO9/view?usp=sharing); [images(0-3)](https://drive.google.com/drive/folders/1EGTIWLCLUAlKfbq7KEeHqXL8PAyKHNQ_?usp=sharing); [images(4-5)](https://drive.google.com/drive/folders/1m7YL3XVDjmiiVEy_rY4gVr0tJxnn8e0Y?usp=sharing);
- [BaiduNetDisk]：[annotations and images](https://pan.baidu.com/s/1VXle03mUYpKtmp3xj6C4dA?pwd=yp5g);

Please choose a download method to download the annotations and all images. Make sure all the split archive files (e.g., `images.zip`, `images.z01`, `images.z02`, etc.) are in the same directory. Use the following command to extract them:

```bash
unzip images.zip
unzip annotations.zip
```
We have released several annotation formats to facilitate subsequent research and use, including COCO, COCOVID, YOLO

## 🛠️ Install
This repository is build on **[Ultralytics](https://github.com/ultralytics/ultralytics) 8.0.143**  which can be installed by running the following scripts. Please ensure that all dependencies have been satisfied before setting up the environment.
```
scp -r -P 2026 /data/jiahaoguo/datasets/gaode_6/annotations/yolo/gaode_6_rm198_exclude14569*  jiahaoguo@115.156.158.8:/data/jiahaoguo/dataset/gaode_6/annotations/yolo/


conda create --name yoloft python=3.10
conda activate yoloft
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/gjhhust/YOLOFT
cd YOLOFT
pip install -r requirements.txt 
pip install -e .
pip install -U openmim
mim install mmcv
pip install mmcv-full

cd ./ultralytics/nn/modules/ops_dcnv3
python setup.py build install

cd ../alt_cuda_corr_sparse
python setup.py build install
```

## :hourglass: Data preparation

If you want to use a custom video dataset for training tests, it needs to be converted to yolo format for annotation, and the dataset files are organized in the following format:

```
data_root_dir/               # Root data directory
├── test.txt                 # List of test data files, each line contains a relative path to an image file
├── train.txt                # List of training data files, each line contains a relative path to an image file
├── images/                  # Directory containing image files
│   ├── video1/              # Directory for image files of the first video
│   │   ├── 0000000.png      # First frame image file of the first video
│   │   └── 0000001.png      # Second frame image file of the first video
│   ├── video2/              # Directory for image files of the second video
│   │   └── ...              # More image files
│   └── ...                  # More video directories
└── labels/                  # Directory containing label files
    ├── video1/              # Directory for label files of the first video
    │   ├── 0000000.txt      # Label file for the first frame of the first video (matches the image file)
    │   └── 0000001.txt      # Label file for the second frame of the first video (matches the image file)
    ├── video2/              # Directory for label files of the second video
    │   └── ...              # More label files
    └── ...                  # More video directories
```


Note: The name of the image and the name of the label in yolo format must be the same, and the format is frameNumber.png, e.g. "0000001.png and 0000001.txt".

## 数据集准备
1. XS-VID（视频训练和单帧训练都行）：
通过该网页直接下载可以
https://modelscope.cn/datasets/lanlanlanrr/XS-VID/files

之后修改一下config/dataset/XS-VIDv2.yaml配置文件中的数据集地址即可

单帧训练：split_length: [1]   train_slit: [0] #从0epoch开始就1帧训练
n帧传递loss训练 split_length: [n]  train_slit: [0] #从0epoch开始就n帧训练
在一次训练中变化帧训练 split_length: [1, n]   train_slit: [0, 10] #意思是先1帧训练，然后10epoch时进行n帧训练


在注意训练的batch_size需要设置在XS-VIDv2.yaml
split_batch_dict:
  1: 32
  2: 15 #s1
  3: 8
  4: 6
  5: 5
  6: 4
  8: 7
意思是1帧训练时，batch设置为32。2帧时自动设置为15。以此类推。

2. coco数据集
ultralytics/data/scripts/get_coco.sh内自带脚本

3. 

## 代码结构简述
yoloft代码基于yolo构建，新增如下
1. 视频帧训练（n个帧计算一次loss0）
2. video的dataset和sampler（详情见models/yoloft/detect/train.py）中如何build dataset
3. 模型结构配置文件 config/yoloft_onxx/yoloftS_dcn_dy_s1_t.yaml 内目前小目标的速度和效率兼具比较好，将C2f换成C2f_DCNv3效果会提升很多，但是速度会下降，取折中得
4. mosia数据增强对于视频训练还是很有用的，目前视频训练一次视频太多会掉点，推测是bbox传递梯度比较差，segment标注来训练视频模块比较好
5. 现在训练阶段 （1）coco预训练200e -> 
              （2）小目标的RGB超大混合数据集预训练100e -> 
              （3）（目前跳过）（还没改好代码）改成segment头，设置其他层学习率低，设置视频模块学习率高进行学习n epoch
              （4）-> 最后一步就是业务数据微调