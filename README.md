# FAA-Net: A Frequency-Aware Attention Network for Single-Frame Infrared Small Target Detection
Shuo Zhuang, Yongxing Hou, Meibin Qi, and Di Wang
## Introduction
Existing detection methods primarily focus on extracting local spatial features, while overlooking the differences between targets and background in the frequency domain. To address this limitation, we propose a Frequency-Aware Attention Network (FAA-Net) for IRSTD.

First,we explore the possibility of transforming feature maps into the frequency domain and then propose 
the innovative FAA-Net for IRSTD. 

Second, we design a Transformer and Convolution Fusion 
Block (TCFB) to capture long-range relationships between the target and background while also 
extracting local edge details of small targets, and integrates feature fusion at each layer of the 
network. 

Third, we apply the DCT to convert spatial feature maps into the frequency domain and 
construct a Frequency-aware Attention Module (FAM) that effectively suppresses noise interference 
and enhances the features of small targets.

## Overall Framework
![image](https://github.com/Dr-Zhuang/FAA-Net-IRSTD/blob/main/framework.png)

## Usage

### 1.Datasets
Our project has the following structure:

datasets/NUDT(NUAA/IRSTD) 

--Train_Folder
               
               --img
               
               --labelcol

--Val_Folder
               
               --img
               
               --labelcol

--Test_Folder
               
               --img
               
               --labelcol


### 2.Configuration parameters in config.py

### 3.train

modify model_type 

python train_FAANet.py
