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

The IRSTD dataset, which combines **NUAA-SIRST**, **NUDT-SIRST**, and **IRSTD-1K**, is used.
* **NUAA-SIRST** &nbsp; [[download]](https://github.com/YimianDai/sirst) &nbsp; [[paper]](https://arxiv.org/pdf/2009.14530.pdf)
* **NUDT-SIRST** &nbsp; [[download]](https://github.com/YeRen123455/Infrared-Small-Target-Detection) &nbsp; [[paper]](https://ieeexplore.ieee.org/abstract/document/9864119)
* **IRSTD-1K** &nbsp; [[download dir]](https://github.com/RuiZhang97/ISNet) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/9880295)
  
Our project has the following structure:
```
├──./datasets/
  │    ├── IRSTD
  │    │    ├── Train_Folder
  │    │    │    ├── imng
  │    │    │    ├── labelcol
  │    │    ├── Val_Folder
  │    │    │    ├── imng
  │    │    │    ├── labelcol
  │    │    ├── Test_Folder
  │    │    │    ├── imng
  │    │    │    ├── labelcol
  │    ├── NUDT
  │    │    ├── Train_Folder
  │    │    │    ├── imng
  │    │    │    ├── labelcol
  │    │    ├── Val_Folder
  │    │    │    ├── imng
  │    │    │    ├── labelcol
  │    │    ├── Test_Folder
  │    │    │    ├── imng
  │    │    │    ├── labelcol
  │    ├── NUAA
  │    │    ├── Train_Folder
  │    │    │    ├── imng
  │    │    │    ├── labelcol
  │    │    ├── Val_Folder
  │    │    │    ├── imng
  │    │    │    ├── labelcol
  │    │    ├── Test_Folder
  │    │    │    ├── imng
  │    │    │    ├── labelcol
 ```

### 2.Configuration parameters in config.py

### 3.train
For example, to train the FAA model on the NUDT-SIRST dataset, modify the parameters in config.py accordingly.
```
task_name = 'NUDT'
learning_rate = 1e-4
batch_size = 8
model_name = 'FAA'
```
Then, begin training: 

python train_FAANet.py

### 3.test and evaluation
For example, to test the FAA model using the weight of NUDT-SIRST dataset, modify the parameters in config.py accordingly.
```
task_name = 'NUDT'
learning_rate = 1e-4
batch_size = 8
model_name = 'FAA'
```
And, modify the parameters in test_FAANet.py accordingly.
```
test_num = 664
model_path = "nudt.pth.tar"
```
Then, begin testing: 
```
python test_FAANet.py
```
### 4.Demo
```
python demo.py --img /path/to/img.png --weights nudt.pth.tar
```
For example,
```
python demo.py --img 000133.png --weights nudt.pth.tar
```
The predicted results will be saved in the "single_pred" folder.

The well-trained weight (NUAA-SIRST) can be obtained from the following link:
1) https://pan.baidu.com/s/1IylcnukpmLVNJ9FA4OVXgQ?pwd=v5ce 提取码: v5ce

2) https://drive.google.com/file/d/1xOmBA5CIDKfepr5UTissTbRTZdL1ooXm/view?usp=sharing

The well-trained weight (NUDT-SIRST) can be obtained from the following link:
1) https://pan.baidu.com/s/1vgcmo_IUayx4hbSU5H0u3Q?pwd=uabr 提取码: uabr

2) https://drive.google.com/file/d/19ysdlzZpAdQaeOtJPBtn3f_gjjd8ayLl/view?usp=sharing

The well-trained weight (IRSTD-1K) can be obtained from the following link:
1) https://pan.baidu.com/s/1lrJIIgqc19d3plBiakkhfg?pwd=h3bc 提取码: h3bc

2) https://drive.google.com/file/d/1fKA_ulrhuTAHOEINE10swisFuY-SH8hX/view?usp=sharing
   
## Contact
**Welcome to raise issues or email to [shuozhuang@hfut.edu.cn](shuozhuang@hfut.edu.cn) for any question regarding our work.**
