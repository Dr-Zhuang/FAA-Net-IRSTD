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
For example, to train the FAA model on the NUAA-SIRST dataset, modify the parameters in config.py accordingly.
```
task_name = 'NUAA'
learning_rate = 1e-4
batch_size = 8
model_name = 'FAA'
```
Then, begin training: 

python train_FAANet.py

### 3.test and demo
For example, to test the FAA model using the weight of NUAA-SIRST dataset, modify the parameters in config.py accordingly.
```
task_name = 'NUAA'
learning_rate = 1e-4
batch_size = 8
model_name = 'FAA'

test_session = "nuaa_Weight"##FAA weight
```
Then, begin testing: 

python test_FAANet.py

## Contact
**Welcome to raise issues or email to [shuozhuang@hfut.edu.cn](shuozhuang@hfut.edu.cn) for any question regarding our work.**
