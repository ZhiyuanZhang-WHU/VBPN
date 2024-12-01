# Code for article " Deep Variational Network for Blind Pansharpening"

## Congratulations!!
Our work has been accepted by  IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS!!!!!!

[VBPN](https://ieeexplore.ieee.org/abstract/document/10632205)

## Introduction
Deep learning-based methods play an important role in pansharpening that utilizes panchromatic images to enhance the spatial resolution of multispectral images while maintaining spectral features. However, most existing methods mainly consider only one fixed degradation in the training process. Therefore, their performance may drop significantly when the degradation of testing data is unknown (blind) and different from the training data, which is common in real-world applications. To address this issue, we proposed a deep variational network for blind pansharpening, named VBPN, which integrates degradation estimation and image fusion into a whole Bayesian framework. First, by taking the noise and blurring parameters of the multispectral image with the noise parameters of the panchromatic image as hidden variables, we parameterize the approximate posterior distribution for the fusion problem using neural networks. Since all parameters in this posterior distribution are explicitly modeled, the degradation parameters of the multispectral image and the panchromatic image are easily estimated. Furthermore, we designed VPBN composed of degradation estimation and image fusion sub-networks, which can  optimize the fusion results guided by the variational inference according to the testing data. As a result, the blind pansharpening performance can be improved. In general, VPBN has good interpretability and generalization ability by combining the advantages of model-based and deep learning-based approaches.

![Difference between traditional deep learning-based pansharpening methods and our proposed variational blind pansharpening method. (a) Training and testing process of traditional deep learning-based pansharpening methods. (b) Training and testing process of our proposed variational pansharpening method.](https://github.com/ZhiyuanZhang-WHU/VBPN/blob/main/imgs/problem.png)

### Experimental results
![Visual results of pansharpening methods on simulated GaoFen-2 dataset with PAN noise level 30 (RGB Bands). (a) LR-MS image. (b) PNN. (c) DRPNN. (d) MSDCNN. (e) DiCNN. (f) FusionNet. (g) Hyper-DSNet. (h) ADKNet. (i) MSDDN. (j) BiMPan. (k) Proposed. (l) Ground Truth.](https://github.com/ZhiyuanZhang-WHU/VBPN/blob/main/imgs/Fig4.png)

### How to Use

### Dataset

The "dataset/pansharping/vpn.py" provides the simulation process of the dataset for training and testing.

The function "__get_image__(self, item)" can define the way of data reading. In this instance, the HRMS (High Resolution Multispectral) and PAN (Panchromatic) images are stored in .mat file. They can be indexed through the "ms_label" and "pan_label", and users can make adjustments according to their own needs.

The "__init__" provides various parameters regarding the degradation process of multispectral and panchromatic images.

### Train
Users can use the following commands to conduct training.
```python
python main.py --yaml /yaml/train_vpn.yaml
```
The "/yaml/train_vpn.yaml" provides configurations regarding the saving location of training logs, training datasets, testing datasets, and parameters related to degradation.

The following items in the project need to be configured by users according to their own requirements:

seed: 20000320 (random seed)

device: '1' (GPU number)

record_dir: 'Experiment' (name of the experiment folder)

note_name: '520_test' (name of the process folder for this training)

train:
target: "/home/zzy/pansharpening_data/Our_Dataset_norandom/TrainFolder" (folder of training data)

test:
target: "/home/zzy/pansharpening_data/Our_Dataset_norandom/TrainFolder" (folder of testing data)

### Test
Users can use the following commands to test.
```python
python main.py --yaml /yaml/test_vpn.yaml
```
The "/yaml/train_vpn.yaml" provides the definition of checkpoints, test folders, and definition of parameters related to the degradation of test data.

The following items in the project need to be configured by users according to their own requirements:

seed: 20000320 (random seed)

device: '1' (GPU number)

record_dir: 'Experiment' (name of the experiment folder)

note_name: '520_test' (name of the process folder for this training)

checkpoint : "your checkpoint or our initial checkpoint provided in'checkpoint/save_model/model_current_0346.pth'"


## Citation
```python
@ARTICLE{10632205,
  author={Zhang, Zhiyuan and Li, Haoxuan and Ke, Chengjie and Chen, Jun and Tian, Xin},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Deep Variational Network for Blind Pansharpening}, 
  year={2024},
  volume={},
  number={},
  pages={1-15},
  keywords={Pansharpening;Image fusion;Degradation;Feature extraction;Spatial resolution;Training;Learning systems;Blind pansharpening;image fusion;remote sensing;variational inference},
  doi={10.1109/TNNLS.2024.3436850}}
```
