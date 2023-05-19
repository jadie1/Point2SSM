# Point2SSM
Implementation of "Point2SSM: Anatomical Statistical Shape Models from Point Clouds"

This code includes the proposed Point2SSM model, as well as implementations of:
- PointNet Autoencoder from ["Learning Representations and Generative Models For 3D Point Clouds"](https://arxiv.org/abs/1707.02392)
- DGCNN Autoencoder from ["Dynamic Graph CNN for Learning on Point Clouds"](https://arxiv.org/abs/1801.07829)
- CPAE from ["Learning 3D Dense Correspondence via Canonical Point Autoencoder"](https://arxiv.org/abs/2107.04867)
- ISR from ["Unsupervised Learning of Intrinsic Structural Representation Points"](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Unsupervised_Learning_of_Intrinsic_Structural_Representation_Points_CVPR_2020_paper.pdf)
- DPC from ["DPC: Unsupervised Deep Point Correspondence via Cross and Self Construction"](https://arxiv.org/abs/2110.08636)


## Setup

To setup an anaconda environment, run:

```
source setup.sh
```

Aligned spleen and pancreas data, as well as pretrained Point2SSM models can be downloaded here: [Dropbox Link](https://www.dropbox.com/s/i0t6zpda0v9odrp/Point2SSM-Anonymous-Submission-20230519T161512Z-001.zip?dl=0)


## Model Training

To train a model, call `train.py` with the appropriate config yaml file. For example:
```
python train.py -c cfgs/point2ssm.yaml
```
Specific parameters are set in the config file, including dataset, noise level, learning rate, etc. 


## Model Testing
To test a model, call `test.py` with the appropriate config yaml file.
A pretrained model can be downloaded [here](https://www.dropbox.com/s/i0t6zpda0v9odrp/Point2SSM-Anonymous-Submission-20230519T161512Z-001.zip?dl=0), and then the following can be run:
```
python test.py -c experiments/spleen/point2ssm/point2ssm.yaml
```


## Acknowledgements
This code utilizes the following Pytorch 3rd-party libraries:
- [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)
- [emd](https://github.com/paul007pl/VRCNet)
- [PointNet](https://github.com/sshaoshuai/Pointnet2.PyTorch)

This code includes the following models:
- [PointNet Autoencoder](https://github.com/optas/latent_3d_points) 
- [DGCNN](https://github.com/WangYueFt/dgcnn)
- [CPAE](https://github.com/AnjieCheng/CanonicalPAE)
- [ISR](https://github.com/NolenChen/3DStructurePoints)
- [DPC](https://github.com/dvirginz/DPC)
- [PointAttN (SFA Block)](https://github.com/ohhhyeahhh/PointAttN)

The original, unaligned meshes are from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/).
