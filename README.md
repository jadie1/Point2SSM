# Point2SSM
Implementation of "[Point2SSM: Learning Morphological Variations of Anatomies from Point Cloud](https://arxiv.org/abs/2305.14486)" spotlight presentation at ICLR 2024. Please cite this paper if you use the code. 

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

## Datastes 
The paper uses aligned versions of the spleen and pancreas Medical Decathlon public datasets.

### Original Medical Decathlon Data
The original, unaligned data is available here: http://medicaldecathlon.com/.
The data is available with a permissive copyright-license (CC-BY-SA 4.0), allowing for data to be shared, distributed and improved upon. All data has been labeled and verified by an expert human rater, and with the best effort to mimic the accuracy required for clinical use. To cite this data, please refer to https://arxiv.org/abs/1902.09063.

### Aligned Medical Decathlon Data
Alignment and pre-processing was performed using [ShapeWorks](https://www.sci.utah.edu/software/shapeworks.html/) mesh grooming tools. The aligned spleen dataset is available in this repo in the `data/spleen/` folder. The aligned version of the pancreas data can be downloaded at [https://www.shapeworks-cloud.org/#/](https://www.shapeworks-cloud.org/#/) with a free account. 

If you use either of these pre-processed datasets in work that leads to published research, we humbly ask that you cite ShapeWorks, and add the following to the 'Acknowledgments' section of your paper:
"The National Institutes of Health supported this work under grant numbers NIBIB-U24EB029011, NIAMS-R01AR076120, NHLBI-R01HL135568, NIBIB-R01EB016701, and NIGMS-P41GM103545."
and add the following 'disclaimer': "The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health."
When referencing this dataset groomed with ShapeWorks, please include a bibliographical reference to the paper below, and, if possible, include a link to [shapeworks.sci.utah.edu](https://www.sci.utah.edu/software/shapeworks.html/).
```
    @incollection{cates2017shapeworks,
    title = {Shapeworks: particle-based shape correspondence and visualization software},
    author = {Cates, Joshua and Elhabian, Shireen and Whitaker, Ross},
    booktitle = {Statistical Shape and Deformation Analysis},
    pages = {257--298},
    year = {2017},
    publisher = {Elsevier}
    }
```

## Model Training

To train a model, call `train.py` with the appropriate config yaml file. For example:
```
python train.py -c cfgs/point2ssm.yaml
```
Specific parameters are set in the config file, including dataset, noise level, learning rate, etc. 


## Model Testing
To test a model, call `test.py` with the appropriate config yaml file. For example:
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

