# Drone-assisted Road Gaussian Splatting with Cross-view Uncertainty

[Saining Zhang](https://sainingzhang.github.io/), Baijun Ye, Xiaoxue Chen, [Yuantao Chen](https://tao-11-chen.github.io/), Zongzheng Zhang, Cheng Peng, Yongliang Shi, [Hao Zhao](https://sites.google.com/view/fromandto) <br />


[[`Project Page`](https://sainingzhang.github.io/project/uc-gs/)][[`arxiv`](https://arxiv.org/abs/2312.00109)]


## Overview

<p align="center">
<img src="assets/pipeline.png" width=100% height=100% 
class="center">
</p>


We introduce Scaffold-GS, which uses anchor points to distribute local 3D Gaussians, and predicts their attributes on-the-fly based on viewing direction and distance within the view frustum.

Our method performs superior on scenes with challenging observing views. e.g. transparency, specularity, reflection, texture-less regions and fine-scale details.

<p align="center">
<img src="assets/teaser_big.png" width=100% height=100% 
class="center">
</p>





## Installation

We tested on a server configured with Ubuntu 18.04, cuda 11.6 and gcc 9.4.0. Other similar configurations should also work, but we have not verified each one individually.

1. Clone this repo:

```
git clone https://github.com/SainingZhang/UC-GS.git --recursive
cd UC-GS
```

2. Install dependencies

```
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate uc_gs
```

## Data

The Synthetic dataset is available in [Google Drive].


## Training

```
bash ./single_train.sh
```

- scene: scene name with a format of ```dataset_name/scene_name/``` or ```scene_name/```;
- exp_name: user-defined experiment name;
- gpu: specify the GPU id to run the code. '-1' denotes using the most idle GPU. 
- voxel_size: size for voxelizing the SfM points, smaller value denotes finer structure and higher overhead, '0' means using the median of each point's 1-NN distance as the voxel size.
- update_init_factor: initial resolution for growing new anchors. A larger one will start placing new anchor in a coarser resolution.


## Evaluation

```
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

## Citation

If you find our work helpful, please consider citing:

```bibtex
@inproceedings{scaffoldgs,
  title={Scaffold-gs: Structured 3d gaussians for view-adaptive rendering},
  author={Lu, Tao and Yu, Mulin and Xu, Linning and Xiangli, Yuanbo and Wang, Limin and Lin, Dahua and Dai, Bo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20654--20664},
  year={2024}
}
```

## Related Work

[Scaffold-GS](https://github.com/city-super/Scaffold-GS)
