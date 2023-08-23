# CAPGrasp

This is a Pytorch implementation of CAPGrasp: An SO(2) Equivariant Continuous Approach-Constrained Generative Grasp Sampler.
In this repository, we provide the CAPGrasp Sampler, Evaluator, constrained grasp refinement together with the pre-trained models.
Besides, we also provide the the object set and grasp dataset which are used for training.

We will release the isaacgym environment that we used for evaluation soon.

## Installation

The code has been tested with Python3.6, Pytorch 1.10, and CUDA 10.0 on Ubuntu 18.04. The trained model also been tested with the 
higher version of CUDA on Ubuntu 20. Here's the instruction of environment setup.


```shell
1. git@github.com:wengzehang/CAPGrasp.git && cd CAPGrasp
2. conda create -n capnet python=3.6
3. conda activate capnet
4. pip install --upgrade pip setuptools wheel
5. pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
6. git clone git@github.com:erikwijmans/Pointnet2_PyTorch.git
7. cd Pointnet2_PyTorch && pip install -r requirements.txt
8. cd .. && pip install -r requirements.txt 
9. pip install trimesh==3.14.1
10. download issacgym package
11. cd isaacgym/python && pip install -e .

```


## Grasp and Object Mesh Dataset

You can download the grasps through Google Drive:
https://drive.google.com/drive/folders/1D-7twxwE-PZ1QmVei5PsCHeKXhZopMLN?usp=sharing

Same for the mesh Dataset:
https://drive.google.com/file/d/1BITM0ntPUNTIWthFdveoUMMblH9zaAOp/view?usp=drive_link


## Pre-trained Model:

Please follow checkpoints_2d/download_checkpoints_instruction.md to download the pretrained models. If you want to train your own sampler and evaluator, we provide examples belows.

## Training

Sampler tranining:

```shell

python train.py \
--dataset_root_folder $LOCALDATAPATH/grasps \
--orientation_constrained \
--clusters -prpc \
--checkpoints_dir ./checkpoints_2d \
--extra_name continuous_equi_sampler \
--print_freq 4800 \
--save_latest_freq 250 \
--save_epoch_freq 1 \
--run_test_freq 1 \
--num_grasps_per_object 100 \
--num_objects_per_batch 3 \
--niter_decay 300 \
--niter 10 \
--latent_size 4 \
--gpu_ids 0,1,2 \
--equivariant

```

where the `$LOCALDATAPATH` is the path to the Acronym dataset you downloaded.

Tensorboard log file is store in checkpoints_2d directory for monitoring.

Evaluator training:

```shell

python train.py \
--dataset_root_folder /media/zehang/LaCie/zehang/ubuntu/dataset/cong_orien_grasp_full/grasps \
--arch evaluator \
--clusters -prpc \
--checkpoints_dir ./checkpoints_2d \
--extra_name shared \
--print_freq 4800 \
--save_latest_freq 250 \
--save_epoch_freq 1 \
--run_test_freq 1 \
--num_grasps_per_object 50 \
--num_objects_per_batch 4 \
--niter_decay 300 \
--niter 10 \
--gpu_ids 0

```

## Generate refined grasps

[coming soon]

## Evaluation in IsaacGym

[coming soon]