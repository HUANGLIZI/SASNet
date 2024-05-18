
### Requirements
This repository is based on PyTorch 1.8.2, CUDA 11.1 and Python 3.8.13; All experiments in our paper were conducted on a single NVIDIA A100 GPU.

### Data Preparation
1. The LA dataset is already available in './SASNet/data/LA' 
2. Download [Pancreas](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT)
   Put the data in './SASNet/data'
2. Download [BraTS](https://github.com/HiLab-git/SSL4MIS/tree/master/data/BraTS2019)
   Put the data in './SASNet/data'
### Usage
1. Clone the repo.;
```
git clone https://github.com/HUANGLIZI/SASNet.git
```
2. Train the model;
```
cd SDHNet
# for 20% labels on LA
python ./code/train.py --dataset_name LA  --labelnum 16 --gpu 0 --temperature 0.1
# for 10% labels on LA
python ./code/train.py --dataset_name LA  --labelnum 8 --gpu 0 --temperature 0.1

# e.g., for 20% labels on Pancreas_CT
python ./code/train.py --dataset_name Pancreas_CT  --labelnum 12 --gpu 0 --temperature 0.1
# e.g., for 10% labels on Pancreas_CT
python ./code/train.py --dataset_name Pancreas_CT  --labelnum 6 --gpu 0 --temperature 0.1

# e.g., for 20% labels on BraTS
python ./code/train.py --dataset_name BraTS  --labelnum 50 --gpu 0 --temperature 0.1
# e.g., for 10% labels on BraTS
python ./code/train.py --dataset_name BraTS  --labelnum 25 --gpu 0 --temperature 0.1
```
3. Test the model;
```
cd SASNet
# for 20% labels on LA
python ./code/test.py --dataset_name LA --labelnum 16 --gpu 0
# for 10% labels on LA
python ./code/test.py --dataset_name LA --labelnum 8 --gpu 0

# for 20% labels on Pancreas-CT
python ./code/test.py --dataset_name Pancreas_CT --labelnum 12 --gpu 0
# for 10% labels on Pancreas-CT
python ./code/test.py --dataset_name Pancreas_CT --labelnum 6 --gpu 0

# for 20% labels on BraTS
python ./code/test.py --dataset_name BraTS --labelnum 50 --gpu 0
# for 10% labels on BraTS
python ./code/test.py --dataset_name BraTS --labelnum 25 --gpu 0

```
### pre-trained models
We provide pre-trained models for LA and Pancreas, which can be found in './SASNet/pretrain', you can load the pre-trained model for training via the following command
```
cd SASNet
# for 20% labels on LA
python ./code/train.py --dataset_name LA  --labelnum 16 --gpu 0 --temperature 0.1 --model_type SASNet_pretrain
# for 10% labels on LA
python ./code/train.py --dataset_name LA  --labelnum 8 --gpu 0 --temperature 0.1 --model_type SASNet_pretrain

# e.g., for 20% labels on Pancreas_CT
python ./code/train.py --dataset_name Pancreas_CT  --labelnum 12 --gpu 0 --temperature 0.1 --model_type SASNet_pretrain
# e.g., for 10% labels on Pancreas_CT
python ./code/train.py --dataset_name Pancreas_CT  --labelnum 6 --gpu 0 --temperature 0.1 --model_type SASNet_pretrain

# e.g., for 20% labels on BraTS
python ./code/train.py --dataset_name BraTS  --labelnum 50 --gpu 0 --temperature 0.1 --model_type SASNet_pretrain
# e.g., for 10% labels on BraTS
python ./code/train.py --dataset_name BraTS  --labelnum 25 --gpu 0 --temperature 0.1 --model_type SASNet_pretrain

```
### result

| Dataset     | Labeled data     | Dice(%)     | Jaccard(%)     | HD95(voxel)     | ASD(voxel)     |
| -------- | -------- | -------- | -------- | -------- | -------- |
| LA | 8  | 89.62 | 81.33 | 6.59| 1.89 |
| LA | 16 | 91.82 | 84.93 | 4.63 | 1.42 |
| Pancreas-CT | 6 | 76.38 | 62.84 | 13.47 | 1.82 |
| Pancreas-CT | 12 | 81.60 | 69.39 | 11.25  | 1.81 |
| BraTS | 25 | 82.84 | 73.00 | 10.91 | 2.31 |
| BraTS | 50 | 85.84 | 76.69 | 7.52  | 1.62 |


### Acknowledgements:
Our code is adapted from [UAMT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [DTC](https://github.com/HiLab-git/DTC), [URPC](https://github.com/HiLab-git/SSL4MIS) , [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) and [MCNet](https://github.com/ycwu1997/MC-Net.git). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.


