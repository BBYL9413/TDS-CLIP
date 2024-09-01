

# TDS-CLIP for Video Understanding
This is the official repo of the paper TDS-CLIP: Temporal Difference Side Network for Image-to-Video Transfer Learning


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tds-clip-temporal-difference-side-network-for/action-recognition-in-videos-on-something-1)](https://paperswithcode.com/sota/action-recognition-in-videos-on-something-1?p=tds-clip-temporal-difference-side-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tds-clip-temporal-difference-side-network-for/action-recognition-in-videos-on-something)](https://paperswithcode.com/sota/action-recognition-in-videos-on-something?p=tds-clip-temporal-difference-side-network-for)


## Requirement
- PyTorch >= 1.9
- RandAugment
- pprint
- dotmap
- yaml
- einops

## Data Preparation
**(Recommend)** To train all of our models, we extract videos into frames for fast reading. Please refer to [mmaction2](https://mmaction2.readthedocs.io/en/latest/user_guides/prepare_dataset.html) repo for the detaied guide of data processing.  
The annotation file is a text file with multiple lines, and each line indicates the directory to frames of a video, total frames of the video and the label of a video, which are split with a whitespace. Here is the format: 
```sh
<video_1> <total frames> <label_1>
<video_2> <total frames> <label_2>
...
<video_N> <total frames> <label_N>
```

**(Optional)** We can also decode the videos in an online fashion using [decord](https://github.com/dmlc/decord). This method is recommended if your dataset is not on a high-speed hard drive. Example of annotation:
```sh
<video_1> <label_1>
<video_2> <label_2>
...
<video_N> <label_N>
```
## Model Zoo

Here we provide some off-the-shelf pre-trained checkpoints of our models in the following tables. More checkpoints will be provided soon.

#### Kinetics-400

| Backbone |#Frame x crops x clips |  Top-1 Acc.(%) | Download |
|:------------:|:-------------------:|:------------------:|:-----------------:|
| ViT-B/16 | 8x3x4 | 83.9 | [Log](https://huggingface.co/BBLY9413/TDS-CLIP/tree/main) / [Checkpoint](https://huggingface.co/BBLY9413/TDS-CLIP/tree/main) |

#### Something-Something V1

| Backbone |#Frame x crops x clips |  Top-1 Acc.(%) | Download |
|:------------:|:-------------------:|:------------------:|:-----------------:|
| ViT-B/16 | 8x3x2 | 60.1 | [Log](https://huggingface.co/BBLY9413/TDS-CLIP/tree/main) / [Checkpoint](https://huggingface.co/BBLY9413/TDS-CLIP/tree/main) |
#### Something-Something V2

| Backbone |#Frame x crops x clips |  Top-1 Acc.(%) | Download |
|:------------:|:-------------------:|:------------------:|:-----------------:|
| ViT-B/16 | 8x3x2 | 71.8 | [Log](https://huggingface.co/BBLY9413/TDS-CLIP/tree/main) / [Checkpoint](https://huggingface.co/BBLY9413/TDS-CLIP/tree/main) |
| ViT-L/14 | 8x3x2 | 73.4 | [Log](https://huggingface.co/BBLY9413/TDS-CLIP/tree/main) / [Checkpoint](https://huggingface.co/BBLY9413/TDS-CLIP/tree/main) |

## Train
- After Data Preparation, you will need to download the CLIP weights from [OpenAI](https://github.com/openai/CLIP?tab=readme-ov-file), and place them in the `clip_pretrain` folder.
```sh
# For example, fine-tuning on Something-Something V1 using the following command:
sh scripts/run_train_vision.sh configs/sthv1/sthv1_train_rgb_vitb-16-side4video.yaml
```

## Test
- Run the following command to test the model.
```sh
sh scripts/run_test_vision.sh configs/sthv1/sthv1_train_rgb_vitb-16-side4video.yaml exp_onehot/ssv1/model_best.pt --test_crops 3 --test_clips 2
```
## Acknowledgment
Our implementation is mainly based on the following codebases. We are sincerely grateful for their work!
- [Side4Video](https://github.com/HJYao00/Side4Video): Side4Video: Spatial-Temporal Side Network for Memory-Efficient Image-to-Video Transfer Learning.
