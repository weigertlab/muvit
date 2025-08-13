# MuViT: MultiResolution MultiModal Transformer for large microscopy images 

**WORK IN PROGRESS - DO NOT  DISTRIBUTE**

This project implements a multi-resolution Vision Transformer (MuViT) architecture designed for processing large microscopy images. The model operates on multiple resolution levels simultaneously, allowing it to capture both fine-grained details and broader context.

![val_inout](https://github.com/user-attachments/assets/161e25b9-f234-432c-b431-74547b5056a7)
Example on Mouse sections, left to right: masked input, reconstruction, GT, error. top row: high resolution, bottom: low resolution 


## Refactor progress

- [x] Building blocks (RotaryEmbedding, TransformerLayer, SaveableModel)
- [x] MuViTEncoder (2d and 3d)
- [x] MuViTDecoder (2d and 3d)
- [x] MuViT
- [x] Lightning
- [x] Dataset/dataloading utilities
- [x] Installable package

## Installation

```bash
mamba create -y -n muvit python=3.11
pip install -e .
```

## Overview

## Dataloader

TODO

## MuViT Model

The model is designed to learn hierarchical representations across multiple resolution levels while maintaining spatial relationships through bounding box coordinates.

TODO

## Papers/resources

    - https://www.reddit.com/r/MachineLearning/comments/1b3bhbd/d_why_is_vit_more_commonly_used_than_swin/
    - https://medium.com/@14prakash/vitdet-the-go-to-architecture-for-image-foundation-models-3f5f44e6ac4a
    - Tian, Rui, et al. "Resformer: Scaling vits with multi-resolution training." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
    - ConvNets Match Vision Transformers at Scale https://arxiv.org/abs/2310.16764
    - https://frankzliu.com/blog/vision-transformers-are-overrated
    - Bachman et al MultiMAE: Multi-modal Multi-task Masked Autoencoders
    - GigaPath: https://github.com/prov-gigapath/prov-gigapath
    - Recursion RxRx3 paper


# Related work

    - HookNet: Multi-resolution convolutional neural networks for semantic segmentation in histopathology whole-slide images https://www.sciencedirect.com/science/article/pii/S1361841520302541?ref=pdf_download&fr=RR-2&rr=90fac81f08684516
    - Global-Local Transformer for Brain Age Estimation https://ieeexplore.ieee.org/abstract/document/9525077


# Datasets:

- histo/ocelot: https://lunit-io.github.io/research/publications/ocelot/, https://zenodo.org/records/7844149

