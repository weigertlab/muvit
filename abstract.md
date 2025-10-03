# MuViT: Multi-Resolution Vision Transformer for Large-Scale Microscopy

MuViT is a multi-resolution Vision Transformer architecture designed for processing large-scale microscopy images. The architecture simultaneously processes multiple resolution levels of the same spatial region, enabling efficient capture of both fine-grained details and broader contextual information.

## Technical Features

**Multi-Resolution Architecture**: MuViT operates on multiple resolution levels simultaneously through a unified encoder-decoder framework. Each resolution level is processed through independent projection layers with optional learnable level embeddings, then jointly modeled through shared transformer layers that exploit cross-scale relationships.

**Spatial Dimension Support**: The architecture supports 2D, 3D, and 4D (spatiotemporal) data through dimension-specific encoder/decoder implementations (MuViTEncoder2d/3d/4d, MuViTDecoder2d/3d/4d), maintaining consistent API across dimensionalities.

**Geometric Relationship Modeling**: Spatial relationships across resolution levels are explicitly encoded through bounding box coordinates that define the spatial extent of each crop. These coordinates are transformed into rotary positional embeddings (RoPE) that provide each patch with awareness of its absolute spatial location and relative position to other patches across scales.

**Masked Autoencoding**: MuViT implements masked autoencoder (MAE) pretraining with sophisticated masking strategies including Dirichlet-weighted sampling across resolution levels. The decoder supports multiple modes: single decoder with shared reconstruction, multi-decoder with cross-attention across levels, and isolated per-level reconstruction.

**Flexible Attention Mechanisms**: The transformer layers support multiple attention patterns including full attention, causal masking for hierarchical processing, same-level attention, and random attention patterns, enabling different inductive biases.

**Input Space Flexibility**: Supports both spatial domain and DCT (Discrete Cosine Transform) domain processing with 2D and 3D DCT implementations, enabling frequency-domain learning.

**Applications**: While designed for self-supervised pretraining via MAE, the encoder can extract spatially-structured features for downstream tasks including semantic segmentation, object detection, and multi-modal learning in microscopy domains.

**Training Infrastructure**: Built on PyTorch Lightning with distributed training support, flexible loss functions (MSE, normalized MSE), gradient clipping, and integration with WandB/TensorBoard logging.

The architecture draws inspiration from multi-scale CNNs (HookNet), hierarchical ViTs (Swin), and multi-task masked autoencoders (MultiMAE), while introducing explicit geometric relationship modeling through coordinate-based positional embeddings across resolution pyramids.
