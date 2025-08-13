from abc import ABC, abstractmethod
from typing import Generic, Literal, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from .bblocks import SaveableModel
from .decoders import MuViTDecoder, MuViTDecoder2d, MuViTDecoder3d
from .encoders import MuViTEncoder, MuViTEncoder2d, MuViTEncoder3d

T = TypeVar("T", bound=Tuple[int, ...])


class MuViTMAE(SaveableModel, ABC, Generic[T]):
    def __init__(self, in_channels:int=1, 
                 levels:tuple[float]=(1,4), 
                 patch_size:Union[int,tuple[int,...]]=16, 
                 num_layers:int=12, 
                 dim:int=512, 
                 num_layers_decoder:int=2,
                 dim_decoder:Optional[int]=256,
                 heads:int=8, 
                 decoder_mode:Literal['single', 'multi', 'multi_iso']='multi',
                 loss:Literal['mse', 'norm_mse']='mse',
                 masking_ratio:float=0.75,
                 use_level_embed:bool=True,
                 use_rotary_embed:bool=True,
                 attention_mode:Literal['all', 'causal', 'same', 'random']='all',
                 masking_mode:Literal['dirichlet', 'random']|tuple[float]='dirichlet',
                 input_space:Literal['real', 'dct']='real',
                 dropout:float=0.0,
                 ):
        """Initialize a Masked Autoencoder with multi-level Vision Transformer.
        
        Args:
            in_channels: Number of input channels
            levels: Tuple of scale factors for each level
            patch_size: Size of image patches
            num_layers: Number of encoder layers
            dim: Hidden dimension
            num_layers_decoder: Number of decoder layers
            dim_decoder: Hidden dimension for decoder
            heads: Number of attention heads
            decoder_mode: Type of decoder ('single', 'multi', 'multi_iso')
            loss: Type of loss function ('mse', 'norm_mse')
            masking_ratio: Ratio of patches to mask
            use_level_embed: Whether to use level embeddings
            use_rotary_embed: Whether to use rotary embeddings
            attention_mode: Type of attention masking
            masking_mode: Type of masking ('dirichlet', 'random' or tuple of probabilities)
            dropout: Dropout probability for transformer layers
        """
        super().__init__()
        
        if dim_decoder is None:
            dim_decoder = dim

        # Select appropriate encoder/decoder based on spatial dimensionality
        EncoderCls = self.encoder_class
        DecoderCls = self.decoder_class

        self._config = dict(in_channels=in_channels, levels=levels, patch_size=patch_size, 
                    num_layers=num_layers, dim=dim, heads=heads,
                    use_level_embed=use_level_embed, use_rotary_embed=use_rotary_embed, 
                    decoder_mode=decoder_mode, loss=loss, masking_ratio=masking_ratio, 
                    masking_mode=masking_mode, attention_mode=attention_mode,
                    num_layers_decoder=num_layers_decoder, dim_decoder=dim_decoder,
                    dropout=dropout, input_space=input_space, ndim=self.ndim)

        self.encoder = EncoderCls(in_channels=in_channels, levels=levels, 
                                 patch_size=patch_size, num_layers=num_layers, dim=dim, heads=heads,
                                 use_level_embed=use_level_embed,
                                 use_rotary_embed=use_rotary_embed,
                                 attention_mode=attention_mode,
                                 dropout=dropout, input_space=input_space)
        
        # Calculate patch dimension for final layer
        patch_dim = in_channels * np.prod(self.encoder.patch_size)
        self.final = nn.Linear(dim_decoder, patch_dim)
        
        self.patch_size = self.encoder.patch_size
        self.masking_ratio = masking_ratio
        self.masking_mode = masking_mode
        self.decoder_mode = decoder_mode
        
        if decoder_mode == 'single':
            self.mask_token = nn.Parameter(torch.randn(1, 1, dim))
            self.decoder = DecoderCls(dim, dim_decoder, num_layers=num_layers_decoder, 
                                      heads=heads, use_rotary_embed=use_rotary_embed,
                                      dropout=dropout)
        elif decoder_mode in ('multi', 'multi_iso'):
            self.mask_token = nn.Parameter(torch.randn(1, len(self.encoder.levels), dim))
            self.decoder = nn.ModuleList([
                DecoderCls(dim, dim_decoder, num_layers=num_layers_decoder, 
                           heads=heads, use_rotary_embed=use_rotary_embed,
                           dropout=dropout)                                     
                for _ in range(len(self.encoder.levels))])
        else:
            raise ValueError(f"Invalid decoder mode: {decoder_mode}")
        self.loss_fn = loss

    @property
    @abstractmethod
    def encoder_class(self) -> Type[MuViTEncoder]:
        """Get the appropriate encoder class for this dimensionality."""
        pass

    @property
    @abstractmethod
    def decoder_class(self) -> Type[MuViTDecoder]:
        """Get the appropriate decoder class for this dimensionality."""
        pass
    
    @property
    @abstractmethod
    def ndim(self) -> int:
        """Get the number of dimensions for this model."""
        pass

    @abstractmethod
    def patch_token_to_image(self, x:Tensor, *shape:int, space_transform:bool=True) -> Tensor:
        """Convert patch tokens back to image."""
        pass
    
    def token_to_patch(self, x:Tensor) -> Tensor:
        if self.ndim == 2:
            return rearrange(x, "b n (p1 p2) -> b n p1 p2", 
                             p1=self.patch_size[0], p2=self.patch_size[1])
        elif self.ndim == 3:
            return rearrange(x, "b n (p1 p2 p3) -> b n p1 p2 p3", 
                             p1=self.patch_size[0], p2=self.patch_size[1], p3=self.patch_size[2])
        else:
            raise ValueError(f"Invalid number of dimensions: {self.ndim}")

    def forward(self, x:Tensor, bbox:Optional[tuple[Tensor,Tensor]]=None, return_all:bool=False, eps:float=1e-2):
        """Process input through the MAE model.
        
        Args:
            x: Input tensor
            bbox: Optional bounding box coordinates
            return_all: Whether to return intermediate results
            eps: Small constant for numerical stability
            
        Returns:
            Dictionary containing:
            - input: Original patches
            - output: Reconstructed patches
            - loss: Reconstruction loss
            - encoded: Encoded features (if return_all)
            - reco: Reconstructed image (if return_all)
            - decoded: Decoded features (if return_all)
            - input_masked: Masked input (if return_all)
            - loss_per_level: Loss per level (if return_all)
        """
        y, coords, patches, batch_range, idx_retain, idx_mask = self.encoder.forward_masked(x, bbox, self.masking_ratio, self.masking_mode)

        N = patches.shape[1]
        N_per_level = N//len(self.encoder.levels)
        
        z = torch.zeros(patches.shape[0], N, y.shape[-1], device=patches.device, dtype=patches.dtype)        
        
        if self.decoder_mode == 'single':        
            z[batch_range, idx_mask] = self.mask_token                
            z[batch_range, idx_retain] = y
            # single self-attention for all levels
            z = self.decoder(z, coords)
        elif self.decoder_mode in ('multi', 'multi_iso'):    
            mask_tokens = torch.repeat_interleave(self.mask_token, N_per_level, dim=1).repeat(patches.shape[0], 1, 1)
            z[batch_range, idx_mask] = mask_tokens[batch_range, idx_mask]
            z[batch_range, idx_retain] = y
            zs = torch.split(z, N_per_level, dim=1)
            cs = torch.split(coords, N_per_level, dim=1)
            if self.decoder_mode == 'multi':
                # allow cross attending to all levels
                z = torch.cat([self.decoder[i](_z, _c, context=z, context_coords=coords) for i, (_z, _c) in enumerate(zip(zs, cs))], dim=1)
            elif self.decoder_mode == 'multi_iso':   
                # do it in isolation for each level
                z = torch.cat([self.decoder[i](_z, _c) for i, (_z, _c) in enumerate(zip(zs, cs))], dim=1)
            else:
                raise ValueError(f"Invalid decoder mode: {self.decoder_mode}")
        else:
            raise ValueError(f"Invalid decoder mode: {self.decoder_mode}")                
            
        z = self.final(z)
        
        # normalize target 
        if self.loss_fn == 'norm_mse':
            p_mean = patches.mean(dim=-1, keepdim=True)
            p_std = patches.std(dim=-1, keepdim=True)
        elif self.loss_fn in ('mse', 'mse_fft'):
            p_mean = torch.tensor(0,device=x.device, dtype=patches.dtype)
            p_std = torch.tensor(1-eps,device=x.device, dtype=patches.dtype)
        else:
            raise ValueError(f"Invalid loss: {self.loss_fn}")
        
        patches_normed = (patches - p_mean)/(p_std+eps)
        
        loss = F.mse_loss(z[batch_range, idx_mask], patches_normed[batch_range, idx_mask])
        if self.loss_fn == 'mse_fft':
            z2 = self.token_to_patch(z[batch_range, idx_mask]).to(torch.float32)
            patches2 = self.token_to_patch(patches_normed[batch_range, idx_mask]).to(torch.float32)
            z2f = torch.fft.rfftn(z2, dim=tuple(-(i+1) for i in range(self.ndim)))
            patches2f = torch.fft.rfftn(patches2, dim=tuple(-(i+1) for i in range(self.ndim)))
            loss = loss + 0.01*F.l1_loss(z2f, patches2f)
        
        out = dict(input=patches, output=z, loss=loss)
        if return_all:
            reco = patches_normed.clone().to(z.dtype)
            reco[batch_range, idx_mask] = z[batch_range, idx_mask]
            reco = reco*(p_std+eps) + p_mean 
            reco = self.encoder.patch_space_transform(reco, inverse=True)            
            reco = self.patch_token_to_image(reco, *x.shape[2:]).to(patches.dtype)
            
            input_masked = patches.clone()    
            input_masked = self.encoder.patch_space_transform(input_masked, inverse=True)            
            input_masked[batch_range, idx_mask] = 0.5
            input_masked = self.patch_token_to_image(input_masked, *x.shape[2:])
            mask = torch.zeros(patches.shape, dtype=bool, device=x.device)
            mask[batch_range, idx_mask] = True
            mask = self.patch_token_to_image(mask, *x.shape[2:])
            # loss per level  
            idx_levels = tuple(torch.arange(N_per_level, device=x.device)+i*N_per_level for i in range(len(self.encoder.levels)))        
            loss_per_level = torch.tensor(tuple(F.mse_loss(z[batch_range, idx_mask[torch.isin(idx_mask, _idx)]], 
                                            patches_normed[batch_range, idx_mask[torch.isin(idx_mask, _idx)]]) for _idx in idx_levels), device=x.device)
            
            out['encoded'] = y
            out['reco'] = reco
            out['decoded'] = z
            out['mask'] = mask
            out['input_masked'] = input_masked
            out['loss_per_level'] = loss_per_level
        return out


class MuViTMAE2d(MuViTMAE[Tuple[int, int]]):
    @property
    def ndim(self) -> int:
        return 2    

    @property
    def encoder_class(self) -> Type[MuViTEncoder]:
        return MuViTEncoder2d
    
    @property
    def decoder_class(self) -> Type[MuViTDecoder]:
        return MuViTDecoder2d

    def patch_token_to_image(self, x:Tensor, C:int, H:int, W:int) -> Tensor:
        """Convert patch tokens back to image.
        
        Input: (B, N, P*P*C) -> Output: (B, L, C, H, W)
        """
        x = torch.split(x, x.shape[1]//len(self.encoder.levels), dim=1)
        x = torch.stack([rearrange(_x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", 
                                 c=C, h=H//self.patch_size[0], w=W//self.patch_size[1], 
                                 p1=self.patch_size[0], p2=self.patch_size[1]) for _x in x], dim=1)
        return x

class MuViTMAE3d(MuViTMAE[Tuple[int, int, int]]):
    def ndim(self) -> int:
        return 3

    @property
    def encoder_class(self) -> Type[MuViTEncoder]:
        return MuViTEncoder3d

    @property
    def decoder_class(self) -> Type[MuViTDecoder]:
        return MuViTDecoder3d
    
    def patch_token_to_image(self, x:Tensor, C:int, D:int, H:int, W:int) -> Tensor:
        """Convert patch tokens back to image.
        
        Input: (B, N, D*H*W*C) -> Output: (B, L, C, D, H, W)
        """
        x = torch.split(x, x.shape[1]//len(self.encoder.levels), dim=1)
        x = torch.stack([rearrange(_x, "b (d h w) (p1 p2 p3 c) -> b c (d p1) (h p2) (w p3)", 
                                 c=C, d=D//self.patch_size[0], h=H//self.patch_size[1], w=W//self.patch_size[2],
                                 p1=self.patch_size[0], p2=self.patch_size[1], p3=self.patch_size[2]) for _x in x], dim=1)
        return x
