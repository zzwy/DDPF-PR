#-*- coding:utf-8 -*-
import math
import time
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from thop import profile
import torch.nn.functional as F
import numbers
from einops import rearrange
from thop import profile
from thop import clever_format
from torch.nn import GroupNorm
from torch.nn.utils import weight_norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=64,prompt_len=5,prompt_size = 96,lin_dim = 64):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        

    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)
        return prompt


class SpatialProcess(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1,
                                      padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + \
               attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn
    
class fre(nn.Module):
    def __init__(self, in_dim=64):
        super().__init__()
        self.main = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1)
        self.mag = nn.Conv2d(in_dim, in_dim, kernel_size=1, padding=0)
        self.pha = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=1, padding=0),
        )

    def forward(self, x):
        _, _, H, W = x.shape
        fre = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(fre)
        pha = torch.angle(fre)
        mag_out = self.mag(mag)
        mag_res = mag_out - mag
        pooling = torch.nn.functional.adaptive_avg_pool2d(mag_res, (1, 1))
        pooling = torch.nn.functional.softmax(pooling, dim=1)
        pha1 = pha * pooling
        pha1 = self.pha(pha1)
        pha_out = pha1 + pha
        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        fre_out = torch.complex(real, imag)
        y = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')
        return self.main(x) + y
    

class HybridFusionBlock(nn.Module):
    def __init__(self, in_channels,num_heads):
        super(HybridFusionBlock, self).__init__()
        # 分别对 X_H 和 X_S 进行不同的处理
        self.ln_xh = LayerNorm(in_channels,LayerNorm_type='0')
        self.conv_1x1_xh = nn.Conv2d(in_channels, in_channels*2, kernel_size=1)
        self.dwconv_xh = nn.Conv2d(in_channels*2, in_channels*2, kernel_size=3, padding=1, groups=in_channels*2)
        
        self.ln_xs = LayerNorm(in_channels,LayerNorm_type='0')
        self.conv_1x1_xs = nn.Conv2d(in_channels, in_channels*3, kernel_size=1)
        self.dwconv_xs = nn.Conv2d(in_channels*3, in_channels*3, kernel_size=3, padding=1, groups=in_channels*3)
        
        self.conv_k=nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        self.conv_v=nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        
        ##---------- Attention -----------------------
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
   
    def forward(self, prompt, feats):
        b,c,h,w=prompt.shape
        # X_H 分支
        prompt=self.ln_xh(prompt)
        prompt_kv = self.dwconv_xh(self.conv_1x1_xh(prompt))
        prompt_k,prompt_v=prompt_kv.chunk(2,1)
        
        # X_S 分支
        shortcut=feats
        feats = self.ln_xs(feats)
        feats_qkv = self.dwconv_xs(self.conv_1x1_xs(feats))
        feats_q,feats_k,feats_v=feats_qkv.chunk(3,1)

        k=self.conv_k(torch.concat([prompt_k,feats_k],1))
        v=self.conv_v(torch.concat([prompt_v,feats_v],1))

        feats_q = rearrange(feats_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        feats_q = torch.nn.functional.normalize(feats_q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (feats_q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out=self.project_out(out)

        return out+shortcut

class FeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        # 跨域特征交互模块
        self.cross_interaction = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.Sigmoid()  # 生成0-1的注意力权重
        )
        
        # 自适应特征加权
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//4, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, freq_feat, spatial_feat):
        # 特征拼接与交互 [B, C, H, W]
        concat_feat = torch.cat([freq_feat, spatial_feat], dim=1)  # [B, 2C, H, W]
        
        # 生成跨域注意力掩码
        cross_mask = self.cross_interaction(concat_feat)  # [B, C, H, W]
        
        # 特征增强
        enhanced_freq = freq_feat * cross_mask
        enhanced_spatial = spatial_feat * (1 - cross_mask)
        
        # 通道注意力加权
        channel_weight = self.channel_att(enhanced_freq + enhanced_spatial)
        fused_feat = (enhanced_freq + enhanced_spatial) * channel_weight
        
        return fused_feat
    
class ContextAwareTransformer(nn.Module):
    
    def __init__(self, dim,  num_heads, img_size,
                 mlp_ratio=4.,):
        super().__init__()
        self.dim = dim

        self.prompt=PromptGenBlock()

        self.fre_feature=fre()
        self.spa_feature=SpatialProcess(dim=dim)
        self.fusion=FeatureFusion(in_channels=dim)
        self.fine=HybridFusionBlock(dim,num_heads)


    def forward(self, x):
        shortcut=x

        fre=self.fre_feature(x)
        spa=self.spa_feature(x)
        
        fusion_feat=self.fusion(fre,spa)
        prompt=self.prompt(fusion_feat)
        out=self.fine(prompt,fusion_feat)

        return out+shortcut

class BasicLayer(nn.Module):

    def __init__(self, dim, depth, num_heads,img_size,
                 mlp_ratio=4.,
                 ):

        super().__init__()
        self.dim = dim
        
        self.depth = depth

        self.blocks = nn.ModuleList([
            ContextAwareTransformer(dim=dim, 
                                 num_heads=num_heads, img_size=img_size,
                                 mlp_ratio=mlp_ratio,
                                 )
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x) # B L C
        return x


class ContextAwareTransformerBlock(nn.Module):

    def __init__(self, dim,  depth, num_heads,img_size,
                 mlp_ratio=4., ):
        super().__init__()

        self.dim = dim
        
        self.residual_group = BasicLayer(dim=dim,                                         
                                         depth=depth,
                                         num_heads=num_heads,img_size=img_size,
                                         mlp_ratio=mlp_ratio,
                                         )

        
        self.dilated_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=2, bias=True, dilation=2)
        

    def forward(self, x):
        res = self.residual_group(x) # B L C
        res = self.dilated_conv(res)+x
        return res

class SpatialAttentionModule(nn.Module):
    def __init__(self, dim):
        super(SpatialAttentionModule, self).__init__()
        self.att1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        f_cat = torch.cat((x1, x2), 1)
        att_map = torch.sigmoid(self.att2(self.relu(self.att1(f_cat))))
        return att_map

class PFShdr(nn.Module):

    def __init__(self, img_size=128,in_chans=6,
                 embed_dim=64, depths=[3, 3,3], num_heads=[8, 8,8],
                 mlp_ratio=4.,
                 drop_path_rate=0.1,):
        super(PFShdr, self).__init__()
        num_in_ch = in_chans
        num_out_ch = 3
        ################################### 1. Feature Extraction Network ###################################
        # coarse feature
        self.conv_f1 = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.conv_f2 = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.conv_f3 = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        
        self.att_module_l = SpatialAttentionModule(embed_dim)
        self.att_module_h = SpatialAttentionModule(embed_dim)
        self.conv_first = nn.Conv2d(embed_dim * 3, embed_dim, 3, 1, 1)
        ################################### 2. HDR Reconstruction Network ###################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ContextAwareTransformerBlock(dim=embed_dim,                        
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],img_size=img_size,
                         mlp_ratio=self.mlp_ratio,
                         )
            self.layers.append(layer)

    
        self.conv_after_body = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)


    def forward_features(self, x):
        for layer in self.layers:
            x = layer(x)        
        return x

    def forward(self, x1, x2, x3):
        # feature extraction network
        # coarse feature
        f1 = self.conv_f1(x1)
        f2 = self.conv_f2(x2)
        f3 = self.conv_f3(x3)

        # spatial feature attention 
        f1_att_m = self.att_module_l(f1, f2)
        f1_att = f1 * f1_att_m
        f3_att_m = self.att_module_h(f3, f2)
        f3_att = f3 * f3_att_m

        x = self.conv_first(torch.cat((f1_att, f2, f3_att), dim=1))
        # CTBs for HDR reconstruction
        x = self.conv_after_body(self.forward_features(x) + x)
        x = torch.sigmoid(x)
        return x
