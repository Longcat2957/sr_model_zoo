import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_layer(in_channels:int,
               out_channels:int,
               kernel_size:int,
               bias:bool=True):
    
    padding = int((kernel_size-1)/2)
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=padding,
        bias=bias
    )

def pixelshuffle_blocks(in_channels,
                        out_channels,
                        upscale_factor=2,
                        kernel_size=3):
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])

class ESA(nn.Module):
    def __init__(self,
                 esa_channels:int,
                 n_feats:int,
                 conv:nn.Module):
        super().__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)         # Pointwise conv
        
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x:torch.Tensor):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3  = F.interpolate(c3 ,(x.size(2), x.size(3)),
                            mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m
    
class RLFB(nn.Module):
    '''
        Residual Local Feature Block(RLFB)
    
    '''
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 esa_channels:int=16):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
            
        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)
        
        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.esa = ESA(esa_channels, out_channels, nn.Conv2d)
        
        self.act = nn.LeakyReLU(0.05)
        
    def forward(self, x:torch.Tensor):
        out = self.c1_r(x)
        out = self.act(out)
        
        out = self.c2_r(out)
        out = self.act(out)
        
        out = self.c3_r(out)
        out = self.act(out)
        
        out = out + x
        out = self.esa(self.c5(out))
        
        return out
        


class RLFN(nn.Module):
    def __init__(self,
                 in_channels:int=3,
                 out_channels:int=3,
                 feature_channels:int=52,
                 upscale_ratio:int=4):
        super().__init__()        
        self.conv_1 = conv_layer(in_channels,
                                 feature_channels,
                                 kernel_size=3)
        self.block_1 = RLFB(feature_channels)
        self.block_2 = RLFB(feature_channels)
        self.block_3 = RLFB(feature_channels)
        self.block_4 = RLFB(feature_channels)
        self.block_5 = RLFB(feature_channels)
        self.block_6 = RLFB(feature_channels)
        
        self.conv_2 = conv_layer(feature_channels,
                                 feature_channels,
                                 kernel_size=3)
        self.upsampler = pixelshuffle_blocks(
            feature_channels,
            out_channels,
            upscale_factor=upscale_ratio
        )
        
    def forward(self, x:torch.Tensor):
        out_feature = self.conv_1(x)
        
        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        out_b5 = self.block_5(out_b4)
        out_b6 = self.block_6(out_b5)
        
        out_los_res = self.conv_2(out_b6) + out_feature
        output = self.upsampler(out_los_res)
        return output