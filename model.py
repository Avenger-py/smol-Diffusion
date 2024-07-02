import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_num_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings

    
class SelfAttention(nn.Module):
    def __init__(self, n_embd, attn_chans):
        super().__init__()
        """ 
        attn_chans is same as head_size
        Low number of heads or high attn_chans = faster forward pass
        """ 
        self.heads = n_embd // attn_chans
        self.attn_chans = attn_chans 
        self.scale = (attn_chans) ** 0.5
        self.norm = nn.BatchNorm2d(n_embd)
        self.qkv = nn.Linear(n_embd, n_embd * 3, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
    
    def forward(self, x):
        n,c,h,w = x.shape
        x = self.norm(x).view(n,c,-1).transpose(1,2) # (n, s=H*W, c)
        x = self.qkv(x)
        x = x.view(n, self.heads, h*w, self.attn_chans*3) # (n, h, s, d*3)
        q, k, v = x.chunk(3, dim=-1) # (n, h, s, d) x 3
        attn_scores = (q @ k.transpose(-2,-1))/self.scale
        x = attn_scores.softmax(dim=-1) @ v
        x = x.view(n, h*w, self.heads*self.attn_chans) # (n, s, h, d)
        x = self.proj(x).transpose(1,2).view(n, c, h, w)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
    
        self.residual = residual
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), 
                                         nn.BatchNorm2d(mid_channels), 
                                         nn.ReLU(), 
                                         nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1), 
                                         nn.BatchNorm2d(out_channels)
                                        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        if self.residual:
            return self.relu(x + self.double_conv(x))
        else:
            return self.double_conv(x) # add relu?

    
    
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, time_embd_dim):
        super().__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, in_channels, residual=True),
            ConvBlock(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(nn.Linear(time_embd_dim, out_channels), nn.ReLU())
#             nn.SiLU(),

    
    def forward(self, x, t):
        x = self.maxpool_conv(x)
        t = self.emb_layer(t)[:, :, None, None]
#         print(f"t shape:{t.shape}, x shape: {x.shape}")
        t = t.repeat(1, 1, x.shape[-2], x.shape[-1])
#         print(f"t shape again: {t.shape}")
        return x + t
        
    
    
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, time_embd_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, residual=True),
            ConvBlock(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(nn.Linear(time_embd_dim, out_channels), nn.ReLU())

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Unet(nn.Module):
    def __init__(self, time_embd_dim, in_channels=3, out_channels=3):
        super().__init__()
        self.time_embd_dim = time_embd_dim
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(self.time_embd_dim), nn.Linear(self.time_embd_dim, self.time_embd_dim), nn.ReLU())
        
        self.in_conv = ConvBlock(in_channels, 64) #nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.down1 = Downsample(64, 128, time_embd_dim=self.time_embd_dim)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Downsample(128, 256, time_embd_dim=self.time_embd_dim)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Downsample(256, 256, time_embd_dim=self.time_embd_dim)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = ConvBlock(256, 512)
        self.bot2 = ConvBlock(512, 512)
        self.bot3 = ConvBlock(512, 256)

        self.up1 = Upsample(512, 128, time_embd_dim=self.time_embd_dim)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Upsample(256, 64, time_embd_dim=self.time_embd_dim)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Upsample(128, 64, time_embd_dim=self.time_embd_dim)
        self.sa6 = SelfAttention(64, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x, t):
        t = self.time_mlp(t)
        x1 = self.in_conv(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.out_conv(x)
        return output