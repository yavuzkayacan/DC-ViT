import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# AUXILIARY LAYERS
# ---------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    Splits the image into patches and embeds them.
    Corresponds to the "Patching & Patch Embedding" block in the DC-ViT architecture.
    """
    def __init__(self, img_size=(256, 256), patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        self.H, self.W = img_size
        self.patch_size = patch_size
        # Project patches using Conv2d with stride = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        x = self.proj(x)  # Shape: (B, Embed_Dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # Flatten spatial dims: (B, Num_Patches, Embed_Dim)
        return x

class PositionalEncoding(nn.Module):
    """
    Adds learnable positional information to the patch embeddings.
    """
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

    def forward(self, x):
        return x + self.pos_embed

class LeFF(nn.Module):
    """
    Locally-enhanced Feed-Forward Network (LeFF).
    Replaces the standard MLP in Vision Transformers to better capture local context 
    in GPR images using Depthwise Convolutions.
    
    Structure: Linear -> Reshape -> DepthwiseConv -> BatchNorm -> GELU -> Flatten -> Linear
    """
    def __init__(self, embed_dim, expansion_ratio=2, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * expansion_ratio)
        
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, H, W):
        # x shape: (B, N, D)
        B, N, D = x.shape
        x = self.linear1(x)
        
        # Spatial Process: Reshape back to image grid for Convolution
        x_img = x.transpose(1, 2).view(B, -1, H, W) 
        x_img = self.dw_conv(x_img)
        x_img = self.bn(x_img)
        x_img = self.act(x_img)
        
        # Flatten back to sequence
        x = x_img.flatten(2).transpose(1, 2)
        x = self.linear2(x)
        x = self.drop(x)
        return x

class TransformerLayer(nn.Module):
    """
    Single Transformer Encoder Layer using LeFF instead of standard MLP.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.leff = LeFF(embed_dim, expansion_ratio=2, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, H, W):
        # Multi-Head Attention part
        x = x + self.drop1(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        # LeFF part
        x = x + self.drop2(self.leff(self.norm2(x), H, W))
        return x

class TransformerEncoderGroup(nn.Module):
    """
    A sequence of Transformer Layers.
    """
    def __init__(self, embed_dim, num_heads, depth, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, dropout) for _ in range(depth)
        ])
        
    def forward(self, x, H, W):
        for layer in self.layers:
            x = layer(x, H, W)
        return x

# ---------------------------------------------------------
# DENSE TRANSFORMER BLOCK
# ---------------------------------------------------------

class DenseTransformerBlock(nn.Module):
    """
    Implements Dense Connections between Transformer Encoder Groups.
    Features from previous groups are concatenated to minimize information loss.
    """
    def __init__(self, embed_dim, num_heads, H, W, num_groups, group_depth, dropout=0.1):
        super().__init__()
        self.H = H
        self.W = W
        self.num_groups = num_groups
        
        self.blocks = nn.ModuleList()
        self.projections = nn.ModuleList()
        
        # 1st Group (Processes input directly)
        self.blocks.append(TransformerEncoderGroup(embed_dim, num_heads, depth=group_depth, dropout=dropout))
        
        # Subsequent Groups (Concatenation -> Linear Projection -> Transformer Block)
        for i in range(1, num_groups):
            # Input dimension grows as we concatenate previous outputs
            in_features = embed_dim * (i + 1) 
            self.projections.append(nn.Linear(in_features, embed_dim, bias=False))
            self.blocks.append(TransformerEncoderGroup(embed_dim, num_heads, depth=group_depth, dropout=dropout))

    def forward(self, x_input):
        features = [x_input]
        
        # Pass through first block
        x_curr = self.blocks[0](x_input, self.H, self.W)
        features.append(x_curr)
        
        # Pass through remaining blocks with dense connections
        for i in range(1, self.num_groups):
            # Concatenate all previous features (Dense Connection)
            cat_out = torch.cat(features, dim=-1)
            # Reduce dimension back to embed_dim via Linear Projection
            proj_out = self.projections[i-1](cat_out)
            # Pass through Transformer Group
            x_next = self.blocks[i](proj_out, self.H, self.W)
            
            features.append(x_next)
            
        return features[-1]

# ---------------------------------------------------------
# RECONSTRUCTION HEAD
# ---------------------------------------------------------

class ReconstructionHead(nn.Module):
    """
    Reconstructs the image from deep features.
    Uses PixelShuffle (Depth-to-Space) to upsample the feature maps.
    """
    def __init__(self, embed_dim, patch_size, H, W, out_chans=3):
        super().__init__()
        self.H = H
        self.W = W
        
        # Calculate channels needed for PixelShuffle
        output_chans_after_shuffle = embed_dim // patch_size
        conv_out_channels = output_chans_after_shuffle * (patch_size ** 2)
        
        self.conv_expand = nn.Conv2d(embed_dim, conv_out_channels, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(patch_size)
        
        # Final 1x1 Conv to get the desired number of output channels
        self.final_conv = nn.Conv2d(output_chans_after_shuffle, out_chans, kernel_size=1)

    def forward(self, x):
        B, N, D = x.shape
        # Reshape sequence back to feature map: (B, D, H_patch, W_patch)
        x = x.transpose(1, 2).view(B, D, self.H, self.W)
        
        x = self.conv_expand(x)
        x = self.pixel_shuffle(x) # Upsampling
        x = self.final_conv(x)
        x = torch.sigmoid(x) # Normalize output to [0, 1]
        return x

# ---------------------------------------------------------
# MAIN MODEL
# ---------------------------------------------------------

class DCViT(nn.Module):
    """
    DC-ViT: Declutter Vision Transformer for GPR Clutter Removal.
    
    Paper: "A Vision Transformer Based Approach to Clutter Removal in GPR: DC-ViT"
    
    Args:
        img_size (tuple): Input image size (Height, Width).
        patch_size (int): Size of the patches (p).
        in_chans (int): Number of input channels.
        embed_dim (int): Projection dimension (D).
        num_heads (int): Number of attention heads.
        num_groups (int): Number of Dense Transformer Groups.
        group_depth (int): Number of Transformer Layers per group.
        dropout (float): Dropout rate.
    """
    def __init__(self, 
                 img_size=(256, 256), 
                 patch_size=8, 
                 in_chans=3, 
                 embed_dim=512, 
                 num_heads=4, 
                 num_groups=4, 
                 group_depth=3,
                 dropout=0.1):
        super().__init__()
        
        self.H_patch = img_size[0] // patch_size
        self.W_patch = img_size[1] // patch_size
        num_patches = self.H_patch * self.W_patch
        
        # 1. Patching, Embedding & Position Encoding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = PositionalEncoding(num_patches, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 2. Dense Transformer Block
        self.dense_transformer = DenseTransformerBlock(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            H=self.H_patch, 
            W=self.W_patch, 
            num_groups=num_groups, 
            group_depth=group_depth,
            dropout=dropout
        )
        
        # 3. Reconstruction Block
        self.reconstruction = ReconstructionHead(
            embed_dim, 
            patch_size, 
            self.H_patch, 
            self.W_patch,
            out_chans=in_chans
        )

    def forward(self, x):
        # Input shape: (B, C, H, W)
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = self.dropout(x)
        
        # Pass through Dense Transformer Network
        fd = self.dense_transformer(x)
        
        # Reconstruct the clutter-free image
        out = self.reconstruction(fd)
        return out
