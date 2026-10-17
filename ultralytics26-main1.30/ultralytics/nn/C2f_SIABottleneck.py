import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv  

class SIA(nn.Module):
    """
    Shallow-deep Information Aggregation (SIA) Module.
    Fuses 3x3 and 5x5 convolutions to expand the receptive field and utilizes 
    Dropout and MLP layers to enhance feature fusion.
    """
    def __init__(self, c1: int, c2: int, dropout_prob: float = 0.1):
        super().__init__()
        # Parallel convolutions for multi-scale feature extraction
        self.conv3x3 = Conv(c1, c1, 3, 1)
        self.conv5x5 = Conv(c1, c1, 5, 1)
        
        # 1x1 Convolution to adjust channels after concatenation
        self.conv1x1_adjust = Conv(c1 * 2, c1, 1, 1)
        
        # Dropout layer for regularization[cite: 1]
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # MLP layer for feature transformation[cite: 1]
        # Implemented using 1x1 convolutions as standard in CNN channel-wise MLPs
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.Sigmoid()  # Used to scale the features between 0 and 1 before multiplication
        )
        
        # Final 1x1 convolution and Batch Normalization (BN) layer[cite: 1]
        self.conv1x1_final = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch 1 & 2: Apply separate 3x3 and 5x5 convolutional layers[cite: 1]
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        
        # Equation 4: Concatenate, adjust channels via 1x1 Conv, and sum with original feature X[cite: 1]
        x_cat = torch.cat([x3, x5], dim=1)
        y1 = self.conv1x1_adjust(x_cat) + x
        
        # Equation 5: Dropout -> MLP -> Element-wise multiplication with Y1[cite: 1]
        mlp_weights = self.mlp(self.dropout(y1))
        y2 = mlp_weights * y1
        
        # Equation 6: Final 1x1 Conv -> BN[cite: 1]
        out = self.bn(self.conv1x1_final(y2))
        return out


class SIA_Bottleneck(nn.Module):
    """
    SIA Bottleneck based on Figure 4.
    Replaces the conventional bottleneck convolutions with an SIA holistic operator.
    """
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        
        # Primary feature extraction branch starting with a CBS (Conv-BN-SiLU)[cite: 1]
        self.cv1 = Conv(c1, c_, 1, 1)
        
        # Integrated SIA module substituting standard convolutions[cite: 1]
        self.sia = SIA(c_, c2)
        
        # True/False shortcut residual connection[cite: 1]
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SIA bottleneck with optional shortcut connection."""
        # If shortcut is True, add original input to the output of the SIA branch[cite: 1]
        return x + self.sia(self.cv1(x)) if self.add else self.sia(self.cv1(x))


class C2f_SIA_Bottleneck(nn.Module):
    """
    Faster Implementation of CSP Bottleneck with 2 convolutions, incorporating the SIA_Bottleneck.
    """
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions and n SIA Bottlenecks.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        
        # Initial 1x1 Conv layer mapping input to 2 * hidden_channels[cite: 2]
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        
        # Final 1x1 Conv layer to map concatenated features to output channels[cite: 2]
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        # Using nn.ModuleList to stack 'n' numbers of SIA_Bottlenecks[cite: 2]
        self.m = nn.ModuleList(
            SIA_Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f_SIA_Bottleneck layer[cite: 2]."""
        # Split the tensor into two branches[cite: 2]
        y = list(self.cv1(x).chunk(2, 1))
        
        # Pass the second chunk sequentially through all stacked SIA Bottlenecks[cite: 2]
        y.extend(m(y[-1]) for m in self.m)
        
        # Concatenate all generated feature maps and process through final convolution[cite: 2]
        return self.cv2(torch.cat(y, 1))