"""
ELEC 475: Lab 3 - Correction Model (Starter Template)
A small CNN that predicts the offset from a patch centre to the true nose.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrectionNet(nn.Module):
    """
    CorrectionNet for (3, 64, 64) ImageNet-normalized patches.

    Spatial dimension arithmetic (each block halves the size with 2x2 max-pooling):
        Input        : (B, 3, 64, 64)
        Block 1      : (B, 32, 32, 32)
        Block 2      : (B, 64, 16, 16)
        Block 3      : (B, 128, 8, 8)
        Block 4      : (B, 128, 4, 4)
        Flatten      : 128 * 4 * 4 = 2,048 features
        FC1          : 2,048 -> 256
        FC2          : 256 -> 2, the normalized offset (du, dv) / HALF_PATCH
    """
    def __init__(self):
        super(CorrectionNet, self).__init__()

        # ======================================================================
        # [POST-LAB STEP 2]: Implement the CorrectionNet Layers
        #
        # Requirements:
        #   1. Four conv blocks, with output channels 32, 64, 128 and 128.
        #      Each block: nn.Conv2d(kernel_size=3, padding=1) -> nn.BatchNorm2d
        #      -> ReLU -> nn.MaxPool2d(kernel_size=2, stride=2)
        #   2. fc1: in_features = 128 * 4 * 4 (2,048), out_features = 256
        #   3. fc2: in_features = 256, out_features = 2
        # ======================================================================
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        # One shared pool (no weights): halves H and W in every block, 64 -> 32 -> 16 -> 8 -> 4
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ======================================================================
        # [POST-LAB STEP 2]: Implement the Forward Pass
        #
        # 1. Apply the four conv blocks
        # 2. Flatten using torch.flatten(x, start_dim=1)
        # 3. Apply fc1 -> ReLU, then fc2 (no activation)
        # 4. Return the (B, 2) normalized offsets
        # ======================================================================
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # (B,  32, 32, 32)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # (B,  64, 16, 16)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # (B, 128,  8,  8)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))   # (B, 128,  4,  4)
        x = torch.flatten(x, start_dim=1)                # (B, 2048)
        x = F.relu(self.fc1(x))                          # (B, 256)
        return self.fc2(x)                               # (B, 2): offset, no activation


class CorrectionNetModified(nn.Module):
    """
    Your modified CorrectionNet (Post-Lab Step 5). Same input (B, 3, 64, 64) and
    output (B, 2) normalized offsets as CorrectionNet.
    """
    def __init__(self):
        super(CorrectionNetModified, self).__init__()
        # ======================================================================
        # [POST-LAB STEP 5]: Implement Your Modified Correction Model
        # ======================================================================
        # Change: VGG-style blocks. Each block has TWO 3x3 conv -> BN -> ReLU layers
        # before the 2x2 max pool, instead of one. Channels, pooling, flatten size
        # and the FC head are identical to CorrectionNet, so the only difference is
        # depth (8 conv layers instead of 4).
        #
        # Why: two stacked 3x3 convs see a 5x5 region with an extra ReLU in between
        # (VGG lecture). The receptive field of each final 4x4 cell grows from
        # 46 to 76 input pixels, i.e. more than the whole 64 x 64 patch, so every
        # cell can relate the nose to surrounding context (eyes, muzzle edge).
        self.block1 = self._block(3, 32)      # (32, 32, 32)
        self.block2 = self._block(32, 64)     # (64, 16, 16)
        self.block3 = self._block(64, 128)    # (128, 8, 8)
        self.block4 = self._block(128, 128)   # (128, 4, 4)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 2)

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        """[conv3x3 -> BN -> ReLU] x 2, then 2x2 max pool (halves H and W)."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ======================================================================
        # [POST-LAB STEP 5]: Implement the Forward Pass
        # ======================================================================
        x = self.block1(x)                    # (B,  32, 32, 32)
        x = self.block2(x)                    # (B,  64, 16, 16)
        x = self.block3(x)                    # (B, 128,  8,  8)
        x = self.block4(x)                    # (B, 128,  4,  4)
        x = torch.flatten(x, start_dim=1)     # (B, 2048)
        x = F.relu(self.fc1(x))               # (B, 256)
        return self.fc2(x)                    # (B, 2)


if __name__ == "__main__":
    from torchinfo import summary
    for model in [CorrectionNet(), CorrectionNetModified()]:
        print(f"Testing {type(model).__name__} Architecture:")
        try:
            summary(model, input_size=(1, 3, 64, 64))
        except Exception as e:
            print(f"Model not fully implemented yet: {e}")