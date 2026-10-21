"""
ELEC 475: Lab 3 - SnoutNet Nose Localization Models (Starter Template)
Three regression models that predict a pet's nose position (u, v):
    SnoutNet        : Custom 3-block CNN trained from scratch
    SnoutNetAlexNet : SnoutNet-A, ImageNet-pretrained torchvision AlexNet backbone
    SnoutNetVGG16   : SnoutNet-V, ImageNet-pretrained torchvision VGG16 backbone

All models take (B, 3, 227, 227) input and return (B, 2) NORMALIZED coordinates
(u/227, v/227). Layer names must match exactly for the provided weights to load.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet, vgg16, AlexNet_Weights, VGG16_Weights


class SnoutNet(nn.Module):
    """
    SnoutNet Architecture for (3, 227, 227) RGB images.

    Spatial dimension arithmetic (pooling with ceil_mode=True rounds UP):
        Input        : (B, 3, 227, 227)
        Conv1 (p=1)  : (B, 64, 227, 227)
        MaxPool1     : ceil((227 - 4)/4) + 1 = 57 --> (B, 64, 57, 57)
        Conv2 (p=1)  : (B, 128, 57, 57)
        MaxPool2     : ceil((57 - 4)/4) + 1 = 15  --> (B, 128, 15, 15)
        Conv3 (p=1)  : (B, 256, 15, 15)
        MaxPool3     : ceil((15 - 4)/4) + 1 = 4   --> (B, 256, 4, 4)
        Flatten      : 256 * 4 * 4 = 4,096 features
        FC1          : 4,096 -> 1,024
        FC2          : 1,024 -> 1,024
        FC3          : 1,024 -> 2
    """
    def __init__(self):
        super(SnoutNet, self).__init__()

        # ======================================================================
        # [PRE-LAB TASK 4]: Implement the SnoutNet Layers
        #
        # Requirements (use these attribute names EXACTLY):
        #   1. self.conv1: in_channels=3,   out_channels=64,  kernel_size=3, padding=1
        #   2. self.conv2: in_channels=64,  out_channels=128, kernel_size=3, padding=1
        #   3. self.conv3: in_channels=128, out_channels=256, kernel_size=3, padding=1
        #   4. self.pool : nn.MaxPool2d(kernel_size=4, stride=4, ceil_mode=True)
        #   5. self.fc1  : in_features=256 * 4 * 4 (4,096), out_features=1024
        #   6. self.fc2  : in_features=1024, out_features=1024
        #   7. self.fc3  : in_features=1024, out_features=2
        # ======================================================================
        # 3x3 conv with padding=1 keeps H and W unchanged; only the pool shrinks them.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # One pooling layer shared by all three blocks. It has no weights, so it
        # adds nothing to the state_dict. ceil_mode=True gives 57/15/4, not 56/14/3.
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4, ceil_mode=True)

        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ======================================================================
        # [PRE-LAB TASK 4]: Implement the Forward Pass
        #
        # 1. Apply conv1 -> ReLU -> pool, conv2 -> ReLU -> pool, conv3 -> ReLU -> pool
        # 2. Flatten spatial features using torch.flatten(x, start_dim=1)
        # 3. Apply fc1 -> ReLU, fc2 -> ReLU, then fc3 (no activation)
        # 4. Return the (B, 2) normalized (u, v) prediction
        # ======================================================================
        x = self.pool(F.relu(self.conv1(x)))   # (B,  64, 57, 57)
        x = self.pool(F.relu(self.conv2(x)))   # (B, 128, 15, 15)
        x = self.pool(F.relu(self.conv3(x)))   # (B, 256,  4,  4)
        x = torch.flatten(x, start_dim=1)      # (B, 4096), keeps the batch dim
        x = F.relu(self.fc1(x))                # (B, 1024)
        x = F.relu(self.fc2(x))                # (B, 1024)
        return self.fc3(x)                     # (B, 2): no activation, it's regression


class SnoutNetAlexNet(nn.Module):
    """
    SnoutNet-A: torchvision AlexNet with its final 1000-class layer replaced
    by a 2-output regression layer.
    """
    def __init__(self, pretrained: bool = True):
        super(SnoutNetAlexNet, self).__init__()

        # ======================================================================
        # [PRE-LAB TASK 4]: Build the AlexNet Backbone
        #
        # Requirements:
        #   1. self.backbone = alexnet(weights=AlexNet_Weights.DEFAULT) if pretrained,
        #      otherwise alexnet(weights=None)
        #   2. Replace self.backbone.classifier[6] with nn.Linear(4096, 2)
        # ======================================================================
        weights = AlexNet_Weights.DEFAULT if pretrained else None
        self.backbone = alexnet(weights=weights)
        # classifier[6] is the final Linear(4096, 1000) ImageNet class layer.
        # Swap it for a 2-output regression head that predicts (u/227, v/227).
        self.backbone.classifier[6] = nn.Linear(4096, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [PRE-LAB TASK 4]: Return self.backbone(x)
        return self.backbone(x)


class SnoutNetVGG16(nn.Module):
    """
    SnoutNet-V: torchvision VGG16 with its final 1000-class layer replaced
    by a 2-output regression layer.
    """
    def __init__(self, pretrained: bool = True):
        super(SnoutNetVGG16, self).__init__()

        # ======================================================================
        # [PRE-LAB TASK 4]: Build the VGG16 Backbone
        #
        # Requirements:
        #   1. self.backbone = vgg16(weights=VGG16_Weights.DEFAULT) if pretrained,
        #      otherwise vgg16(weights=None)
        #   2. Replace self.backbone.classifier[6] with nn.Linear(4096, 2)
        # ======================================================================
        weights = VGG16_Weights.DEFAULT if pretrained else None
        self.backbone = vgg16(weights=weights)
        self.backbone.classifier[6] = nn.Linear(4096, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [PRE-LAB TASK 4]: Return self.backbone(x)
        return self.backbone(x)


if __name__ == "__main__":
    # ==========================================================================
    # [PRE-LAB TASK 4, STEP 3]: Test Your Models
    #
    # 1. Print a torchinfo summary of SnoutNet for input_size=(1, 3, 227, 227)
    # 2. Pass torch.randn(1, 3, 227, 227) through each model (use pretrained=False)
    #    and print its output shape and parameter count
    #
    # [PRE-LAB TASK 5]: Load the provided weights and reproduce a known
    # prediction (see the Pre-Lab document for the code and expected output)
    # ==========================================================================
    from pathlib import Path
    from torchinfo import summary

    # --- Task 4, Step 3: shapes and parameter counts ---------------------------
    # Compare the Output Shape column with the table in the Pre-Lab:
    # (64, 57, 57) -> (128, 15, 15) -> (256, 4, 4) -> 4096 -> 1024 -> 1024 -> 2
    summary(SnoutNet(), input_size=(1, 3, 227, 227),
            col_names=("output_size", "num_params"))

    x = torch.randn(1, 3, 227, 227)
    for model in [SnoutNet(), SnoutNetAlexNet(pretrained=False), SnoutNetVGG16(pretrained=False)]:
        model.eval()
        with torch.no_grad():
            out = model(x)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{type(model).__name__:<20}output {tuple(out.shape)}     params {n_params:,}")

    # --- Task 5, Step 1: load the provided weights (code from the Pre-Lab) -----
    names = ["snoutnet.pth", "alexnet.pth", "vgg16.pth"]
    missing = [n for n in names if not (Path("checkpoints") / n).exists()]
    if missing:
        print(f"\nSkipping Task 5: not in checkpoints/ yet: {', '.join(missing)}")
        raise SystemExit

    checks = [(SnoutNet(), "snoutnet.pth"),
              (SnoutNetAlexNet(pretrained=False), "alexnet.pth"),
              (SnoutNetVGG16(pretrained=False), "vgg16.pth")]
    for model, name in checks:
        state = torch.load(f"checkpoints/{name}", map_location="cpu",
                           weights_only=True)
        model.load_state_dict(state, strict=True)
        print(f"Loaded {name}")

    # --- Task 5, Step 2: reproduce a known prediction (code from the Pre-Lab) --
    # Expected: Ground truth: tensor([311., 152.])
    #           Prediction:   tensor([350.3443, 181.7347])  (within 1 pixel)
    from utils.dataset import PetNoseDataset
    model = checks[0][0]  # SnoutNet (the loop leaves VGG16 in model)
    dataset = PetNoseDataset(split="test")
    model.eval()
    image, label, orig_size = dataset[0]
    with torch.no_grad():
        # Model output is normalized: (u/227, v/227)
        pred = model(image.unsqueeze(0))[0]
    print("Ground truth:", label / 227 * orig_size)
    print("Prediction: ", pred * orig_size)
