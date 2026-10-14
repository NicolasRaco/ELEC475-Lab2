"""
ELEC 475: Lab 2 - Modernized LeNet Architecture (Starter Template)
Ablation Study 1 & 2: Incorporating ReLU and Overlapping Max Pooling.

Key Modifications from Base LeNet-5:
    1. Tanh -> ReLU (Lecture 05 Slide 4-5)
    2. 2x2 Average Pooling -> 3x3 Overlapping Max Pooling with stride=2 (Lecture 05 Slide 7)
"""
import torch
import torch.nn as nn

class LeNetModern(nn.Module):
    """
    Modernized LeNet architecture incorporating ReLU activation and
    overlapping Max Pooling (3x3 with stride 2).

    Spatial dimension arithmetic:
        Input   : (B, 3, 64, 64)
        Conv1   : (64 - 5 + 2*2)/1 + 1 = 64  --> (B, 6, 64, 64)
        ReLU    : (B, 6, 64, 64)
        MaxPool1: floor((64 - 3)/2) + 1 = 31 --> (B, 6, 31, 31)
        Conv2   : (31 - 5 + 2*2)/1 + 1 = 31  --> (B, 16, 31, 31)
        ReLU    : (B, 16, 31, 31)
        MaxPool2: floor((31 - 3)/2) + 1 = 15 --> (B, 16, 15, 15)
        Flatten : 16 * 15 * 15 = 3,600 features
        FC1     : 3,600 -> 120 (with ReLU)
        FC2     : 120 -> 84 (with ReLU)
        FC3     : 84 -> num_classes
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super(LeNetModern, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Ablation 1: Tanh -> ReLU (no gradient saturation).
        # Ablation 2: AvgPool(2,s=2) -> MaxPool(3,s=2), kernel > stride so windows overlap.
        # padding=2 also makes conv resolution-preserving, which is why flatten
        # grows 2,704 -> 3,600 even though channel counts are unchanged.
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2),   # (B, 6, 64, 64)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                           # (B, 6, 31, 31)
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),            # (B, 16, 31, 31)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                           # (B, 16, 15, 15)
        )

        # Same 120 -> 84 -> classes shape as LeNet5, but FC1 widens to 3,600 in.
        self.classifier = nn.Sequential(
            nn.Linear(16 * 15 * 15, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        return logits

if __name__ == "__main__":
    from torchinfo import summary
    model = LeNetModern(num_classes=10, in_channels=3)
    dummy_input = torch.randn(1, 3, 64, 64)
    print("Testing Modernized LeNet Architecture:")
    try:
        summary(model, input_size=(1, 3, 64, 64))
    except Exception as e:
        print(f"Model not fully implemented yet: {e}")
