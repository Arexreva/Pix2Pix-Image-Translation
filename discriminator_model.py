import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                bias=False,
                padding=1,
                padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels=3,
        features=[64, 128, 256, 512]
    ):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2)
        )

        layers = []
        current_channels = features[0]

        for i, feature in enumerate(features[1:]):
            stride = 1 if i == len(features[1:]) - 1 else 2

            layers.append(
                CNNBlock(
                    current_channels,
                    feature,
                    stride=stride
                )
            )

            current_channels = feature

        layers.append(
            nn.Conv2d(
                current_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect"
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        # Concatenate input and target along channels
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)


def test():
    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 3, 256, 256)

    model = Discriminator()
    preds = model(x, y)

    print("Discriminator output shape:", preds.shape)


if __name__ == "__main__":
    test()
