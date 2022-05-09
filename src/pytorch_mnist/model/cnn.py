import torch

import torch.nn as nn


class CNN(nn.Module):
    """Convolutional neural network class module. This inherits from the
    basic module neural network class."""

    def __init__(self) -> None:
        """
        Initializes the internal layers of the CNN.
            Conv1 - Input convolutional layer. Output Pooled.
            Conv2 - Hidden convolutional layer. Output Pooled.
            out - Output linear layer. Takes in a flatten Tensor from Conv2.
                  Output of which is the probability of each class."""

        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

        # TODO
        # These dimensions are hard coded and hard to explore. There should be
        # a configuration variable - and the method should calculate the
        # dimensions of this layer automatically from the configuration.

    def forward(self, x: torch.Tensor):
        """Batch passing function. Pushes a tensor of dimensions
        (batch_size, 1, H, W) through the neural network.

        Args:
            x (torch.Tensor): Takes in a bi-dimensional function

        Returns:
            torch.Tensor: Output of the simple three layer convolutional neural
            network.
        """

        x = self.conv1(x)  # input layer
        x = self.conv2(x)  # hidden conv layer.

        x = x.view(
            x.size(0), -1
        )  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)

        return self.out(x)
