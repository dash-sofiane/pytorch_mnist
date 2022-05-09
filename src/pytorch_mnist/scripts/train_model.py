import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from pytorch_mnist.model.cnn import CNN
from pytorch_mnist.utils.get_mnist import train_data, test_data
from pytorch_mnist.process.fit import FitTestClass


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get the loaders
loaders = {
    "train": DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
    "test": DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1),
}

# model settings
cnn = CNN()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.01)
num_epochs = 10
early_stopping_patience = 2
model_save_path = "./last_trained_model.pt"

# send model to gpu
# cnn = cnn.to(device)
# TODO: fix gpu utilization

fitter = FitTestClass(model=cnn, config={"some": "values?"})

fitter.train(
    train_data=loaders["train"],
    validation_data=loaders["test"],
    num_epochs=num_epochs,
    loss_function=loss_func,
    optimizer=optimizer,
    model_save_path=model_save_path,
    early_stopping_patience=early_stopping_patience,
)
