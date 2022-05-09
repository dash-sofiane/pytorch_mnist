import torch
from torch import nn
from torch import optim
from cnn import CNN
from get_mnist import loaders
from train import train

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cnn = CNN()
loss_func = nn.CrossEntropyLoss()   
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)   
num_epochs = 1


train(model=cnn, 
    train_data=loaders['train'], 
    validation_data=loaders['test'], 
    num_epochs=num_epochs, 
    loss_function=loss_func,
    optimizer=optimizer
)