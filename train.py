import torch
from test import test
from torch.autograd import Variable


def train(
    model, 
    train_data, 
    validation_data,
    num_epochs,
    loss_function,
    optimizer,
    ):
    
    model.train()
        
    # Train the model
    total_step = len(train_data)
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_data):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = model(b_x)[0]
            loss = loss_function(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward() 
            # apply gradients             
            optimizer.step()

            
            if (i+1) % 100 == 0:
                # calculate accuracy on validation data
                val_loss, val_accuracy = test(model, validation_data, loss_function)

                #print the results
                print(f'epoch [{epoch + 1}/{num_epochs}], step [{i + 1}/{total_step}], loss: {loss.item():.4f}, val_loss: {val_loss:.4f}, val_acc {val_accuracy:.4f}')

