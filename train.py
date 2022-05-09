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
    model_save_path='./last_trained_model.pt',
    early_stopping_patience=None
    ):
    
    model.train()
        
    # Train the model
    total_step = len(train_data)

    # for early stopping
    last_improvement = 0
    prev_loss = float("inf")

    for epoch in range(num_epochs):
        for i, (X, y) in enumerate(train_data):

            output = model(X)
            loss = loss_function(output, y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()
            # apply gradients             
            optimizer.step()

            # print the results per step
            if (i+1) % 100 == 0:
                print(f'    epoch [{epoch + 1}/{num_epochs}], step [{i + 1}/{total_step}], loss: {loss.item():.4f}')

        # calculate loss & accuracy on validation data
        val_loss, val_acc = test(model, validation_data, loss_function)
        print(f'epoch [{epoch + 1}/{num_epochs}], val_loss: {val_loss:.4f}, val_acc {val_acc:.4f}')

        # saving the model if it improved (only monitoring val_loss for now)
        if val_loss < prev_loss:
            print(f"val_loss improved from {prev_loss:.4f} to {val_loss:.4f}.")
            print(f"Saving model in {model_save_path}")
            torch.save(model.state_dict(), model_save_path)

            prev_loss = val_loss
            last_improvement = 0
        else:
            print(f"val_loss did not improve from {prev_loss:.4f}.")
            last_improvement += 1

        # early stopping (only monitoring val_loss for now)
        if early_stopping_patience and last_improvement >= early_stopping_patience:
            print(f"val_loss did not improve in the last {last_improvement} epochs. Stopping training..")
            break


