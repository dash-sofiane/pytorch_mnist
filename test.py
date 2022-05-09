import torch 

def test(model, test_data, loss_function):
    # Test the model
    model.eval()
    with torch.no_grad():
        for images, labels in test_data:
            test_output, last_layer = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            loss = loss_function(test_output, labels)

    return loss, accuracy
    