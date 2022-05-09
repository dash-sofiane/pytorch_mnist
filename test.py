import torch 

def test(model, test_data, loss_function):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        sum_loss = 0
        total = 0
        for X, y in test_data:
            output = model(X)
            pred_y = torch.max(output, 1)[1].data.squeeze()

            correct += (pred_y == y).sum().item()
            sum_loss += loss_function(output, y)
            total += y.size(0)

        # calculating dataset accuracy and loss
        accuracy = correct / total
        loss = sum_loss / len(test_data)

    return loss, accuracy
    