import torch
from typing import Union, Collection


class FitTestClass:
    """_summary_"""

    def __init__(self, model: torch.nn.Module, config: dict) -> None:
        self.model = model
        self.config = config

        # TODO!
        # Configure dictionary should take in configurations for the testing.
        # Paths, learning rates - early stopping rounds etc.

    def test(
        self,
        test_data: torch.Tensor,
        loss_function: Union[
            torch.nn.CrossEntropyLoss, torch.nn.MSELoss
        ] = torch.nn.CrossEntropyLoss,
    ) -> Collection[float]:
        """_summary_

        Args:
            test_data (torch.Tensor): _description_
            loss_function (Union[ torch.nn.CrossEntropyLoss, torch.nn.MSELoss ], optional): _description_. Defaults to torch.nn.CrossEntropyLoss.

        Returns:
            Collection[float, float]: _description_
        """
        # Test the model
        self.model.eval()
        with torch.no_grad():
            sum_correct = 0
            sum_loss = 0
            total = 0
            for X, y in test_data:
                output = self.model(X)
                pred_y = torch.max(output, 1)[1].data.squeeze()

                sum_correct += (pred_y == y).sum().item()
                sum_loss += loss_function(output, y)
                total += y.size(0)

            # calculating dataset accuracy and loss
            accuracy = sum_correct / total
            loss = sum_loss / len(test_data)

        return loss, accuracy

    def train(
        self,
        train_data: torch.Tensor,
        validation_data: torch.Tensor,
        num_epochs: int,
        optimizer: torch.optim.Adam,
        model_save_path: str = "./last_trained_model.pt",
        early_stopping_patience: Union[None, int] = None,
        loss_function: Union[
            torch.nn.CrossEntropyLoss, torch.nn.MSELoss
        ] = torch.nn.CrossEntropyLoss,
    ) -> None:
        """_summary_

        Args:
            train_data (torch.Tensor): _description_
            validation_data (torch.Tensor): _description_
            num_epochs (int): _description_
            optimizer (torch.optim.Adam): _description_
            model_save_path (str, optional): _description_. Defaults to "./last_trained_model.pt".
            early_stopping_patience (Union[None, int], optional): _description_. Defaults to None.
            loss_function (Union[ torch.nn.CrossEntropyLoss, torch.nn.MSELoss ], optional): _description_. Defaults to torch.nn.CrossEntropyLoss.
        """

        # TODO
        # you have far far too many inputs here. The data for example should be
        # passed to the training class. Same as stuff as the model and the
        # optimizer. If you do this it will be super simple to make another
        # script that uses a different type of nn and has another structure.

        self.model.train()

        # Train the model
        total_step = len(train_data)

        # for early stopping
        last_improvement = 0
        prev_loss = float("inf")

        for epoch in range(num_epochs):
            for i, (X, y) in enumerate(train_data):

                output = self.model(X)
                loss = loss_function(output, y)

                # clear gradients for this training step
                optimizer.zero_grad()

                # backpropagation, compute gradients
                loss.backward()
                # apply gradients
                optimizer.step()

                # print the results per step
                if (i + 1) % 100 == 0:
                    print(
                        f"    epoch [{epoch + 1}/{num_epochs}], step [{i + 1}/{total_step}], loss: {loss.item():.4f}"
                    )

            # calculate loss & accuracy on validation data
            val_loss, val_acc = self.test(
                test_data=validation_data, loss_function=loss_function
            )
            print(
                f"epoch [{epoch + 1}/{num_epochs}], val_loss: {val_loss:.4f}, val_acc {val_acc:.4f}"
            )

            # saving the model if it improved (only monitoring val_loss for now)
            if val_loss < prev_loss:
                print(f"val_loss improved from {prev_loss:.4f} to {val_loss:.4f}.")
                print(f"Saving model in {model_save_path}")
                torch.save(self.model.state_dict(), model_save_path)

                prev_loss = val_loss
                last_improvement = 0
            else:
                print(f"val_loss did not improve from {prev_loss:.4f}.")
                last_improvement += 1

            # early stopping (only monitoring val_loss for now)
            if early_stopping_patience and last_improvement >= early_stopping_patience:
                print(
                    f"val_loss did not improve in the last {last_improvement} epochs. Stopping training.."
                )
                break
