import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
from torch.nn import Module, Sequential, Linear, ReLU, Sigmoid, BCELoss, BatchNorm1d
from torch.utils.data import DataLoader
from torch.optim import Adam
import time
import datetime
from sklearn.metrics import accuracy_score


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.layers = Sequential(
            *[
                Linear(in_features=2, out_features=50, bias=True),
                BatchNorm1d(num_features=50),
                ReLU(),
                Linear(in_features=50, out_features=1, bias=True),
                Sigmoid(),
            ]
        )

    def forward(self, x):
        return self.layers(x)


# def train(model, train_loader, loss_function, optimizer, n_epochs):
#     # n_epochs = train_dict.get("n_epochs")
#     # train_loader = train_dict.get("train_loader")
#     # optimizer = train_dict.get("optimizer")
#     model.train()
#     start_time = time.time()
#     for epoch in range(n_epochs):
#         print(f"Start epoch: {epoch}")
#         running_loss = 0.0
#         for batch_count, batch in enumerate(train_loader):
#             inputs, labels = batch
#             optimizer.zero_grad()  ## Clear out accuulated gradient in x.grad before training on batch
#             y_pred = model(inputs)  ## Get predictions on batch
#             loss = loss_function(y_pred, labels)  ## Get loss on batch
#             loss.backward()  ## Calculate dLoss/dx for all params x and accumulate in x.grad
#             optimizer.step()  ## Update params as per optimizer using accumulated gradient
#             running_loss += loss.item()

#             if batch_count % 10 == 0:
#                 print(
#                     f"Epoch: {epoch+1}, Minibatch: {batch_count+1}, Loss: {running_loss}"
#                 )
#                 running_loss = 0.0
#     print("Training finished")
#     end_time = time.time()
#     # print("Training process has been completed. ")
#     print(
#         "Training time:", str(datetime.timedelta(seconds=end_time - start_time))
#     )  # for calculating the training time in minutes and seconds format

#     return model


# def train_one_epoch(epoch_index):


if __name__ == "__main__":
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    # df = pd.DataFrame(X, columns=["x", "y"])
    # df["label"] = y
    # scatter = sns.scatterplot(data=df, x="x", y="y", hue="label")
    # plt.show()

    ## SPlit the data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=16)
    val_loader = DataLoader(list(zip(X_test, y_test)), batch_size=16)

    ## Initialize the model
    model = MLP()
    # print(f"model: {model}")
    n_epochs = 100
    loss_function = BCELoss()
    optimizer = Adam(params=model.parameters(), lr=1e-3)

    start_time = time.time()
    training_epoch_loss = []
    validation_epoch_loss = []

    training_epoch_accuracy_metric = []
    validation_epoch_accuracy_metric = []
    for epoch in range(n_epochs):
        print(f"Start epoch: {epoch}")
        model.train()
        batch_step_loss = []  ## Loss for every batch in epoch
        training_gt = []
        training_pred = []
        running_loss = 0.0
        for batch_count, batch in enumerate(train_loader):
            inputs, labels = batch
            optimizer.zero_grad()  ## Clear out accuulated gradient in x.grad before training on batch
            y_pred = model(inputs)  ## Get predictions on batch
            # if batch_count == 0:
            #     print(y_pred)
            loss = loss_function(y_pred, labels)  ## Get loss on batch
            # print(f"loss: {loss}")
            loss.backward()  ## Calculate dLoss/dx for all params x and accumulate in x.grad
            optimizer.step()  ## Update params as per optimizer using accumulated gradient

            batch_loss = loss.item()  ## Loss for this minibatch
            batch_step_loss.append(batch_loss)  ## Per mini batch loss in an epoch
            running_loss += loss.item()
            training_gt.extend(labels.detach().numpy())
            training_pred.extend(torch.max(y_pred, 1).indices.detach().numpy())
            # print(f"loss in batch: {loss.item()}")
            if batch_count % 10 == 0:
                ## Displays loss every 10 minibatches and the resets it
                print(
                    f"Epoch: [{epoch+1}/{n_epochs}], Minibatch: [{batch_count+1}/{len(train_loader)}], Loss: {running_loss}"
                )
                running_loss = 0.0

        ## epoch finished
        training_epoch_loss.append(np.array(batch_step_loss).mean())
        ## Calculate metrics on training data
        training_epoch_accuracy_metric.append(
            accuracy_score(training_gt, training_pred)
        )

        ## Evaluate on evaluation dataset
        model.eval()
        validation_step_loss = []
        validation_gt = []
        validation_pred = []
        for val_batch_index, val_batch in enumerate(val_loader):
            val_inputs, val_labels = val_batch
            val_y_pred = model(val_inputs)
            validation_loss = loss_function(val_y_pred, val_labels)
            validation_batch_loss = validation_loss.item()
            validation_step_loss.append(validation_batch_loss)
            validation_gt.extend(val_labels.detach().numpy())
            validation_pred.extend(torch.max(val_y_pred, 1).indices.detach().numpy())
        validation_epoch_loss.append(np.array(validation_step_loss).mean())
        validation_epoch_accuracy_metric.append(
            accuracy_score(validation_gt, validation_pred)
        )

    print("Training finished")
    end_time = time.time()
    # print("Training process has been completed. ")
    print(
        "Training time:", str(datetime.timedelta(seconds=end_time - start_time))
    )  # for calculating the training time in minutes and seconds format

    print(f"training_epoch_accuracy_metric: {training_epoch_accuracy_metric}")
    print(f"validation_epoch_accuracy_metric: {validation_epoch_accuracy_metric}")

    train_loss_df = pd.DataFrame(
        {"loss": training_epoch_loss, "epoch": range(n_epochs), "loss_type": "train"}
    )
    val_loss_df = pd.DataFrame(
        {"loss": validation_epoch_loss, "epoch": range(n_epochs), "loss_type": "val"}
    )
    loss_df = pd.concat([train_loss_df, val_loss_df], axis=0)

    lineplt = sns.lineplot(data=loss_df, x="epoch", y="loss", hue="loss_type")
    plt.show()
