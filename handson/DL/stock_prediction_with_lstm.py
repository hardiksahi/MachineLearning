## Stock prediction problem using LSTM
import pandas as pd
import numpy as np
import torch
from torch.nn import Module, LSTM, Dropout, Linear, MSELoss, Sigmoid
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import time


class GLU(Module):
    ## gated Linear Unit => Uses Sigmnoid as activation
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.lin_layer1 = Linear(in_features=input_dim, out_features=out_dim, bias=True)
        self.lin_layer2 = Linear(in_features=input_dim, out_features=out_dim, bias=True)
        self.sigmoid = Sigmoid()

    def __call__(self, x):
        out1 = self.lin_layer1(x)
        out2 = self.lin_layer2(x)
        out2 = self.sigmoid(out2)
        final = out1 * out2
        return final


class SwiGLU(Module):
    ## Swish Gated Linear Unit => Uses SwishBeta function as activation
    # def swishBeta()
    def __init__(self, input_dim, out_dim, beta):
        super().__init__()
        self.beta = beta
        self.lin_layer1 = Linear(in_features=input_dim, out_features=out_dim, bias=True)
        self.lin_layer2 = Linear(in_features=input_dim, out_features=out_dim, bias=True)

    def __call__(self, x):
        out1 = self.lin_layer1(x)
        out2 = self.lin_layer2(x)

        ## Apply SwishBeta function on out2
        betax = -1 * self.beta * out2
        inverse = 1 / (1 + torch.exp(betax))
        final = out2 * inverse

        return out1 * final


class Sequence(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lstm = LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=4,
            bias=True,
            batch_first=True,
            dropout=0.2,
        )
        self.dropout = Dropout(p=0.2)
        self.linear = Linear(in_features=self.hidden_size, out_features=1, bias=True)
        # self.linear = SwiGLU(input_dim=self.hidden_size, out_dim=1, beta=1)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.dropout(out)[
            :, -1, :
        ]  ## Gets output of last time step after dropout application
        out = self.linear(out)
        return out


class SequentialData(Dataset):
    ## https://medium.com/analytics-vidhya/pytorch-for-deep-learning-lstm-for-sequence-data-d0708fdf5717
    ## Automatically breaks data into batches by default:  https://pytorch.org/docs/stable/data.html#automatic-batching-default
    ## It is a Map style dataset (https://pytorch.org/docs/stable/data.html#map-style-datasets)
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


def create_sequences(data, seq_length=60):
    ## seq_length is number of days which will serve as input feature
    ## e.g. Day1 to Day 60 mid_price as input => Day61 mid_price as output
    X = []
    y = []
    for i in range(seq_length, len(data)):
        features = data[i - seq_length : i, 0]
        label = data[i, 0]
        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)


def predict(test_data_loader, model_path, input_size, hidden_size, seq_len):
    best_model = Sequence(input_size=input_size, hidden_size=hidden_size)
    best_model.load_state_dict(torch.load(model_path))
    best_model.eval()
    prediction_list = []
    label_list = []
    with torch.no_grad():
        for _, batch in enumerate(test_data_loader):
            data, label = batch
            y_pred = best_model(data.view(-1, seq_len, input_size)).reshape(-1)
            prediction_list.extend(y_pred.detach().numpy())
            label_list.extend(label.detach().numpy())

    return np.array(prediction_list), np.array(label_list)


if __name__ == "__main__":
    ##Step1: Read dataset
    df = pd.read_csv("data/aapl.us.txt")
    df.columns = [col.lower() for col in df.columns]
    print(f"Shape of df: {df.shape}")
    df["mid_price"] = (df["high"] + df["low"]) / 2
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["date"])
    df["data_type"] = "train"
    df.loc[df["date"].dt.year.isin([2015, 2016]), "data_type"] = "val"
    df.loc[df["date"].dt.year.isin([2017]), "data_type"] = "test"
    print(df.head())

    # test_df = df[df.data_type == "test"][["date"]]
    # line_plot = sns.lineplot(data=df, x="date", y="mid_price", hue="data_type")
    # plt.show()

    # print(f"Min date: {df['date'].min()}")
    # print(f"Max date: {df['date'].max()}")

    ## Break data into train and test set
    train_data = df[df.data_type == "train"]["mid_price"].values.reshape(-1, 1)
    val_data = df[df.data_type == "val"]["mid_price"].values.reshape(-1, 1)
    test_data = df[df.data_type == "test"]["mid_price"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data_scaled = scaler.fit_transform(train_data)
    val_data_scaled = scaler.transform(val_data)
    test_data_scaled = scaler.transform(test_data)

    seq_len = 60
    train_X, train_y = create_sequences(train_data_scaled, seq_length=seq_len)
    val_X, val_y = create_sequences(val_data_scaled, seq_length=seq_len)
    test_X, test_y = create_sequences(test_data_scaled, seq_length=seq_len)
    # train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], 1))
    # train_y = train_y.reshape((train_y.shape[0], 1, 1))

    # test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], 1))
    # test_y = test_y.reshape((test_y.shape[0], 1, 1))
    print(f"Shape of train_X: {train_X.shape}, train_y: {train_y.shape}")
    print(f"Shape of val_X: {val_X.shape}, val_y: {val_y.shape}")
    print(f"Shape of test_X: {test_X.shape}, val_y: {test_y.shape}")

    # train_loader = DataLoader(list(zip(train_X, train_y)), shuffle=False, batch_size=32)

    train_dataset = SequentialData(train_X, train_y)
    val_dataset = SequentialData(val_X, val_y)
    test_dataset = SequentialData(test_X, test_y)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    val_dataloader = DataLoader(val_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    input_size = 1  ## Dimension of input to LSTM at a time step
    hidden_size = 50  ## Dimension of output/ previous state at a time step

    # iiter = iter(train_dataloader)
    # batch1 = next(iiter)
    # print(f"Len of batch1: {len(batch1)}")
    # print(f"Shape of batch1[0]: {batch1[0].size()}")
    # print(f"Shape of batch1[1]: {batch1[1].size()}")

    # print(batch1[0].view(-1, seq_len, input_size).size())

    model = Sequence(input_size=input_size, hidden_size=hidden_size)
    print(model)
    n_epochs = 100  # 100
    loss_function = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    training_data_epoch_loss = []  ## Length will be equal to number of epochs
    val_data_epoch_loss = []  ## Length will be equal to number of epochs

    start_time = time.time()
    best_val_loss = float("inf")
    epochs_no_improve = 0
    early_stop = False
    patience = 5
    for epoch in range(n_epochs):
        print(f"Start epoch: {epoch+1}")
        running_loss = 0.0
        train_batch_loss = []
        model.train()
        for batch_index, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            data, label = batch
            # data = batch[0]
            # label = batch[1]
            ## reshape data
            y_pred = model(data.view(-1, seq_len, input_size)).reshape(
                -1
            )  ## Tensor of (batch_size,1) to (batch_size, )
            loss = loss_function(y_pred, label)
            batch_loss = loss.item()
            train_batch_loss.append(batch_loss)
            running_loss += batch_loss

            loss.backward()
            optimizer.step()

            if batch_index % 10 == 0:
                ## Running loss every 10 batches (10*batch_size = 320 examples)
                print(
                    f"[Epoch: {epoch+1}/{n_epochs}], Mini Batch: [{batch_index+1}/{len(train_dataloader)}], Loss: {running_loss}"
                )
                running_loss = 0.0
        mean_train_loss_for_epoch = np.array(train_batch_loss).mean()
        training_data_epoch_loss.append(
            mean_train_loss_for_epoch
        )  ## Average loss in the epoch

        ## Evaluate on validation dataset
        model.eval()
        val_batch_loss = []
        with torch.no_grad():
            for batch_index, batch in enumerate(val_dataloader):
                data, label = batch
                y_pred = model(data.view(-1, seq_len, input_size)).reshape(-1)
                loss = loss_function(y_pred, label)
                loss_value = loss.item()
                val_batch_loss.append(loss_value)
        mean_val_loss_for_epoch = np.array(val_batch_loss).mean()
        val_data_epoch_loss.append(mean_val_loss_for_epoch)
        print(f"Epoch: {epoch+1} mean_val_loss_for_epoch: {mean_val_loss_for_epoch}")

        if mean_val_loss_for_epoch < best_val_loss:
            best_val_loss = mean_val_loss_for_epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print("Early stopping!")
            early_stop = True
            break

    # if early_stop:
    #     model.load_state_dict(torch.load("best_model.pth"))
    #     model.eval()

    end_time = time.time()
    print(f"Training finished in {(end_time-start_time)/60} min")
    print("Best model saved at best_model.pth")

    test_prediction_list, test_label_list = predict(
        test_dataloader, "best_model.pth", input_size, hidden_size, seq_len
    )
    test_prediction = scaler.inverse_transform(
        test_prediction_list.reshape(-1, 1)
    ).reshape(-1)
    test_labels = scaler.inverse_transform(test_label_list.reshape(-1, 1)).reshape(-1)
    print(f"Shape of test_labels: {test_labels.shape}")
    print(f"Shape of test_prediction: {test_prediction.shape}")
    print(f"""len: {len(df[df.data_type == "test"]["date"].values[60:])}""")
    ## Plot test_prediction and test_labels
    test_prediction_df = pd.DataFrame(
        {
            "stock_price": test_prediction,
            "date": df[df.data_type == "test"]["date"].values[60:],
            "ttype": "prediction",
        }
    )
    test_label_df = pd.DataFrame(
        {
            "stock_price": test_labels,
            "date": df[df.data_type == "test"]["date"].values[60:],
            "ttype": "labels",
        }
    )
    combined_test_df = pd.concat([test_prediction_df, test_label_df], axis=0)

    lineplt = sns.lineplot(
        data=combined_test_df, x="date", y="stock_price", hue="ttype"
    )
    lineplt.figure.savefig("swiglu_test.png")

    # train_loss_df = pd.DataFrame(
    #     {
    #         "loss": training_data_epoch_loss,
    #         "epoch": range(len(training_data_epoch_loss)),
    #         "loss_type": "train",
    #     }
    # )
    # val_loss_df = pd.DataFrame(
    #     {
    #         "loss": val_data_epoch_loss,
    #         "epoch": range(len(val_data_epoch_loss)),
    #         "loss_type": "val",
    #     }
    # )
    # loss_df = pd.concat([train_loss_df, val_loss_df], axis=0)

    # lineplt = sns.lineplot(data=loss_df, x="epoch", y="loss", hue="loss_type")
    # plt.show()
