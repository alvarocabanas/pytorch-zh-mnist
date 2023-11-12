import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from ray import tune
import ray

from typing import Tuple
from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
train_losses = []
test_losses = []
train_accs = []
test_accs = []

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim,
    criterion: torch.nn.functional,
    log_interval: int,
) -> Tuple[float, float]:
    # Activate the train=True flag inside the model
    model.train()

    train_loss = []
    acc = 0.
    avg_weight = 0.1
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        acc += accuracy(target, output)
        train_loss.append(loss.item())

        if batch_idx % log_interval == 0:
            print('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    avg_acc = 100. * acc / len(train_loader.dataset)

    return np.mean(train_loss), avg_acc


def eval_single_epoch(
    test_loader: DataLoader,
    model: nn.Module,
    criterion: torch.nn.functional,
) -> Tuple[float, float]:
    model.eval()

    test_loss = []
    acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss.append(criterion(output, target).item())
            acc += accuracy(target, output)

        # Average accuracy across all correct predictions batches now
        test_acc = 100. * acc / len(test_loader.dataset)
        test_loss = np.mean(test_loss)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, acc, len(test_loader.dataset), test_acc,
        ))

    return test_loss, test_acc


def train_model(config, train_dataset, val_dataset):
    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=config['test_batch_size'],
        shuffle=False,
        drop_last=True,
    )

    my_model = MyModel().to(device)
    optimizer = torch.optim.RMSprop(my_model.parameters(), lr=config["learning_rate"])
    criterion = nn.NLLLoss(reduction='mean')

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        train_loss, train_acc = train_single_epoch(train_loader, my_model, optimizer, criterion, config["log_interval"])
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        test_loss, test_accuracy = eval_single_epoch(test_loader, my_model, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_accuracy)

    return my_model


def test_model(config, model, test_dataset):
    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=config['test_batch_size'],
        shuffle=False,
        drop_last=True,
    )

    my_model = MyModel().to(device)
    optimizer = torch.optim.RMSprop(my_model.parameters(), lr=config["learning_rate"])
    criterion = nn.NLLLoss(reduction='mean')

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch + 1}/{config['epochs']}")

        test_loss, test_accuracy = eval_single_epoch(test_loader, my_model, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_accuracy)

    return my_model


if __name__ == "__main__":
    dataset_path = "/Users/acabanas/Documents/chinese_mnist/data/"
    labels_path = "/Users/acabanas/Documents/chinese_mnist/chinese_mnist.csv"

    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])

    my_dataset = MyDataset(dataset_path, labels_path, transform)
    train_ds, test_ds, valid_ds = torch.utils.data.random_split(my_dataset, (10000, 2500, 2500))
    ray.init(configure_logging=False)
    analysis = tune.run(
        train_model,
        metric="val_loss",
        mode="min",
        num_samples=5,
        config={
            "hyperparam_1": tune.uniform(1, 10),
            "hyperparam_2": tune.grid_search(["relu", "tanh"]),
        })

    print("Best hyperparameters found were: ", analysis.best_config)
    print(test_model(analysis.best_config, train_model, test_ds))