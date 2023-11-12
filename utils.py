import torch


def accuracy(labels, outputs):
    preds = outputs.argmax(-1)
    #acc = (preds == labels.view_as(preds)).float().detach().numpy().mean()
    acc = preds.eq(labels.view_as(preds)).sum().item()

    return acc


def save_model(model, path):
    torch.save(model.state_dict(), path)