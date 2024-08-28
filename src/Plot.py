import numpy as np


def plot_loss_curve(ax, loss_values):
    ax.plot(loss_values)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve Per Epoch")
    ax.grid(True)
    ax.set_xticks(range(0, len(loss_values), 10))

def plot_accuracy_curve(ax, accuracy_values):
    ax.plot(accuracy_values)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training Accuracy Curve Per Epoch")
    ax.grid(True)
    ax.set_xticks(range(0, len(accuracy_values), 10))

def plot_epoch_loss_curve(ax, loss_summaries, train_subset_length):
    epoch_losses = np.array([summary["loss"] for summary in loss_summaries]).reshape(
        -1, train_subset_length
    )
    epoch_losses = np.mean(epoch_losses, axis=1)
    ax.plot(epoch_losses)
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve Per Round")
    ax.grid(True)

def plot_epoch_accuracy_curve(ax, loss_summaries, train_subset_length):
    epoch_accuracies = np.array(
        [summary["accuracy"] for summary in loss_summaries]
    ).reshape(-1, train_subset_length)
    epoch_accuracies = np.mean(epoch_accuracies, axis=1)
    ax.plot(epoch_accuracies)
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training Accuracy Curve Per Round")
    ax.grid(True)
    