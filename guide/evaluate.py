from typing import Dict, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(
    classifier: nn.Module,
    val_loaders: List[DataLoader],
    device,
    task_history: Dict[int, List[float]] = {},
):
    classifier.eval()
    current_accuracies = []

    with torch.no_grad():
        for task_id, loader in enumerate(val_loaders):
            correct = 0
            total = 0
            for batch in loader:
                images, labels_one_hot = batch
                labels_one_hot = labels_one_hot["y"]
                images = images.to(device)
                labels = torch.argmax(labels_one_hot, dim=1).to(device)
                outputs = classifier(images)
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = (correct / total) * 100
            current_accuracies.append(accuracy)

            # Store the accuracy for this task
            if task_id in task_history:
                task_history[task_id].append(accuracy)
            else:
                task_history[task_id] = [accuracy]

    # Calculate forgetting
    forgetting = []
    for task_id, history in task_history.items():
        if len(history) > 1:
            max_accuracy = max(history[:-1])  # Max accuracy excluding the most recent
            forgetting.append(max_accuracy - history[-1])
        else:
            forgetting.append(0)  # No forgetting for the most recent task

    return current_accuracies, forgetting, task_history
