import torch
import torch.nn.functional as F


@torch.no_grad()
def calculate_accuracy_with_classifier(
    model,
    task_id,
    val_loader,
    device,
    train_loader=None,
    max_class=0,
    train_with_disjoint_classifier=False,
):
    model.eval()
    loader = (
        {"test": val_loader, "train": train_loader}
        if train_loader is not None
        else {"test": val_loader}
    )
    correct = {"test": 0.0, "train": 0.0}
    total = {"test": 0.0, "train": 0.0}
    loss = {"test": 0.0, "train": 0.0}
    print("Calculating accuracy:")
    for phase in loader.keys():
        for idx, batch in enumerate(loader[phase]):
            x, cond = batch
            x = x.to(device)
            y = cond["y"].to(device)
            out_classifier = model(x)
            preds = torch.argmax(out_classifier, 1)
            correct[phase] += (preds == torch.argmax(y, 1)).sum()
            total[phase] += len(y)
            loss[phase] += F.cross_entropy(
                out_classifier[:, : max_class + 1], y[:, : max_class + 1]
            )
        loss[phase] /= idx
        correct[phase] /= total[phase]
    model.train()
    return {
        "loss": loss,
        "accuracy": correct,
    }
