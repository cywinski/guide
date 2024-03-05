import os

import torch
import numpy as np

from evaluations.fid import calculate_frechet_distance
from evaluations.prd import compute_prd_from_embedding, prd_to_max_f_beta_pair
from torchmetrics.image.kid import poly_mmd


class Validator:
    def __init__(
        self,
        n_classes,
        device,
        dataset,
        stats_file_name,
        fid_dataloaders,
        clf_dataloaders,
        score_model_device=None,
        force_inception_for_fid=False,
    ):
        self.n_classes = n_classes
        self.device = device
        if not score_model_device:
            score_model_device = device
        self.dataset = dataset
        self.score_model_device = score_model_device
        self.fid_dataloaders = fid_dataloaders # train and test data or only test data
        self.clf_dataloaders = clf_dataloaders # always only test data
        self.force_inception_for_fid = force_inception_for_fid
        self.calculate_inception_score = False
        self.used_model_name = None
        self.classifier_loss = torch.nn.CrossEntropyLoss()

        print("Preparing validator")
        if (
            dataset in ["MNIST", "Omniglot"] and not force_inception_for_fid
        ):  # , "DoubleMNIST"]:
            if dataset in ["Omniglot"]:
                from evaluations.evaluation_models.lenet_Omniglot import Model
            else:
                from evaluations.evaluation_models.lenet import Model
            self.used_model_name = "lenet"
            net = Model()
            model_path = "evaluations/evaluation_models/lenet_" + dataset
            net.load_state_dict(torch.load(model_path))
            net.to(device)
            net.eval()
            self.dims = 128 if dataset in ["Omniglot", "DoubleMNIST"] else 84  # 128
            self.score_model_func = net.part_forward

            if dataset == "MNIST":
                self.calculate_inception_score = True
                self.get_logits_func = net.forward
        else:
            from evaluations.evaluation_models.inception import InceptionV3

            self.used_model_name = "inception"
            self.calculate_inception_score = True
            self.dims = 2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
            #  As we already have images in range [-1, 1], do not normalize in Inception (see constructor docs).
            model = InceptionV3([block_idx, 4], normalize_input=False).to(device)
            if score_model_device:
                model = model.to(score_model_device)
            model.eval()
            self.score_model_func = lambda batch: model(batch)
        self.stats_file_name = f"{stats_file_name}_dims_{self.dims}"

    @torch.no_grad()
    def calculate_accuracy_with_classifier(self, model, task_id, train_loader=None, max_class=0, train_with_disjoint_classifier=False):
        model.eval()
        loader = (
            {"test": self.clf_dataloaders[task_id], "train": train_loader}
            if train_loader is not None
            else {"test": self.clf_dataloaders[task_id]}
        )
        correct = {"test": 0.0, "train": 0.0}
        total = {"test": 0.0, "train": 0.0}
        loss = {"test": 0.0, "train": 0.0}
        print("Calculating accuracy:")
        for phase in loader.keys():
            for idx, batch in enumerate(loader[phase]):
                x, cond = batch
                x = x.to(self.device)
                y = cond["y"].to(self.device)
                out_classifier = model.classify(x) if not train_with_disjoint_classifier else model(x)
                preds = torch.argmax(out_classifier, 1)
                correct[phase] += (preds == torch.argmax(y, 1)).sum()
                total[phase] += len(y)
                loss[phase] += self.classifier_loss(out_classifier[:, :max_class + 1], y[:, :max_class + 1])
            loss[phase] /= idx
            correct[phase] /= total[phase]
        model.train()
        return {
            "loss": loss,
            "accuracy": correct,
        }

    @torch.no_grad()
    def calculate_results(
        self, train_loop, task_id, n_generated_examples, dataset=None, batch_size=128
    ):
        distribution_orig = []
        distribution_gen = []
        gen_logits = []

        stats_dir = os.path.join(
            os.environ.get("DIFFUSION_DATA", "results"), "orig_stats"
        )

        precalculated_statistics = False
        os.makedirs(stats_dir, exist_ok=True)
        stats_file_path = (
            f"{stats_dir}/{self.dataset}_{self.stats_file_name}_{task_id}.npy"
        )
        if os.path.exists(stats_file_path):
            print(
                f"Loading cached original data statistics from: {self.stats_file_name}"
            )
            distribution_orig = np.load(stats_file_path)
            precalculated_statistics = True

        print("Calculating FID:")
        if not precalculated_statistics:
            # Calculate FID on original samples from tasks 0..task_id
            for test_loader in [self.fid_dataloaders[i] for i in range(task_id + 1)]:
                for idx, batch in enumerate(test_loader):
                    x, cond = batch
                    x = x.to(self.device)
                    if (dataset.lower() in ["fashionmnist", "doublemnist"]) or (
                        dataset.lower() in ["mnist", "omniglot"]
                        and self.force_inception_for_fid
                    ):
                        x = x.repeat([1, 3, 1, 1])
                    distribution_orig.append(
                        self.score_model_func(x.to(self.score_model_device))[0]
                        .cpu()
                        .detach()
                        .numpy()
                    )
            distribution_orig = np.array(np.concatenate(distribution_orig)).reshape(
                -1, self.dims
            )
            np.save(stats_file_path, distribution_orig)

        examples_to_generate = n_generated_examples
        while examples_to_generate > 0:
            example, _, _ = train_loop.generate_examples(
                task_id=task_id,
                n_examples_per_task=min(batch_size, examples_to_generate),
                only_one_task=True,
                use_diffusion_for_validation=True,
            )
            example = example.to(self.score_model_device)
            if (dataset.lower() in ["fashionmnist", "doublemnist"]) or (
                dataset.lower() in ["mnist", "omniglot"]
                and self.force_inception_for_fid
            ):
                example = example.repeat([1, 3, 1, 1])
            model_func_out = self.score_model_func(example)
            dist_reps = (
                model_func_out[0]
                if self.used_model_name == "inception"
                else model_func_out
            )
            distribution_gen.append(dist_reps.cpu().detach())

            if self.calculate_inception_score:
                if self.used_model_name == "inception":
                    gen_logits.append(model_func_out[1])
                else:
                    gen_logits.append(self.get_logits_func(example))

            examples_to_generate -= batch_size
        distribution_gen = torch.cat(distribution_gen).numpy().reshape(-1, self.dims)
        if self.calculate_inception_score:
            gen_logits = torch.cat(gen_logits)

        num_data_for_prd = min(len(distribution_orig), len(distribution_gen))
        precision, recall = compute_prd_from_embedding(
            eval_data=distribution_gen[
                np.random.choice(len(distribution_gen), num_data_for_prd, replace=False)
            ],
            ref_data=distribution_orig[
                np.random.choice(
                    len(distribution_orig), num_data_for_prd, replace=False
                )
            ],
        )
        precision, recall = prd_to_max_f_beta_pair(precision, recall)
        print(f"Precision: {precision}, recall: {recall}")
        fid = calculate_frechet_distance(distribution_gen, distribution_orig)
        kid = float(
            poly_mmd(
                torch.tensor(distribution_gen, device=self.device),
                torch.tensor(distribution_orig, device=self.device),
            )
        )

        inception_score = (
            calculate_is(gen_logits) if self.calculate_inception_score else -1
        )

        return {
            "fid": fid,
            "kid": kid,
            "is": inception_score,
            "precision": precision,
            "recall": recall,
        }


def calculate_is(logits):
    prob = logits.softmax(dim=1)
    log_prob = logits.log_softmax(dim=1)
    mean_prob = prob.mean(dim=0, keepdim=True)
    kl_ = prob * (log_prob - mean_prob.log())
    kl = kl_.sum(dim=1).mean().exp()
    return float(kl)
