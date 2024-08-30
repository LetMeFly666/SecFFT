import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
import logging
import copy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
random.seed(200)


class Worker:
    def __init__(
        self,
        idx,
        processor,
        model,
        data_set,
        device,
        class_names,
        rounds,
        round_to_start_attack,
        epochs,
        attack_type="",
        attack_params={},
        batch_size=256,
    ):
        self.device = device
        self.processor = processor
        self.model = model
        self.idx = idx
        self.data_set = data_set
        self.class_names = class_names
        self.rounds = rounds
        self.round = 0
        self.epochs = epochs
        self.attack_type = attack_type
        self.round_to_start_attack = round_to_start_attack
        self.attack_params = attack_params
        self.lr_init = 1e-2
        self.has_flip = False
        self.has_backdoor = False

        self.train_summaries = []

        # if self.attack_type == "gradient ascent":
        #     assert "attack_rate" in self.attack_params

        # if self.attack_type == "label flip":
        #     assert "source_labels" in self.attack_params
        #     assert "target_label" in self.attack_params
        #     assert "flip_rate" in self.attack_params
        #     # assert "bidirectional" in self.attack_params
        #     assert self.attack_params["source_labels"] != self.attack_params["target_label"]

        if self.attack_type == "backdoor":
            # assert "trigger" in self.attack_params
            assert "trigger_position" in self.attack_params
            assert "trigger_size" in self.attack_params
            assert "trigger_value" in self.attack_params
            assert "backdoor_rate" in self.attack_params
            assert "target_label" in self.attack_params

        self.data_loader = torch.utils.data.DataLoader(
            self.data_set, batch_size=batch_size, shuffle=True
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

    # def label_flip(self):
    #     original_targets = np.array(self.data_set.dataset.targets)
    #     subset_indices = self.data_set.indices
    #     all_flip_indices = []
    #     for source_label in self.attack_params["source_labels"]:
    #         target_indices = np.random.permutation(
    #             np.where(original_targets[subset_indices] == source_label)[0]
    #         )
    #         n_flip = int(len(target_indices) * self.attack_params["flip_rate"])
    #         flip_indices = target_indices[:n_flip]
    #         all_flip_indices.extend(flip_indices)

    #     for idx in all_flip_indices:
    #         self.data_set.dataset.targets[subset_indices[idx]] = self.attack_params[
    #             "target_label"
    #         ]

    #     source_labels_str = ", ".join(
    #         [self.class_names[label] for label in self.attack_params["source_labels"]]
    #     )
    #     logger.info(
    #         f"Flipped {len(all_flip_indices)} labels from {source_labels_str} "
    #         f"to {self.class_names[self.attack_params['target_label']]}"
    #     )

    def backdoor_attack(self):
        subset_indices = self.data_set.indices
        np.random.shuffle(subset_indices)
        n_backdoor = int(len(subset_indices) * self.attack_params["backdoor_rate"])
        backdoor_indices = subset_indices[:n_backdoor]
        for idx in backdoor_indices:
            image = self.data_set.dataset.data[idx]
            x, y = self.attack_params["trigger_position"]
            h, w = self.attack_params["trigger_size"]
            image[x : x + h, y : y + w, :] = self.attack_params["trigger_value"]
            self.data_set.dataset.targets[idx] = self.attack_params["target_label"]
            self.data_set.dataset.data[idx] = image
        target_label_name = self.class_names[self.attack_params["target_label"]]
        logger.info(
            f"Worker {self.idx}: Injected {n_backdoor} backdoor samples with label {target_label_name}"
        )

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)

    def train(self):
        target_model_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                target_model_params[name] = param.clone().detach().requires_grad_(False)

        # TODO
        lr = self.lr_init * 0.1 ** (self.round % 10)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        self.model.train()
        text_descriptions = [
            f"This is a photo of a {label}" for label in self.class_names
        ]
        tokenized_text = self.processor(
            text=text_descriptions, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        if (
            self.round >= self.round_to_start_attack
            and self.attack_type == "backdoor"
            and self.has_backdoor == False
        ):
            self.backdoor_attack()
            self.has_backdoor = True

        # assert len(self.data_loader) >= self.epochs
        batch_idxs = [i for i in range(len(self.data_loader))]
        if len(self.data_loader) <= self.epochs:
            selected_idxs = batch_idxs
        else:
            selected_idxs = random.sample(batch_idxs, self.epochs)

        for cur_idx, (images, labels) in enumerate(
            tqdm(self.data_loader, desc=f"Worker {self.idx} Round {self.round} Train")
        ):
            if cur_idx not in selected_idxs:
                continue

            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()

            text_features = self.model.get_text_features(**tokenized_text)
            image_features = self.model.get_image_features(images)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_probs = 100.0 * image_features @ text_features.T
            loss_value = self.loss_fn(text_probs, labels)

            loss_value.backward()
            self.optimizer.step()

            self.train_summaries.append(
                {
                    "epoch": self.round,
                    "loss": loss_value.item(),
                    "accuracy": text_probs.argmax(dim=-1)
                    .eq(labels)
                    .float()
                    .mean()
                    .item(),
                    "batch_size": images.size(0),
                }
            )
        self.scheduler.step()  # 是否是这个影响了准确率
        self.round += 1

        gradient = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    gradient[name] = param - target_model_params[name]
        # print(f"gradient: {gradient}")
        return gradient

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
