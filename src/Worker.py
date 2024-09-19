import torch
from tqdm import tqdm
import random
import logging


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
        round_to_start_attack,
        inner_epochs,
        start_round=0,
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
        self.epochs = inner_epochs
        self.start_round = start_round
        self.attack_type = attack_type
        self.round_to_start_attack = round_to_start_attack
        self.attack_params = attack_params
        self.lr_init = 1e-3
        self.has_flip = False
        self.backdoored = False

        self.train_summaries = []

        if self.attack_type == "backdoor":
            assert "trigger_position" in self.attack_params
            assert "trigger_size" in self.attack_params
            assert "trigger_value" in self.attack_params
            assert "backdoor_rate" in self.attack_params
            assert "target_label" in self.attack_params

        self.data_loader = torch.utils.data.DataLoader(
            self.data_set, batch_size=batch_size, shuffle=True
        )

    def add_trigger(self, images, labels):
        n_backdoor = int(len(images) * self.attack_params["backdoor_rate"])
        backdoor_indices = range(n_backdoor)
        x, y = self.attack_params["trigger_position"]
        h, w = self.attack_params["trigger_size"]
        print(
            f"Worker {self.idx} Add Trigger at ({x}, {y}) with size ({h}, {w}) images.shape: {images.shape}"
        )
        for idx in backdoor_indices:
            images[idx][:, x : x + h, y : y + w] = self.attack_params["trigger_value"]
            labels[idx] = self.attack_params["target_label"]

        return images, labels

    def train(self, cur_round):
        target_model_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                target_model_params[name] = param.clone().detach().requires_grad_(False)

        lr = self.lr_init * 0.1 ** ((cur_round - self.start_round) // 10)
        print(f"Worker {self.idx} Round {cur_round} Learning Rate: {lr}")
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.2,
        )

        loss_fn_image = torch.nn.CrossEntropyLoss()

        if (
            cur_round >= self.round_to_start_attack
            and self.attack_type == "backdoor"
            and self.backdoored == False
        ):
            self.backdoored = True

        self.model.train()
        text_inputs = [f"This is a photo of a {label}" for label in self.class_names]

        batch_idxs = [i for i in range(len(self.data_loader))]
        if len(self.data_loader) <= self.epochs:
            selected_idxs = batch_idxs
        else:
            selected_idxs = random.sample(batch_idxs, self.epochs)

        for cur_idx, (images, labels) in enumerate(
            tqdm(self.data_loader, desc=f"Worker {self.idx} Round {cur_round} Train")
        ):
            if cur_idx not in selected_idxs:
                continue
            optimizer.zero_grad()

            # TODO add trigger if self.backdoored
            if self.backdoored:
                images, labels = self.add_trigger(images, labels)

            inputs = self.processor(
                text=text_inputs,
                images=images,
                return_tensors="pt",
                padding=True,
                do_rescale=False,
            ).to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            loss = loss_fn_image(logits_per_image, labels)
            loss.backward()
            optimizer.step()

            accuracy = (logits_per_image.argmax(dim=1) == labels).sum().item()
            logger.info(
                f"Worker {self.idx} Round {cur_round} Train Loss: {loss.item()}, Accuracy: {accuracy / images.size(0)}"
            )
            self.train_summaries.append(
                {
                    "loss": loss.item(),
                    "accuracy": accuracy / images.size(0),
                }
            )

        gradient = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    gradient[name] = param - target_model_params[name]
        return gradient

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
