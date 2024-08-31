import torch
from tqdm import tqdm
import copy


class Aggregator:
    def __init__(
        self, class_names, processor, target_model, device=torch.device("cpu")
    ):
        self.device = device
        self.processor = processor
        self.class_names = class_names
        self.model = target_model

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def predict(self, images):
        self.model.eval()
        text_inputs = [f"This is a photo of a {label}" for label in self.class_names]
        with torch.no_grad():
            inputs = self.processor(
                text=text_inputs, images=images, return_tensors="pt", padding=True
            ).to(self.device)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            return logits_per_image.softmax(dim=1)

    def eval(self, data_loader, attack_params={}):
        self.model.eval()
        correct = 0
        loss = 0
        total = 0
        loss_fn_image = torch.nn.CrossEntropyLoss()
        text_inputs = [f"This is a photo of a {label}" for label in self.class_names]

        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc="Evaluating"):
                if attack_params:
                    x, y = attack_params["trigger_position"]
                    h, w = attack_params["trigger_size"]
                    for idx in range(len(images)):
                        images[idx][x : x + h, y : y + w, :] = attack_params[
                            "trigger_value"
                        ]
                        labels[idx] = attack_params["target_label"]

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

                probs = logits_per_image.softmax(dim=1)

                correct += (probs.argmax(dim=1) == labels).sum().item()
                loss += loss_fn_image(logits_per_image, labels).item() * len(images)

                total += len(images)

        return correct / total, loss / total
