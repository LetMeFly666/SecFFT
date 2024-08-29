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
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def predict(self, images):
        self.model.eval()
        text_descriptions = [
            f"This is a photo of a {label}" for label in self.class_names
        ]
        tokenized_text = self.processor(
            text=text_descriptions, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        text_features = self.model.get_text_features(**tokenized_text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            images = images.to(self.device)
            image_features = self.model.get_image_features(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_probs = 100.0 * image_features @ text_features.T
            return text_probs.argmax(dim=-1)

    def eval(self, data_loader):
        self.model.eval()
        text_descriptions = [
            f"This is a photo of a {label}" for label in self.class_names
        ]
        tokenized_text = self.processor(
            text=text_descriptions, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        text_features = self.model.get_text_features(**tokenized_text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        correct = 0
        loss = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                image_features = self.model.get_image_features(images)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                text_probs = 100.0 * image_features @ text_features.T
                batch_loss = self.loss_fn(text_probs, labels)
                loss += batch_loss.item() * labels.size(0)
                correct += text_probs.argmax(dim=-1).eq(labels).sum().item()
                total += labels.size(0)

        return correct / total, loss / total
