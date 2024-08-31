# backdoor on CIFAR-100 with 10 workers and 3 attackers
import os
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100
from transformers import CLIPProcessor
from peft import LoraConfig
import matplotlib.pyplot as plt
import seaborn as sns
from src import Worker
from src import Aggregator
from scipy.fftpack import dct
from transformers import CLIPProcessor, CLIPModel
from peft import get_peft_model
import logging
import pandas as pd
import warnings
import argparse
import copy


# command line arguments
parser = argparse.ArgumentParser(description="Run PEFT on CIFAR-100 dataset")
parser.add_argument("-n", "--num_workers", type=int, help="Number of workers")
parser.add_argument("-r", "--rounds", type=int, help="Number of rounds")
parser.add_argument(
    "-s", "--round_to_start_attack", type=int, help="Round to start attack"
)
parser.add_argument("-o", "--output_dir", type=str, help="Output directory")
parser.add_argument("-a", "--advertiser_num", type=int, help="Number of attackers")
args = parser.parse_args()

warnings.filterwarnings(
    "ignore", message=".*Torch was not compiled with flash attention.*"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(200)


# Load CIFAR-100 dataset
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
cache_dir = os.path.expanduser("~/Desktop/clip_lora/.cache/models")

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# load the processor
processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32", cache_dir=cache_dir
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert device.type == "cuda", "Please make sure CUDA is available"

# Load the CIFAR-100 dataset
cifar100_train = CIFAR100(
    os.path.expanduser("~\Desktop\clip_lora\.cache"),
    train=True,
    transform=transforms.ToTensor(),
    download=False,
)
cifar100_test = CIFAR100(
    os.path.expanduser("~\Desktop\clip_lora\.cache"),
    train=False,
    transform=transforms.ToTensor(),
    download=False,
)

class_names = cifar100_train.classes
train_size = len(cifar100_train)
test_size = len(cifar100_test)

test_loader = DataLoader(cifar100_test, batch_size=128, shuffle=False)

# config for federated learning
num_workers = args.num_workers
outer_rounds = args.rounds
round_to_start_attack = args.round_to_start_attack
advertiser_num = args.advertiser_num
local_epochs = 20
batch_size = 64
logger.info(
    f"Number of workers: {num_workers}, Number of rounds: {outer_rounds}, Round to start attack: {round_to_start_attack}, Number of attackers: {advertiser_num}"
)
attackers = [i for i in range(advertiser_num)]
attack_type = "backdoor"
attack_params = {
    "trigger_position": (0, 0),
    "trigger_size": (1, 1),
    "backdoor_rate": 0.5,
    "target_label": 88,
    "trigger_value": 1,
}

# Split the train dataset into {num_workers} subsets
subset_size = train_size // num_workers
remaining = train_size % num_workers
subset_lengths = [
    subset_size + 1 if i < remaining else subset_size for i in range(num_workers)
]
train_subsets = random_split(cifar100_train, subset_lengths)

logger.info(f"Train dataset size: {len(cifar100_train)}")
logger.info(f"Test dataset size: {len(cifar100_test)}")

target_modules = []
layers = [f"vision_model.encoder.layers.{i}" for i in range(12)]
modules = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "mlp.fc1",
    "mlp.fc2",
]
target_modules.extend([f"{layer}.{module}" for layer in layers for module in modules])
target_modules.append("visual_projection")
# logger.info(f"Target modules: {target_modules}")
config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.1,
    bias="none",
)

# loccal model
base_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32", cache_dir=cache_dir
)
local_model = get_peft_model(base_model, config)
local_model.print_trainable_parameters()
local_model.to(device)

# target model
base_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32", cache_dir=cache_dir
)
target_model = get_peft_model(base_model, config)
target_model.print_trainable_parameters()
target_model.to(device)


# Create workers for each subset
workers = []
for idx, train_subset in enumerate(train_subsets):
    if idx in attackers:
        worker = Worker(
            idx=idx,
            processor=processor,
            model=local_model,
            data_set=train_subset,
            device=device,
            class_names=class_names,
            inner_epochs=local_epochs,
            start_round=0,
            round_to_start_attack=round_to_start_attack,  # epoch_to_start_attack
            attack_type=attack_type,
            attack_params=attack_params,
            batch_size=batch_size,
        )
    else:
        worker = Worker(
            idx=idx,
            processor=processor,
            model=local_model,
            data_set=train_subset,
            device=device,
            class_names=class_names,
            inner_epochs=local_epochs,
            round_to_start_attack=outer_rounds + 1,
            batch_size=batch_size,
            start_round=0,
        )
    workers.append(worker)

# create the aggregator
aggregator = Aggregator(
    class_names=class_names,
    processor=processor,
    target_model=target_model,
    device=device,
)

normal_accuracy, normal_loss = aggregator.eval(test_loader)
logger.info(
    f"Round {-1} -Normal Accuracy: {normal_accuracy:.4f}, Loss: {normal_loss:.4f}"
)


csv_file = os.path.join(output_dir, "aggregator_summary.csv")
aggregator_summary = []

# save the target model parameters
target_model_params = {}
for name, param in target_model.named_parameters():
    if param.requires_grad:
        target_model_params[name] = param.clone().detach().requires_grad_(False)

for round in range(outer_rounds):
    gradients = {}
    for idx, worker in enumerate(workers):
        for name, param in local_model.named_parameters():
            if param.requires_grad:
                param.data = copy.deepcopy(target_model_params[name])
        gradient = worker.train(round)
        gradients[idx] = gradient

    flattend_gradients = []
    for idx, gradient in gradients.items():
        flattend_gradients.append(
            torch.cat([params.flatten() for params in gradient.values()])
        )
    gradients_tensor = torch.stack(flattend_gradients)
    # gradients_tensor = torch.stack(gradients)
    normalized_gradients = gradients_tensor / gradients_tensor.norm(dim=1, keepdim=True)
    cosine_similarity_matrix = (
        torch.mm(normalized_gradients, normalized_gradients.T).to("cpu").numpy()
    )

    # heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cosine_similarity_matrix, annot=False, cmap="Reds", cbar=True)
    plt.title(f"Cosine Similarity Matrix - round {round}")
    plt.xlabel("Worker")
    plt.ylabel("Worker")
    plt.savefig(os.path.join(output_dir, f"cosine_similarity_round_{round}.png"))
    plt.close()

    # save the gradients tensor
    torch.save(
        gradients_tensor, os.path.join(output_dir, f"gradients_tensor_round_{round}.pt")
    )

    # t-SNE
    # tsne = TSNE(n_components=2, random_state=200, perplexity=5)
    # grandients_tsne = tsne.fit_transform(gradients_tensor[:, -20481:-1].cpu().numpy()) # only the last layer
    # plt.figure(figsize=(10, 8))
    # plt.scatter(grandients_tsne[:, 0], grandients_tsne[:, 1])
    # plt.title(f"t-SNE - round {round}")
    # plt.xlabel("t-SNE 1")
    # plt.ylabel("t-SNE 2")
    # plt.savefig(os.path.join(output_dir, f"tsne_round_{round}.png"))
    # plt.close()

    gradients_tensor_copy = gradients_tensor.clone()
    # DCT transform
    gradients_numpy = gradients_tensor_copy.cpu().numpy()
    gradients_dct = dct(gradients_numpy, axis=1, norm="ortho")
    gradients_tensor_dct = torch.tensor(gradients_dct, dtype=torch.float32).to(device)
    # cosine similarity matrix after DCT
    normalized_gradients_dct = gradients_tensor_dct / gradients_tensor_dct.norm(
        dim=1, keepdim=True
    )
    cosine_similarity_matrix_dct = (
        torch.mm(normalized_gradients_dct, normalized_gradients_dct.T).to("cpu").numpy()
    )

    # heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cosine_similarity_matrix_dct, annot=False, cmap="Reds", cbar=True)
    plt.title(f"Cosine Similarity Matrix after DCT - round {round + 1}")
    plt.xlabel("Worker")
    plt.ylabel("Worker")
    plt.savefig(os.path.join(output_dir, f"cosine_similarity_dct_round_{round}.png"))
    plt.close()

    # t-SNE after DCT
    # tsne = TSNE(n_components=2, random_state=200, perplexity=5)
    # gradients_tsne_dct = tsne.fit_transform(gradients_tensor_dct.cpu().numpy())
    # plt.figure(figsize=(10, 8))
    # plt.scatter(gradients_tsne_dct[:, 0], gradients_tsne_dct[:, 1])
    # plt.title(f"t-SNE after DCT - round {round}")
    # plt.xlabel("t-SNE 1")
    # plt.ylabel("t-SNE 2")
    # plt.savefig(os.path.join(output_dir, f"tsne_dct_round_{round}.png"))
    # plt.close()

    mean_gradients = gradients_tensor.mean(dim=0)

    logger.info(f"Round {round} - Mean gradients: {mean_gradients.shape}")

    start = 0
    for name, param in target_model_params.items():
        end = start + param.numel()
        param.data += mean_gradients[start:end].reshape(param.shape)
        start = end

    assert start == mean_gradients.numel(), "The number of parameters does not match"

    for name, param in target_model.named_parameters():
        if param.requires_grad:
            param.data = copy.deepcopy(target_model_params[name])

    normal_accuracy, normal_loss = aggregator.eval(test_loader)
    print(
        f"Round {round} -Normal Accuracy: {normal_accuracy:.4f}, Loss: {normal_loss:.4f}"
    )
    backdoor_accuracy, backdoor_loss = aggregator.eval(test_loader, attack_params)
    print(
        f"Round {round} -Backdoor Accuracy on Test: {backdoor_accuracy:.4f}, Loss: {backdoor_loss:.4f}"
    )
    for idx in range(advertiser_num):
        backdoor_accuracy, backdoor_loss = aggregator.eval(
            workers[idx].data_loader, attack_params
        )
        print(
            f"Round {round} -Backdoor Accuracy on Worker {idx}: {backdoor_accuracy:.4f}, Loss: {backdoor_loss:.4f}"
        )
    aggregator_summary.append(
        {
            "normal_accuracy": normal_accuracy,
            "normal_loss": normal_loss,
            "backdoor_accuracy": backdoor_accuracy,
            "backdoor_loss": backdoor_loss,
        }
    )

    # save the summary of aggregator
    df = pd.DataFrame(aggregator_summary)
    df.to_csv(csv_file, index=False)

    # save the summary of worker
    for idx, worker in enumerate(workers):
        csv_file = os.path.join(output_dir, f"woker{idx}_summary.csv")
        df = pd.DataFrame(worker.train_summaries)
        df.to_csv(csv_file, index=False)


save_path = f"{output_dir}/models_cifar100"
os.makedirs(save_path)

aggregator.save(f"{save_path}/aggregator.pt")

for idx, worker in enumerate(workers):
    worker.save(f"{save_path}/worker_{idx}.pt")
