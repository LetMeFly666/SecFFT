'''
Author: LetMeFly
Date: 2024-08-11 17:24:10
LastEditors: LetMeFly
LastEditTime: 2024-08-13 10:23:42
'''
import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
from typing import Optional, Tuple


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()
    return args


def run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    print("\n-------- Searching hyperparameters on the val set. --------")
    # Zero-shot CLIP
    clip_logits = 100. * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta: int = cfg['init_beta']
    alpha: int = cfg['init_alpha']
    
    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)

    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP  # 使用 CLIP 模型直接对验证集特征进行推理，计算初始的分类准确率。
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter    
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))


def run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F, adapter: Optional[nn.Linear]=None):
    # 如果传入了 `adapter`，使用聚合后的适配器；否则创建新的适配器
    if adapter is None:
        adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
        adapter.weight = nn.Parameter(cache_keys.t())
    else:
        adapter = adapter.to(clip_model.dtype).cuda()
    print(f'adapter.weight.size() - just run_tip_adapter_F\'s param : {adapter.weight.size()}')  # torch.Size([102, 1024])

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    beta: int = cfg['init_beta']
    alpha: int = cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        print(f'adapter.weight.size() - after train : {adapter.weight.size()}')  # torch.Size([102, 1024])
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            print("Affinity shape:", affinity.shape)  # torch.Size([256, 102])
            print("Cache values shape:", cache_values.shape)  # torch.Size([1632, 102])
            affinity = affinity.to(cache_values.dtype)
            print("Adapter weight shape:", adapter.weight.shape)  # torch.Size([102, 1024])
            print("Adapter weight dtype:", adapter.weight.dtype)  # torch.float16

            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()

        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = 100. * test_features @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, test_labels)

        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights, adapter=adapter)

    print("\n-------- Evaluating on the test set. --------")
   
    affinity = adapter(test_features)
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))


def client_training(client_id: int, cfg: dict, clip_model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, nn.Module]:
    """
    在客户端执行的训练和缓存构建。
    """
    random.seed(client_id + 1)
    torch.manual_seed(client_id + 1)

    # 加载并准备数据
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_transform, is_train=True, shuffle=False)

    # 微调模型参数 (例如 Adapter 或整个 CLIP 模型)
    adapter = nn.Linear(clip_model.visual.output_dim, len(dataset.classnames)).to(clip_model.dtype).cuda()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    
    adapter.train()
    for epoch in tqdm(range(cfg['train_epoch'])):
        for images, targets in train_loader_cache:
            images, targets = images.cuda(), targets.cuda()
            features = clip_model.encode_image(images)
            logits = adapter(features)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 构建本地缓存模型
    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)
    
    return cache_keys, cache_values, adapter

def federated_aggregation(client_cache_keys: List[torch.Tensor], client_cache_values: List[torch.Tensor], client_adapters: List[nn.Module]) -> Tuple[torch.Tensor, torch.Tensor, nn.Module]:
    """
    聚合客户端生成的缓存和模型参数。
    """
    # 聚合 cache_keys
    aggregated_keys = torch.stack(client_cache_keys).mean(dim=0)
    
    # 聚合 cache_values
    aggregated_values = torch.stack(client_cache_values).mean(dim=0)
    
    # 聚合 Adapter 参数 (简单平均聚合)
    for adapter in client_adapters:
        print(f'adapter.weight.size: {adapter.weight.size()}')
    aggregated_adapter_weight = torch.stack([adapter.weight for adapter in client_adapters]).mean(dim=0)
    print(f'aggregated_adapter_weight.size(): {aggregated_adapter_weight.size()}')
    aggregated_adapter = nn.Linear(aggregated_keys.shape[0], aggregated_keys.shape[1], bias=False).to(aggregated_keys.dtype).cuda()
    print(f'aggregated_adapter.weight.size(): {aggregated_adapter.weight.size()}')
    aggregated_adapter.weight = nn.Parameter(aggregated_adapter_weight)
    print(f'aggregated_adapter.weight.size(): {aggregated_adapter.weight.size()}')
    
    return aggregated_keys, aggregated_values, aggregated_adapter

def main():
    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg: dict = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # 每个客户端的训练
    num_clients = cfg.get('num_clients', 2)
    client_cache_keys = []
    client_cache_values = []
    client_adapters = []

    for client_id in range(num_clients):
        print(f"\nTraining on client {client_id + 1}/{num_clients}")
        cache_keys, cache_values, adapter = client_training(client_id, cfg, clip_model)
        client_cache_keys.append(cache_keys)
        client_cache_values.append(cache_values)
        client_adapters.append(adapter)
    
    # 聚合客户端的缓存和适配器模型
    print("\nAggregating client models.")
    cache_keys, cache_values, aggregated_adapter = federated_aggregation(client_cache_keys, client_cache_values, client_adapters)

    # Prepare dataset
    random.seed(0)  # 客户端的随机种子是从1开始的
    torch.manual_seed(0)
    
    print("Preparing dataset. - global")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    # train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
    train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # # Construct the cache model by few-shot training set
    # print("\nConstructing cache model by few-shot visual features and labels.")
    # cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights)

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F, adapter=aggregated_adapter)


if __name__ == '__main__':
    main()