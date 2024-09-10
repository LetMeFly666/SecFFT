import copy
import random
import time

import torch

from FL_Backdoor_CV.roles.evaluation import test_poison_cv, test_cv
from configs import args
from FL_Backdoor_CV.image_helper import plot_image
import torch.optim as optim


class Client:
    def __init__(self, client_id, local_data, local_data_size, malicious=False):
        self.client_id = client_id
        self.local_data = local_data
        self.local_data_size = local_data_size
        self.malicious = malicious
        self.server = None
        self.adversarial_index = None
        self.range_no_id = None

    def local_train(
        self,
        local_model,
        helper,
        epoch,
        do_attack=False,
        criterion=torch.nn.CrossEntropyLoss(),
    ):
        local_data = self.local_data
        text_inputs = [f"This is a photo of a {label}" for label in helper.classes]
        if self.malicious and do_attack:
            # === clean part of malicious clients.
            # === we need to remove indices corresponding to the target label when attackers conduct local training. ===
            if self.range_no_id is None:
                range_no_id = list(range(len(helper.train_dataset)))

                if args.attack_mode.lower() in ["mr", "dba", "flip"]:
                    for ind, x in enumerate(helper.train_dataset):
                        imge, label = x
                        if label == helper.params["poison_label_swap"]:
                            range_no_id.remove(ind)  # 移除目标类别的图片

                    if args.dataset == "cifar10":  # 移除指定的图片
                        if args.attack_mode.lower() == "mr":
                            for image in (
                                helper.params["poison_images_test"]
                                + helper.params["poison_images"]
                            ):
                                if image in range_no_id:
                                    range_no_id.remove(image)

                elif args.attack_mode.lower() == "combine":
                    target_label = None
                    if self.adversarial_index == 0:
                        target_label = helper.params["poison_label_swaps"][0]
                    elif self.adversarial_index == 1:
                        target_label = helper.params["poison_label_swaps"][1]
                    elif self.adversarial_index == 2:
                        target_label = helper.params["poison_label_swaps"][2]
                    elif self.adversarial_index == 3:
                        target_label = helper.params["poison_label_swap"]
                    for ind, x in enumerate(helper.train_dataset):
                        imge, label = x
                        if label == target_label:
                            range_no_id.remove(ind)

                    if args.dataset == "cifar10":
                        if self.adversarial_index == 3:
                            for image in (
                                helper.params["poison_images_test"]
                                + helper.params["poison_images"]
                            ):
                                if image in range_no_id:
                                    range_no_id.remove(image)

                elif args.attack_mode.lower() in ["edge_case", "neurotoxin"]:
                    pass
                # 移除目标索引后的图片
                random.shuffle(range_no_id)
                self.range_no_id = range_no_id

            # === malicious training ===
            poison_optimizer = torch.optim.Adam(
                local_model.parameters(),
                lr=1e-3,
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=0.2,
            )
            if args.attack_mode.lower() in ["mr", "dba", "flip", "combine"]:
                for internal_epoch in range(1, 1 + helper.params["retrain_poison"]):

                    indices = random.sample(
                        self.range_no_id, args.batch_size - args.num_poisoned_samples
                    )

                    if args.alternating_minimization:

                        for idx, x in enumerate(helper.poisoned_train_data):
                            inputs_p, labels_p = None, None
                            if args.attack_mode.lower() == "mr":
                                inputs_p, labels_p = helper.get_poison_batch(x)
                            elif args.attack_mode.lower() in ["dba", "combine"]:
                                inputs_p, labels_p = helper.get_poison_batch(
                                    x, adversarial_index=self.adversarial_index
                                )
                            elif args.attack_mode.lower() == "flip":
                                inputs_p, labels_p = x
                                for pos in range(labels_p.size(0)):
                                    labels_p[pos] = helper.params["poison_label_swap"]
                            poison_optimizer.zero_grad()
                            clip_inputs = helper.processor(
                                text=text_inputs,
                                images=inputs_p,
                                return_tensors="pt",
                                padding=True,
                                do_rescale=False,
                            ).to(args.device)
                            labels_p = labels_p.to(args.device)
                            outputs = local_model(**clip_inputs)
                            logits_per_image = outputs.logits_per_image
                            loss = criterion(logits_per_image, labels_p)
                            loss.backward()
                            poison_optimizer.step()

                            break

                        for x in helper.get_train(indices):
                            inputs_c, labels_c = x
                            poison_optimizer.zero_grad()
                            clip_inputs = helper.processor(
                                text=text_inputs,
                                images=inputs_c,
                                return_tensors="pt",
                                padding=True,
                                do_rescale=False,
                            ).to(args.device)
                            labels_c = labels_c.to(args.device)
                            outputs = local_model(**clip_inputs)
                            logits_per_image = outputs.logits_per_image
                            loss = criterion(logits_per_image, labels_c)
                            loss.backward()
                            poison_optimizer.step()
                            break  # TODO 为什么这里加上了break

                    else:

                        for idx, (x1, x2) in enumerate(
                            zip(helper.poisoned_train_data, helper.get_train(indices))
                        ):
                            inputs_p, labels_p = None, None
                            if args.attack_mode.lower() == "mr":
                                inputs_p, labels_p = helper.get_poison_batch(x1)
                                # plot_image(inputs_p, labels_p, helper.classes, idx)
                            elif args.attack_mode.lower() in ["dba", "combine"]:
                                inputs_p, labels_p = helper.get_poison_batch(
                                    x1, adversarial_index=self.adversarial_index
                                )
                            elif args.attack_mode.lower() == "flip":
                                inputs_p, labels_p = x1
                                for pos in range(labels_p.size(0)):
                                    labels_p[pos] = helper.params["poison_label_swap"]

                            inputs_c, labels_c = x2
                            if args.attack_mode.lower() == "flip":
                                for pos in range(labels_c.size(0)):
                                    if labels_c[pos] == 7:
                                        labels_c[pos] = helper.params[
                                            "poison_label_swap"
                                        ]

                            inputs = torch.cat((inputs_p, inputs_c))
                            labels = torch.cat((labels_p, labels_c))
                            poison_optimizer.zero_grad()
                            clip_inputs = helper.processor(
                                text=text_inputs,
                                images=inputs,
                                return_tensors="pt",
                                padding=True,
                                do_rescale=False,
                            ).to(args.device)
                            labels = labels.to(args.device)
                            outputs = local_model(**clip_inputs)
                            logits_per_image = outputs.logits_per_image
                            loss = criterion(logits_per_image, labels)
                            loss.backward()
                            poison_optimizer.step()

                            break

                    # === test poison ===
                    if args.show_process:
                        if args.attack_mode.lower() == "combine":
                            poison_loss, poison_acc = test_poison_cv(
                                helper,
                                helper.poisoned_test_data,
                                local_model,
                                helper.classes,
                                helper.processor,
                                self.adversarial_index,
                            )
                        else:
                            poison_loss, poison_acc = test_poison_cv(
                                helper,
                                helper.poisoned_test_data,
                                local_model,
                                helper.classes,
                                helper.processor,
                            )
                        print(
                            f"Malicious id: {self.client_id}, "
                            f"P o i s o n - N o w ! "
                            f"Epoch: {internal_epoch}, "
                            f"Local poison accuracy: {poison_acc: .4f}, "
                            f"Local poison loss: {poison_loss: .4f}."
                        )
                    else:
                        if internal_epoch % helper.params["retrain_poison"] == 0:
                            if args.attack_mode.lower() == "combine":
                                poison_loss, poison_acc = test_poison_cv(
                                    helper,
                                    helper.poisoned_test_data,
                                    local_model,
                                    helper.classes,
                                    helper.processor,
                                    self.adversarial_index,
                                )
                            else:
                                poison_loss, poison_acc = test_poison_cv(
                                    helper,
                                    helper.poisoned_test_data,
                                    local_model,
                                    helper.classes,
                                    helper.processor,
                                )
                            print(
                                f"Malicious id: {self.client_id}, "
                                f"P o i s o n - N o w ! "
                                f"Epoch: {internal_epoch}, "
                                f"Local poison accuracy: {poison_acc: .4f}, "
                                f"Local poison loss: {poison_loss: .4f}."
                            )

            elif args.attack_mode.lower() in ["edge_case", "neurotoxin"]:
                # === get gradient mask use global model and clean data ===
                mask_grad_list = None
                if args.attack_mode.lower() == "neurotoxin":
                    assert helper.params["gradmask_ratio"] != 1
                    num_clean_data = 20
                    subset_data_chunks = random.sample(
                        helper.params["participant_clean_data"], num_clean_data
                    )
                    sampled_data = [
                        helper.benign_train_data[pos] for pos in subset_data_chunks
                    ]
                    mask_grad_list = helper.grad_mask_cv(
                        helper,
                        local_model,
                        sampled_data,
                        criterion,
                        ratio=helper.params["gradmask_ratio"],
                    )

                for internal_epoch in range(1, 1 + helper.params["retrain_poison"]):
                    # === malicious train ===
                    indices = random.sample(
                        self.range_no_id, args.batch_size - args.num_poisoned_samples
                    )
                    for x1, x2 in zip(
                        helper.poisoned_train_data, helper.get_train(indices)
                    ):
                        inputs_p, labels_p = x1
                        inputs_c, labels_c = x2
                        inputs = torch.cat((inputs_p, inputs_c))
                        for pos in range(labels_p.size(0)):
                            labels_p[pos] = helper.params["poison_label_swap"]
                        labels = torch.cat((labels_p, labels_c))

                        poison_optimizer.zero_grad()
                        clip_inputs = helper.processor(
                            text=text_inputs,
                            images=inputs,
                            return_tensors="pt",
                            padding=True,
                            do_rescale=False,
                        ).to(args.device)
                        labels = labels.to(args.device)
                        outputs = local_model(**clip_inputs)
                        logits_per_image = outputs.logits_per_image
                        loss = criterion(logits_per_image, labels)
                        loss.backward()
                        poison_optimizer.step()

                        if args.attack_mode.lower() == "neurotoxin":
                            mask_grad_list_copy = iter(mask_grad_list)
                            for name, parms in local_model.named_parameters():
                                if parms.requires_grad:
                                    parms.grad = parms.grad * next(mask_grad_list_copy)
                        poison_optimizer.step()

                    if args.show_process:
                        poison_loss, poison_acc = test_poison_cv(
                            helper,
                            helper.poisoned_test_data,
                            local_model,
                            helper.classes,
                            helper.processor,
                        )
                        print(
                            f"Malicious id: {self.client_id}, "
                            f"P o i s o n - N o w ! "
                            f"Epoch: {internal_epoch}, "
                            f"Local poison accuracy: {poison_acc: .4f}, "
                            f"Local poison loss: {poison_loss: .4f}."
                        )
                    else:
                        if internal_epoch % helper.params["retrain_poison"] == 0:
                            poison_loss, poison_acc = test_poison_cv(
                                helper,
                                helper.poisoned_test_data,
                                local_model,
                                helper.classes,
                                helper.processor,
                            )
                            print(
                                f"Malicious id: {self.client_id}, "
                                f"P o i s o n - N o w ! "
                                f"Epoch: {internal_epoch}, "
                                f"Local poison accuracy: {poison_acc: .4f}, "
                                f"Local poison loss: {poison_loss: .4f}."
                            )

            # === malicious test ===
            test_loss, test_acc = test_cv(
                helper.benign_test_data, local_model, helper.classes, helper.processor
            )
            print(
                f"Malicious id: {self.client_id}, "
                f"Test accuracy: {test_acc: .4f}, "
                f"Test loss: {test_loss: .4f}."
            )
        else:
            optimizer = torch.optim.Adam(
                local_model.parameters(),
                lr=1e-3,
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=0.2,
            )
            epochs = helper.params["retrain_no_times"]

            # === local training ===
            for _ in range(epochs):
                for inputs, labels in local_data:
                    optimizer.zero_grad()
                    clip_inputs = helper.processor(
                        text=text_inputs,
                        images=inputs,
                        return_tensors="pt",
                        padding=True,
                        do_rescale=False,
                    ).to(args.device)
                    labels = labels.to(args.device)
                    outputs = local_model(**clip_inputs)
                    logits_per_image = outputs.logits_per_image
                    loss = criterion(logits_per_image, labels)
                    loss.backward()
                    optimizer.step()

        print(f"Client {self.client_id} local training done.")

        return local_model
