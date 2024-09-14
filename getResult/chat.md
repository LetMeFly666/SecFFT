解释这段代码
```
def fltrust(model_updates, param_updates, clean_param_update):
    cos = torch.nn.CosineSimilarity(dim=0)
    g0_norm = torch.norm(clean_param_update)
    weights = []
    for param_update in param_updates:
        weights.append(F.relu(cos(param_update.view(-1, 1), clean_param_update.view(-1, 1))))
    weights = torch.tensor(weights).to(args.device).view(1, -1)
    weights = weights / weights.sum()
    weights = torch.where(weights[0].isnan(), torch.zeros_like(weights), weights)
    nonzero_weights = torch.count_nonzero(weights.flatten())
    nonzero_indices = torch.nonzero(weights.flatten()).flatten()

    print(f'g0_norm: {g0_norm}, '
          f'weights_sum: {weights.sum()}, '
          f'*** {nonzero_weights} *** model updates are considered to be aggregated !')

    normalize_weights = []
    for param_update in param_updates:
        normalize_weights.append(g0_norm / torch.norm(param_update))

    global_update = dict()
    for name, params in model_updates.items():
        if 'num_batches_tracked' in name or 'running_mean' in name or 'running_var' in name:
            global_update[name] = 1 / nonzero_weights * params[nonzero_indices].sum(dim=0, keepdim=True)
        else:
            global_update[name] = torch.matmul(
                weights,
                params * torch.tensor(normalize_weights).to(args.device).view(-1, 1))
    return global_update
```






联邦学习限制大小的攻击






可不可以将隐蔽攻击的方法类型分为：限制大小、限制角度、限制符号的攻击




每种攻击类型具体都有哪些





限制符号的攻击具体有哪些？





我想要的是梯度符号和正常客户端相同从而比较隐蔽的攻击





阅读这段代码，你只需要解释这段代码有关fltrust的部分
```
def train_and_aggregate(self, global_lr, round, root_path):
        # === trained local models ===
        trained_models = dict()
        param_updates = list()
        trained_params = list()

        # path = os.path.join(
        #     os.getcwd(),
        #     f"./{root_path}/participants/{args.dataset}_{args.attack_mode}.csv",
        # )

        # print(f"round: {round}")
        # with open(path, "a") as file:
        #     file.write(f"{round}: ")
        #     file.write(" ".join(map(str, self.participants)) + "\n")
        # 获得当前全局模型的参数
        target_model_params = dict()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                target_model_params[name] = param.clone().detach().requires_grad_(False)
        target_model_params_list = [
            param for _, param in self.model.named_parameters() if param.requires_grad
        ]
        print(f"participants: {self.participants}")
        for client_id in self.participants:
            local_model = self.helper.local_model
            for name, param in local_model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(target_model_params[name])
            trained_local_model = self.clients[client_id].local_train(
                local_model,
                self.helper,
                self.current_round,
                (self.current_round in self.poison_rounds),
            )
            trained_local_model_params_list = [
                param
                for _, param in trained_local_model.named_parameters()
                if param.requires_grad
            ]
            if args.aggregation_rule.lower() == "fltrust":
                param_updates.append(
                    parameters_to_vector(trained_local_model_params_list)
                    - parameters_to_vector(target_model_params_list)
                )
            elif args.aggregation_rule.lower() == "flame":
                trained_param = (
                    parameters_to_vector(trained_local_model_params_list).detach().cpu()
                )
                trained_params.append(trained_param)
            for name, param in trained_local_model.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in trained_models:
                    trained_models[name] = param.data.view(1, -1).clone()
                else:
                    trained_models[name] = torch.cat(
                        (trained_models[name], param.data.view(1, -1).clone()), dim=0
                    )

        model_updates = dict()
        for (name, param), local_param in zip(
            target_model_params.items(), trained_models.values()
        ):
            model_updates[name] = local_param.data - param.data.view(1, -1)  # 广播机制

            if args.attack_mode.lower() in [
                "mr",
                "dba",
                "flip",
                "edge_case",
                "neurotoxin",
                "combine",
            ]:
                if "num_batches_tracked" not in name:
                    for i in range(args.participant_sample_size):
                        if self.clients[self.participants[i]].malicious:
                            mal_boost = 1
                            if args.is_poison:
                                if args.mal_boost:
                                    if args.attack_mode.lower() in [
                                        "mr",
                                        "flip",
                                        "edge_case",
                                        "neurotoxin",
                                        "combine",
                                    ]:
                                        mal_boost = (
                                            args.mal_boost / args.number_of_adversaries
                                        )
                                    elif args.attack_mode.lower() == "dba":
                                        mal_boost = args.mal_boost / (
                                            args.number_of_adversaries
                                            / args.dba_trigger_num
                                        )
                                else:
                                    if args.attack_mode.lower() in [
                                        "mr",
                                        "flip",
                                        "edge_case",
                                        "neurotoxin",
                                        "combine",
                                    ]:
                                        mal_boost = (
                                            args.participant_sample_size
                                            / args.number_of_adversaries
                                        )
                                    elif args.attack_mode.lower() == "dba":
                                        mal_boost = args.participant_sample_size / (
                                            args.number_of_adversaries
                                            / args.dba_trigger_num
                                        )
                            model_updates[name][i] *= mal_boost / args.global_lr

        # 保存模型梯度
        path = os.path.join(
            os.getcwd(),
            f"./{root_path}/model_updates/{args.dataset}_{args.attack_mode}_{round}.pkl",
        )
        with open(path, "wb") as file:
            pickle.dump(model_updates, file)

        # 判断前后两行是不是相等
        # first_layers = trained_models.values()[0]
        # is_all_rows_same = (first_layers[0] == first_layers).all(dim=1).all()
        # print(f"all rows are same: {is_all_rows_same}")

        # === aggregate ===
        global_update = None
        if args.aggregation_rule.lower() == "secfft":
            global_update = secfft(model_updates)
        elif args.aggregation_rule.lower() == "avg":
            global_update = avg(model_updates)
        # elif args.aggregation_rule.lower() == 'fedcie':
        #     global_update = Server.fedcie(model_updates, previous_model_update, last_model_params)
        elif args.aggregation_rule.lower() == "roseagg":
            global_update = roseagg(model_updates, current_round=self.current_round)
        elif args.aggregation_rule.lower() == "foolsgold":
            global_update = foolsgold(model_updates)
        elif args.aggregation_rule.lower() == "flame":
            current_model_param = (
                parameters_to_vector(target_model_params_list).detach().cpu()
            )
            global_param, global_update = flame(
                trained_params, current_model_param, model_updates
            )
            vector_to_parameters(global_param, target_model_params_list)
            return
        elif args.aggregation_rule.lower() == "fltrust":
            if self.current_round > 500:
                lr = args.local_lr * args.local_lr_decay ** (
                    (self.current_round - 500) // args.decay_step
                )
            else:
                lr = args.local_lr
            model = copy.deepcopy(self.model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(target_model_params[name])

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=1e-3,
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=0.2,
            )
            epochs = self.helper.params["retrain_no_times"]
            criterion = torch.nn.CrossEntropyLoss()
            text_inputs = [
                f"This is a photo of a {label}" for label in self.helper.classes
            ]
            for _ in range(epochs):
                for inputs, labels in self.root_dataset:
                    optimizer.zero_grad()
                    clip_inputs = self.helper.processor(
                        text=text_inputs,
                        images=inputs,
                        return_tensors="pt",
                        padding=True,
                        do_rescale=False,
                    ).to(args.device)
                    labels = labels.to(args.device)
                    outputs = model(**clip_inputs)
                    logits_per_image = outputs.logits_per_image
                    loss = criterion(logits_per_image, labels)
                    loss.backward()
                    optimizer.step()
            model_param_list = [
                param for _, param in model.named_parameters() if param.requires_grad
            ]
            clean_param_update = parameters_to_vector(
                model_param_list
            ) - parameters_to_vector(target_model_params_list)

            global_update = fltrust(model_updates, param_updates, clean_param_update)
        elif args.aggregation_rule.lower() == "rlr":
            global_update = robust_lr(model_updates)

        # === update the global model ===
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = param.data + global_lr * global_update[name].view(
                        param.size()
                    )
```





```
    print(model_updates.keys())
    print(type(model_updates.keys()))
```
```
dict_keys(['base_model.model.vision_model.encoder.layers.0.self_attn.k_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.0.self_attn.k_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.0.self_attn.v_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.0.self_attn.v_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.0.self_attn.q_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.0.self_attn.q_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.0.self_attn.out_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.0.self_attn.out_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.0.mlp.fc1.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.0.mlp.fc1.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.0.mlp.fc2.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.0.mlp.fc2.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.1.self_attn.k_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.1.self_attn.k_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.1.self_attn.v_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.1.self_attn.v_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.1.self_attn.q_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.1.self_attn.q_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.1.self_attn.out_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.1.self_attn.out_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.1.mlp.fc1.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.1.mlp.fc1.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.1.mlp.fc2.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.1.mlp.fc2.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.2.self_attn.k_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.2.self_attn.k_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.2.self_attn.v_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.2.self_attn.v_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.2.self_attn.q_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.2.self_attn.q_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.2.self_attn.out_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.2.self_attn.out_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.2.mlp.fc1.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.2.mlp.fc1.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.2.mlp.fc2.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.2.mlp.fc2.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.3.self_attn.k_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.3.self_attn.k_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.3.self_attn.v_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.3.self_attn.v_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.3.self_attn.q_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.3.self_attn.q_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.3.self_attn.out_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.3.self_attn.out_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.3.mlp.fc1.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.3.mlp.fc1.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.3.mlp.fc2.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.3.mlp.fc2.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.4.self_attn.k_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.4.self_attn.k_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.4.self_attn.v_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.4.self_attn.v_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.4.self_attn.q_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.4.self_attn.q_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.4.self_attn.out_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.4.self_attn.out_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.4.mlp.fc1.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.4.mlp.fc1.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.4.mlp.fc2.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.4.mlp.fc2.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.5.self_attn.k_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.5.self_attn.k_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.5.self_attn.v_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.5.self_attn.v_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.5.self_attn.q_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.5.self_attn.q_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.5.self_attn.out_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.5.self_attn.out_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.5.mlp.fc1.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.5.mlp.fc1.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.5.mlp.fc2.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.5.mlp.fc2.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.6.self_attn.k_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.6.self_attn.k_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.6.self_attn.v_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.6.self_attn.v_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.6.self_attn.q_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.6.self_attn.q_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.6.self_attn.out_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.6.self_attn.out_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.6.mlp.fc1.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.6.mlp.fc1.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.6.mlp.fc2.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.6.mlp.fc2.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.7.self_attn.k_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.7.self_attn.k_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.7.self_attn.v_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.7.self_attn.v_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.7.self_attn.q_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.7.self_attn.q_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.7.self_attn.out_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.7.self_attn.out_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.7.mlp.fc1.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.7.mlp.fc1.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.7.mlp.fc2.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.7.mlp.fc2.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.8.self_attn.k_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.8.self_attn.k_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.8.self_attn.v_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.8.self_attn.v_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.8.self_attn.q_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.8.self_attn.q_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.8.self_attn.out_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.8.self_attn.out_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.8.mlp.fc1.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.8.mlp.fc1.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.8.mlp.fc2.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.8.mlp.fc2.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.9.self_attn.k_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.9.self_attn.k_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.9.self_attn.v_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.9.self_attn.v_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.9.self_attn.q_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.9.self_attn.q_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.9.self_attn.out_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.9.self_attn.out_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.9.mlp.fc1.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.9.mlp.fc1.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.9.mlp.fc2.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.9.mlp.fc2.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.10.self_attn.k_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.10.self_attn.k_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.10.self_attn.v_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.10.self_attn.v_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.10.self_attn.q_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.10.self_attn.q_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.10.self_attn.out_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.10.self_attn.out_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.10.mlp.fc1.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.10.mlp.fc1.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.10.mlp.fc2.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.10.mlp.fc2.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.11.self_attn.k_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.11.self_attn.k_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.11.self_attn.v_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.11.self_attn.v_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.11.self_attn.q_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.11.self_attn.q_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.11.self_attn.out_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.11.self_attn.out_proj.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.11.mlp.fc1.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.11.mlp.fc1.lora_B.default.weight', 'base_model.model.vision_model.encoder.layers.11.mlp.fc2.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.11.mlp.fc2.lora_B.default.weight', 'base_model.model.visual_projection.lora_A.default.weight', 'base_model.model.visual_projection.lora_B.default.weight'])
<class 'dict_keys'>
```
解释dict_keys，以及如何从中取值







```
def fltrust_original(model_updates, param_updates, clean_param_update):
    cos = torch.nn.CosineSimilarity(dim=0)
    g0_norm = torch.norm(clean_param_update)
    weights = []
    for param_update in param_updates:
        weights.append(F.relu(cos(param_update.view(-1, 1), clean_param_update.view(-1, 1))))
    weights = torch.tensor(weights).to('cuda:0').view(1, -1)
    weights = weights / weights.sum()
    weights = torch.where(weights[0].isnan(), torch.zeros_like(weights), weights)
    nonzero_weights = torch.count_nonzero(weights.flatten())
    nonzero_indices = torch.nonzero(weights.flatten()).flatten()

    print(f'g0_norm: {g0_norm}, '
          f'weights_sum: {weights.sum()}, '
          f'*** {nonzero_weights} *** model updates are considered to be aggregated !')

    normalize_weights = []
    for param_update in param_updates:
        normalize_weights.append(g0_norm / torch.norm(param_update))

    global_update = dict()
    for name, params in model_updates.items():
        if 'num_batches_tracked' in name or 'running_mean' in name or 'running_var' in name:
            global_update[name] = 1 / nonzero_weights * params[nonzero_indices].sum(dim=0, keepdim=True)
        else:
            global_update[name] = torch.matmul(
                weights,
                params * torch.tensor(normalize_weights).to('cuda:0').view(-1, 1))
    return global_update
```

写一个函数`fltrust(roundsNum: List[int], dirPath: str, modelPath: str)`，例如`fltrust([15, 16, 17, 18, 19], '../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-14_15-11-15-fltrust-fmnist_NEUROTOXIN', '../NormalRun/FL_Backdoor_CV/saved_models/Revision_1/fltrust_NEUROTOXIN_09141511-fmnist')`
其中roundsNum是要选取的轮次，每个轮次有10个客户端，我选择`[15, 16, 17, 18, 19]`这5个轮次的话，相当于把这5个轮次拼接起来，视为只有一个轮次，并且这个轮次有50个客户端。每个轮次中0-2号是恶意客户端，3-9号是良性客户端。拼接的时候注意把恶意的拼接到一块，良性的拼接到一块，变成0-14号是恶意，15-49号是良性。

请你补全`fltrust`函数，你也可以修改`fltrust_original`函数，返回两个数据：
1. fltrust的恶意客户端的识别结果（返回恶意客户端的编号列表）。若fltrust并没有明确指出哪些是良性客户端，哪些是恶意客户端，则依据fltrust的原理写一个识别函数。
2. 50个客户端的评分。可以是1x50的数组，也可以是50x50的数组。

我的数据文件结构如下：
1. `dirPath`例如`../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-14_15-11-15-fltrust-fmnist_NEUROTOXIN`文件夹下有3个子文件夹，分别是`clean_param_updates`、`model_updates`、`participants`。其中`clean_param_updates`文件夹下是每个原始轮次的`clean_param_update`，命名是`fmnist_NEUROTOXIN_0.pkl`、`fmnist_NEUROTOXIN_1.pkl`、......；其中`model_updates`文件夹下是每个原始轮次的10个客户端更新，命名是`fmnist_NEUROTOXIN_0.pkl`、......；其中`participants`文件夹下有`participants.csv`这1个文件，可以直接调用`getParticipants(roundNum: int, dirPath: str) -> np.ndarray:`函数获取参与者列表，例如`participants_thisRound = getParticipants(roundNum, dirPath)`，这样的话model_updates每个键值对应的`10x参数个数`的数组中，第`i`个`参数个数`数组对应客户端编号为`participants_thisRound[i]`。

如果你需要哪些内容或者数据，请及时告诉我。
如果你有哪些不理解的地方，请及时向我询问。
在你完全理解我的意图后，在你取得足够的信息后，再开始编写代码。






GPT的回答，非我所问，很准确👇
```
感谢你提供详细的信息！为了更好地完成这个任务，让我先总结一下我对你的需求的理解，并确认我是否完全理解了要求。

理解概要
输入参数:

roundsNum: List[int]: 一个包含要选取轮次编号的列表，每个轮次有 10 个客户端。例如，[15, 16, 17, 18, 19] 表示选择轮次 15 到 19。
dirPath: str: 一个目录路径，包含三个子文件夹，分别是 clean_param_updates、model_updates、participants。
modelPath: str: 一个用于存储模型或相关数据的路径。
数据结构:

在 dirPath 下有三个子文件夹：
clean_param_updates：包含每个轮次的 clean_param_update，文件命名格式为 fmnist_NEUROTOXIN_0.pkl、fmnist_NEUROTOXIN_1.pkl 等。
model_updates：包含每个轮次 10 个客户端的模型更新，每个文件包含一个 10x参数个数 的数组。
participants：包含一个 participants.csv 文件，可使用 getParticipants(roundNum, dirPath) 函数来获取特定轮次的参与者列表。
拼接数据:

选取的轮次（例如，5 个轮次，每轮 10 个客户端）拼接成一个新的“轮次”，这个新轮次包含 50 个客户端，其中前 15 个（0-14 号）是恶意客户端，后 35 个（15-49 号）是良性客户端。
需要补全的 fltrust 函数:

该函数需要调用 fltrust_original 来进行聚合，并根据 fltrust 原理来识别恶意客户端。
函数返回两个结果：
恶意客户端的识别结果（恶意客户端的编号列表）。
50 个客户端的评分（可以是 1x50 的数组或 50x50 的数组）。
如何识别恶意客户端：

使用 fltrust 聚合算法的结果，根据聚合权重或相似度来判断哪些客户端是恶意的。
需要的确认
在我开始编写代码之前，我有几个需要确认的地方：

fltrust 中恶意客户端识别的标准：在 fltrust_original 函数中，我们使用余弦相似度和 ReLU 激活来计算客户端更新的权重，所有权重加和为 1。我的理解是，如果某些客户端的权重非常低（例如，低于某个阈值），我们可以认为它们是恶意的。请确认这一点，或者提供更多关于恶意客户端识别标准的细节。

数据读取和合并的细节：是否可以假设每个 clean_param_update 和 model_update 文件的格式是一致的，并且可以使用 pickle.load 直接加载成张量或数组格式？

聚合后的评分格式：对于 50x50 的评分矩阵，是否希望以余弦相似度矩阵的形式显示每个客户端与其他客户端的相似度？如果是，那么应基于哪些数据计算这些相似度？

请确认这些问题，以便我继续编写代码
```





你的理解非常正确！以下是针对你的疑问进行的回答：
1. fltrust中恶意客户端识别的标准：fltrust似乎并没有给出明确的聚类方式，因此你可以自定义聚类方式，推荐使用`KMeans`进行聚类。
2. 数据读取和合并的细节：每个 clean_param_update 和 model_update 文件的格式是一致的，以使用 pickle.load 直接加载成张量或字典，例如：
```
def loadPkl(roundNum: int, dirPath: str) -> Dict[str, torch.Tensor]:  # 把pkl变成{'key1': [[user3], [user5]]}
    # dirPath = '../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-12_23-26-43-avg-fmnist_NEUROTOXIN/'
    pklPrefix = os.path.basename(os.path.normpath(dirPath)).split('-')[-1]
    pklName = os.path.join(dirPath, f'model_updates/{pklPrefix}_{roundNum}.pkl')
    with open(pklName, 'rb') as f:
        update: Dict[str, torch.Tensor] = pickle.load(f)
    return update
model_updates = loadPkl(roundsNum[th], dirPath)
keys = list(model_updates.keys())
print(keys)
firstKey = keys[0]
firstValue = model_updates[firstKey]
print(type(firstValue))
print(firstValue.shape)
```
的运行结果为
```
['base_model.model.vision_model.encoder.layers.0.self_attn.k_proj.lora_A.default.weight', 'base_model.model.vision_model.encoder.layers.0.self_attn.k_proj.lora_B.default.weight', ...]
<class 'torch.Tensor'>
torch.Size([10, 12288])
```
其中`firstValue[i]`的client编号是`participants_thisRound[i]`，其他values也一致。
3. 聚合后的评分格式：可以使用余弦相似度进行计算，关于基于哪些数据，请你在fltrust的计算过程中，寻找一个合适的数据进行计算。







报错`FileNotFoundError: [Errno 2] No such file or directory: '../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-14_15-11-15-fltrust-fmnist_NEUROTOXIN\\clean_param_updates\\model_updates/clean_param_updates_15.pkl'`

其实真正的路径是`../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-14_15-11-15-fltrust-fmnist_NEUROTOXIN/clean_param_updates/clean_param_updates_15.pkl`

你多写了个`model_updates`





报错
```
malicious_clients, cosine_similarity_matrix = fltrust([15, 16, 17, 18, 19], '../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-14_15-11-15-fltrust-fmnist_NEUROTOXIN', '../NormalRun/FL_Backdoor_CV/saved_models/Revision_1/fltrust_NEUROTOXIN_09141511-fmnist')

# foolsgoldMaliciousIndex, foolsgoldScore2 = foolsgold([15, 16, 17, 18, 19], '../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-13_23-15-48-foolsgold-fmnist_NEUROTOXIN')
# print(foolsgoldMaliciousIndex)
# print(foolsgoldScore2)
# datas = [foolsgoldScore2] * 12
# plot_detection_heatmaps_3x4(*datas)
```



```
# fltrust
# https://github.com/LetMeFly666/SecFFT/blob/706bb287c3b00f6143e2190edc74714ed88f3532/getResult/FL_Backdoor_CV/roles/aggregation_rules.py#L301-L340

def fltrust_original(model_updates, param_updates, clean_param_update):
    cos = torch.nn.CosineSimilarity(dim=0)
    g0_norm = torch.norm(clean_param_update)
    weights = []
    for param_update in param_updates:
        weights.append(F.relu(cos(param_update.view(-1, 1), clean_param_update.view(-1, 1))))
    weights = torch.tensor(weights).to('cuda:0').view(1, -1)
    weights = weights / weights.sum()
    weights = torch.where(weights[0].isnan(), torch.zeros_like(weights), weights)
    nonzero_weights = torch.count_nonzero(weights.flatten())
    nonzero_indices = torch.nonzero(weights.flatten()).flatten()

    print(f'g0_norm: {g0_norm}, '
          f'weights_sum: {weights.sum()}, '
          f'*** {nonzero_weights} *** model updates are considered to be aggregated !')

    normalize_weights = []
    for param_update in param_updates:
        normalize_weights.append(g0_norm / torch.norm(param_update))

    global_update = dict()
    for name, params in model_updates.items():
        if 'num_batches_tracked' in name or 'running_mean' in name or 'running_var' in name:
            global_update[name] = 1 / nonzero_weights * params[nonzero_indices].sum(dim=0, keepdim=True)
        else:
            global_update[name] = torch.matmul(
                weights,
                params * torch.tensor(normalize_weights).to('cuda:0').view(-1, 1))
    return global_update

def fltrust(roundsNum: List[int], dirPath: str, modelPath: str):
    clientPerRound = 10  # 这里就先写死了
    maliciousPerRound = 3
    # gradients = [0] * len(roundsNum) * clientPerRound
    # maliciouses = []
    gradientsList = []  # 里面存放每一轮的梯度，最后再聚合
    for th, roundNum in enumerate(roundsNum):
        participants_thisRound = getParticipants(roundNum, dirPath)
        # get model updates
        pklName = f'../NormalRun/FL_Backdoor_CV/saved_models/Revision_1/fltrust_NEUROTOXIN_09141511-fmnist/fltrust_{roundNum}.pth'
        # print(pklName)
        with open(pklName, 'rb') as f:
            model_updates: Dict[str, torch.Tensor] = torch.load(f)
        # print(model_updates)
        pklName = f'../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-14_15-11-15-fltrust-fmnist_NEUROTOXIN/model_updates/fmnist_NEUROTOXIN_{roundNum}.pkl'
        with open(pklName, 'rb') as f:
            param_updates: Dict[str, torch.Tensor] = pickle.load(f)
        param_updates = torch.cat(list(param_updates.values()), dim=1)
        # print(param_updates[list(param_updates.keys())[0]].shape)  # torch.Size([10, 12288])
        pklName = f'../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-14_15-11-15-fltrust-fmnist_NEUROTOXIN/clean_param_updates/fmnist_NEUROTOXIN_{roundNum}.pkl'
        with open(pklName, 'rb') as f:
            clean_param_update: torch.Tensor = pickle.load(f)
        
        # global_update = fltrust_original(model_updates, [param_updates[key] for key in param_updates.keys()], clean_param_update)
        global_update = fltrust_original(model_updates, param_updates, clean_param_update)

        # print(clean_param_update.shape)  # torch.Size([2674688])
        thisGradients = [0] * clientPerRound


        gradientsList.append(thisGradients)
        break
malicious_clients, cosine_similarity_matrix = fltrust([15, 16, 17, 18, 19], '../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-14_15-11-15-fltrust-fmnist_NEUROTOXIN', '../NormalRun/FL_Backdoor_CV/saved_models/Revision_1/fltrust_NEUROTOXIN_09141511-fmnist')
```
报错
```
{
	"name": "AttributeError",
	"message": "'CLIPModel' object has no attribute 'items'",
	"stack": "---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
File c:\\ProgramData\\Anaconda3\\envs\\clip_lora\\lib\\site-packages\\peft\\peft_model.py:737, in PeftModel.__getattr__(self, name)
    736 try:
--> 737     return super().__getattr__(name)  # defer to nn.Module's logic
    738 except AttributeError:

File c:\\ProgramData\\Anaconda3\\envs\\clip_lora\\lib\\site-packages\\torch\
n\\modules\\module.py:1729, in Module.__getattr__(self, name)
   1728         return modules[name]
-> 1729 raise AttributeError(f\"'{type(self).__name__}' object has no attribute '{name}'\")

AttributeError: 'PeftModel' object has no attribute 'items'

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
File c:\\ProgramData\\Anaconda3\\envs\\clip_lora\\lib\\site-packages\\peft\\tuners\\lora\\model.py:356, in LoraModel.__getattr__(self, name)
    355 try:
--> 356     return super().__getattr__(name)  # defer to nn.Module's logic
    357 except AttributeError:

File c:\\ProgramData\\Anaconda3\\envs\\clip_lora\\lib\\site-packages\\torch\
n\\modules\\module.py:1729, in Module.__getattr__(self, name)
   1728         return modules[name]
-> 1729 raise AttributeError(f\"'{type(self).__name__}' object has no attribute '{name}'\")

AttributeError: 'LoraModel' object has no attribute 'items'

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
c:\\Users\\admin\\Desktop\\LLM\\wb2\\Codes\\getResult\\getUpdates.ipynb Cell 10 line 1
----> <a href='vscode-notebook-cell:/c%3A/Users/admin/Desktop/LLM/wb2/Codes/getResult/getUpdates.ipynb#X11sZmlsZQ%3D%3D?line=0'>1</a> malicious_clients, cosine_similarity_matrix = fltrust([15, 16, 17, 18, 19], '../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-14_15-11-15-fltrust-fmnist_NEUROTOXIN', '../NormalRun/FL_Backdoor_CV/saved_models/Revision_1/fltrust_NEUROTOXIN_09141511-fmnist')
      <a href='vscode-notebook-cell:/c%3A/Users/admin/Desktop/LLM/wb2/Codes/getResult/getUpdates.ipynb#X11sZmlsZQ%3D%3D?line=2'>3</a> # foolsgoldMaliciousIndex, foolsgoldScore2 = foolsgold([15, 16, 17, 18, 19], '../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-13_23-15-48-foolsgold-fmnist_NEUROTOXIN')
      <a href='vscode-notebook-cell:/c%3A/Users/admin/Desktop/LLM/wb2/Codes/getResult/getUpdates.ipynb#X11sZmlsZQ%3D%3D?line=3'>4</a> # print(foolsgoldMaliciousIndex)
      <a href='vscode-notebook-cell:/c%3A/Users/admin/Desktop/LLM/wb2/Codes/getResult/getUpdates.ipynb#X11sZmlsZQ%3D%3D?line=4'>5</a> # print(foolsgoldScore2)
      <a href='vscode-notebook-cell:/c%3A/Users/admin/Desktop/LLM/wb2/Codes/getResult/getUpdates.ipynb#X11sZmlsZQ%3D%3D?line=5'>6</a> # datas = [foolsgoldScore2] * 12
      <a href='vscode-notebook-cell:/c%3A/Users/admin/Desktop/LLM/wb2/Codes/getResult/getUpdates.ipynb#X11sZmlsZQ%3D%3D?line=6'>7</a> # plot_detection_heatmaps_3x4(*datas)

c:\\Users\\admin\\Desktop\\LLM\\wb2\\Codes\\getResult\\getUpdates.ipynb Cell 10 line 5
     <a href='vscode-notebook-cell:/c%3A/Users/admin/Desktop/LLM/wb2/Codes/getResult/getUpdates.ipynb#X11sZmlsZQ%3D%3D?line=54'>55</a>     clean_param_update: torch.Tensor = pickle.load(f)
     <a href='vscode-notebook-cell:/c%3A/Users/admin/Desktop/LLM/wb2/Codes/getResult/getUpdates.ipynb#X11sZmlsZQ%3D%3D?line=56'>57</a> # global_update = fltrust_original(model_updates, [param_updates[key] for key in param_updates.keys()], clean_param_update)
---> <a href='vscode-notebook-cell:/c%3A/Users/admin/Desktop/LLM/wb2/Codes/getResult/getUpdates.ipynb#X11sZmlsZQ%3D%3D?line=57'>58</a> global_update = fltrust_original(model_updates, param_updates, clean_param_update)
     <a href='vscode-notebook-cell:/c%3A/Users/admin/Desktop/LLM/wb2/Codes/getResult/getUpdates.ipynb#X11sZmlsZQ%3D%3D?line=59'>60</a> # print(clean_param_update.shape)  # torch.Size([2674688])
     <a href='vscode-notebook-cell:/c%3A/Users/admin/Desktop/LLM/wb2/Codes/getResult/getUpdates.ipynb#X11sZmlsZQ%3D%3D?line=60'>61</a> thisGradients = [0] * clientPerRound

c:\\Users\\admin\\Desktop\\LLM\\wb2\\Codes\\getResult\\getUpdates.ipynb Cell 10 line 2
     <a href='vscode-notebook-cell:/c%3A/Users/admin/Desktop/LLM/wb2/Codes/getResult/getUpdates.ipynb#X11sZmlsZQ%3D%3D?line=21'>22</a>     normalize_weights.append(g0_norm / torch.norm(param_update))
     <a href='vscode-notebook-cell:/c%3A/Users/admin/Desktop/LLM/wb2/Codes/getResult/getUpdates.ipynb#X11sZmlsZQ%3D%3D?line=23'>24</a> global_update = dict()
---> <a href='vscode-notebook-cell:/c%3A/Users/admin/Desktop/LLM/wb2/Codes/getResult/getUpdates.ipynb#X11sZmlsZQ%3D%3D?line=24'>25</a> for name, params in model_updates.items():
     <a href='vscode-notebook-cell:/c%3A/Users/admin/Desktop/LLM/wb2/Codes/getResult/getUpdates.ipynb#X11sZmlsZQ%3D%3D?line=25'>26</a>     if 'num_batches_tracked' in name or 'running_mean' in name or 'running_var' in name:
     <a href='vscode-notebook-cell:/c%3A/Users/admin/Desktop/LLM/wb2/Codes/getResult/getUpdates.ipynb#X11sZmlsZQ%3D%3D?line=26'>27</a>         global_update[name] = 1 / nonzero_weights * params[nonzero_indices].sum(dim=0, keepdim=True)

File c:\\ProgramData\\Anaconda3\\envs\\clip_lora\\lib\\site-packages\\peft\\peft_model.py:741, in PeftModel.__getattr__(self, name)
    739 if name == \"base_model\":  # see #1892: prevent infinite recursion if class is not initialized
    740     raise
--> 741 return getattr(self.base_model, name)

File c:\\ProgramData\\Anaconda3\\envs\\clip_lora\\lib\\site-packages\\peft\\tuners\\lora\\model.py:360, in LoraModel.__getattr__(self, name)
    358 if name == \"model\":  # see #1892: prevent infinite recursion if class is not initialized
    359     raise
--> 360 return getattr(self.model, name)

File c:\\ProgramData\\Anaconda3\\envs\\clip_lora\\lib\\site-packages\\torch\
n\\modules\\module.py:1729, in Module.__getattr__(self, name)
   1727     if name in modules:
   1728         return modules[name]
-> 1729 raise AttributeError(f\"'{type(self).__name__}' object has no attribute '{name}'\")

AttributeError: 'CLIPModel' object has no attribute 'items'"
}
```