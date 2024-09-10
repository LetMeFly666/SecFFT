import argparse

parser = argparse.ArgumentParser(description="PPDL")

parser.add_argument("--reduction_factor", type=int, default=5)

# === dataset, data partitioning mode, device, model, and rounds
parser.add_argument(
    "--dataset",
    default="fmnist",
    choices=["cifar10", "emnist", "fmnist"],
    help="dataset",
)

parser.add_argument("--params", default="utils/fmnist_params.yaml", dest="params")


parser.add_argument(
    "--emnist_style", default="digits", type=str, help="byclass digits letters"
)


parser.add_argument(
    "--rounds", default=20, type=int, help="total rounds for convergence"
)


parser.add_argument(
    "--participant_population", default=50, type=int, help="total clients"
)


parser.add_argument(
    "--participant_sample_size", default=50, type=int, help="participants each round"
)


parser.add_argument("--is_poison", default=1, type=int, help="poison or not")


parser.add_argument(
    "--number_of_adversaries", default=10, type=int, help="the number of attackers"
)


parser.add_argument(
    "--random_compromise", default=1, type=int, help="randomly compromise benign client"
)

parser.add_argument(
    "--poison_rounds",
    default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
    type=str,
    help="",
)

parser.add_argument(
    "--retrain_rounds",
    default=20,
    type=int,
    help="continue to train {retrain_rounds} rounds starting when resume is true",
)  # end_rounds = curr_round + retrain_rounds + window 这个参数的意思是指明中毒攻击情况下训练的轮数

parser.add_argument(
    "--aggregation_rule",
    default="avg",
    type=str,
    choices=["avg", "rlr", "flame", "foolsgold", "roseagg", "fltrust", "fedcie"],
    help="aggregation method",
)


parser.add_argument(
    "--class_imbalance",
    type=int,
    default=1,
    help="split the dataset in a non-iid (class imbalance) fashion",
)

parser.add_argument("--balance", type=float, default=1, help="balance of the data size")


parser.add_argument(
    "--classes_per_client", type=int, default=10, help="class per client"
)


parser.add_argument(
    "--attack_mode",
    default="NEUROTOXIN",
    type=str,
    help="aggregation method, [MR, DBA, EDGE_CASE, FLIP, NEUROTOXIN, COMBINE]",
)

parser.add_argument("--show_process", default=1, type=int)


parser.add_argument("--resume", default=0, type=int, help="resume or not")


parser.add_argument(
    "--resumed_name",
    default="./avg_MR_09032342/avg_400.pth",  # RoseAgg\FL_Backdoor_CV\saved_models\Revision_1\avg_MR_09032342\avg_400.pth
    type=str,
)


# === client selection mode ===


parser.add_argument(
    "--poison_prob", type=float, default=0, help="poison probability each round"
)

# === aggregation rule on the server ===


parser.add_argument("--device", default="cuda", type=str, help="device")

# === configuration of local training ===
parser.add_argument("--local_lr", type=float, default=0.1, help="learning rate")

parser.add_argument(
    "--local_lr_decay", type=float, default=0.991, help="learning rate decay"
)

parser.add_argument("--decay_step", type=int, default=5)

parser.add_argument("--local_lr_min", type=float, default=0.001, help="")

parser.add_argument("--global_lr", type=float, default=1, help="")

parser.add_argument("--global_lr_decay", type=float, default=1, help="")

parser.add_argument("--batch_size", type=int, default=128, help="local batch size")

parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")

parser.add_argument("--decay", type=float, default=5e-4, help="SGD weight_decay")

# === attack mode ===
parser.add_argument(
    "--num_poisoned_samples",
    default=64,
    type=int,
    help="the number of poisoned samples in one batch",
)

parser.add_argument(
    "--dba_trigger_num", default=4, type=int, help="the number of distributed triggers"
)

parser.add_argument(
    "--dba_poison_rounds",
    default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
    type=str,
    help="if {attack_mode} == 'DBA', the poison rounds is {dba_poison_rounds}",
)

parser.add_argument(
    "--mal_boost", default=0, type=int, help="scale up the poisoned model update"
)

parser.add_argument(
    "--gradmask_ratio",
    default=0.5,
    type=float,
    help="The proportion of the gradient retained in GradMask",
)

parser.add_argument(
    "--multi_objective_num",
    default=4,
    type=int,
    help="The number of injected backdoors. Default: wall ---> 2, pixel ---> 3",
)

parser.add_argument("--alternating_minimization", default=0, type=int, help="")

# === save model ===
parser.add_argument(
    "--record_step",
    default=1,
    type=int,
    help="save the model every {record_step} round",
)

parser.add_argument(
    "--record_res_step",
    default=1,
    type=int,
    help="save the model every {record_res_step} round",
)

# === roseagg ===
parser.add_argument(
    "--threshold",
    default=0.2,
    type=float,
    help="similarity threshold between two model updates, >{threshold} ---> cluster",
)  # 没用

parser.add_argument(
    "--gradient_correction", default=0, type=int, help="whether correct the gradient"
)  # 没用

parser.add_argument(
    "--correction_coe", default=0.1, type=float, help="weight of previous gradient"
)  # 没用

parser.add_argument(
    "--perturbation_coe", default=0.8, type=float, help="weight of random noise"
)  # 没用

parser.add_argument(
    "--windows", default=0, type=int, help="window of previous gradient"
)  # 如果没有设置poison_rounds，那么在windows轮数内，保证不会进行攻击


parser.add_argument(
    "--cie_evaluation",
    default=0,
    type=int,
    help="ablation: evaluate clean ingredient analysis",
)  # 没用

# === rlr ===
parser.add_argument("--robustLR_threshold", default=10, type=int, help="")


parser.add_argument(
    "--run_name", default=None, type=str, help="name of this experiment run (for wandb)"
)  # 没用

parser.add_argument(
    "--start_epoch",
    default=2001,
    type=int,
    help="Load pre-trained benign model that has been trained "
    "for start_epoch - 1 epoches, and resume from here",
)  # 没用


parser.add_argument(
    "--semantic_target", default=False, type=bool, help="semantic_target"  # 没用
)

parser.add_argument("--defense", default=True, type=bool, help="defense")  # 没用


parser.add_argument("--s_norm", default=1.0, type=float, help="s_norm")  # 没用


parser.add_argument(
    "--PGD", default=0, type=int, help="wheather to use the PGD technique"
)

parser.add_argument(
    "--attack_num", default=40, type=int, help="attack_num 10, 20, 30"
)  # 没用

parser.add_argument("--edge_case", default=0, type=int, help="edge_case or not")  # 没用


parser.add_argument("--aggregate_all_layer", default=1, type=int)
# 这段代码的主要目的是根据梯度的绝对值大小为每一层或所有层生成掩码，
# 用于梯度稀疏化或剪枝，以减少计算成本和通信负担。在 aggregate_all_layer == 1 的情况下，
# 所有层的梯度被聚合处理；否则，每一层梯度单独处理。代码通过计算每层的梯度绝对值或 L2 范数，
# 选择最重要的梯度部分进行保留

args = parser.parse_args()
