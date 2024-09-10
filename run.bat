@echo off

python trainer.py --dataset "fmnist" --params "utils/fmnist_params.yaml" --aggregation_rule "avg" --attack_mode "NEUROTOXIN" --number_of_adversaries 3 || echo Error ignored
python trainer.py --dataset "fmnist" --params "utils/fmnist_params.yaml" --aggregation_rule "flame" --attack_mode "NEUROTOXIN" --number_of_adversaries 3 || echo Error ignored
python trainer.py --dataset "fmnist" --params "utils/fmnist_params.yaml" --aggregation_rule "fltrust" --attack_mode "NEUROTOXIN" --number_of_adversaries 3 || echo Error ignored

python trainer.py --dataset "fmnist" --params "utils/fmnist_params.yaml" --aggregation_rule "avg" --attack_mode "MR" --number_of_adversaries 3 || echo Error ignored
python trainer.py --dataset "fmnist" --params "utils/fmnist_params.yaml" --aggregation_rule "flame" --attack_mode "MR" --number_of_adversaries 3 || echo Error ignored
python trainer.py --dataset "fmnist" --params "utils/fmnist_params.yaml" --aggregation_rule "fltrust" --attack_mode "MR" --number_of_adversaries 3 || echo Error ignored


python trainer.py --dataset "fmnist" --params "utils/fmnist_params.yaml" --aggregation_rule "avg" --attack_mode "EDGE_CASE" --number_of_adversaries 3 || echo Error ignored
python trainer.py --dataset "fmnist" --params "utils/fmnist_params.yaml" --aggregation_rule "flame" --attack_mode "EDGE_CASE" --number_of_adversaries 3 || echo Error ignored
python trainer.py --dataset "fmnist" --params "utils/fmnist_params.yaml" --aggregation_rule "fltrust" --attack_mode "EDGE_CASE" --number_of_adversaries 3 || echo Error ignored



@REM python trainer.py --dataset "cifar10" --params "utils/cifar10_params.yaml" --aggregation_rule "avg" --attack_mode "NEUROTOXIN" || echo Error ignored
python trainer.py --dataset "cifar10" --params "utils/cifar10_params.yaml" --aggregation_rule "avg" --attack_mode "MR" || echo Error ignored
python trainer.py --dataset "cifar10" --params "utils/cifar10_params.yaml" --aggregation_rule "avg" --attack_mode "DBA" || echo Error ignored
python trainer.py --dataset "cifar10" --params "utils/cifar10_params.yaml" --aggregation_rule "avg" --attack_mode "EDGE_CASE" || echo Error ignored
python trainer.py --dataset "cifar10" --params "utils/cifar10_params.yaml" --aggregation_rule "avg" --attack_mode "FLIP" || echo Error ignored

python trainer.py --dataset "cifar10" --params "utils/cifar10_params.yaml" --aggregation_rule "flame" --attack_mode "NEUROTOXIN" || echo Error ignored
python trainer.py --dataset "cifar10" --params "utils/cifar10_params.yaml" --aggregation_rule "rlr" --attack_mode "NEUROTOXIN" || echo Error ignored
python trainer.py --dataset "cifar10" --params "utils/cifar10_params.yaml" --aggregation_rule "foolsgold" --attack_mode "NEUROTOXIN" || echo Error ignored
python trainer.py --dataset "cifar10" --params "utils/cifar10_params.yaml" --aggregation_rule "fltrust" --attack_mode "NEUROTOXIN" || echo Error ignored


python trainer.py --dataset "femnist" --params "utils/femnist_params.yaml" --aggregation_rule "avg" --attack_mode "MR" || echo Error ignored
python trainer.py --dataset "cifar10" --params "utils/cifar10_params.yaml" --aggregation_rule "avg" --attack_mode "DBA" || echo Error ignored
python trainer.py --dataset "cifar10" --params "utils/cifar10_params.yaml" --aggregation_rule "avg" --attack_mode "EDGE_CASE" || echo Error ignored
python trainer.py --dataset "cifar10" --params "utils/cifar10_params.yaml" --aggregation_rule "avg" --attack_mode "FLIP" || echo Error ignored

echo All commands executed.
pause



