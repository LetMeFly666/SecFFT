@echo off

@REM python trainer.py --dataset "cifar10" --params "utils/cifar10_params.yaml" --aggregation_rule "avg" --device "cuda:1" --attack_mode "NEUROTOXIN" || echo Error ignored
@REM python trainer.py --dataset "cifar10" --params "utils/cifar10_params.yaml" --aggregation_rule "avg" --device "cuda:1" --attack_mode "NEUROTOXIN" || echo Error ignored
@REM python trainer.py --dataset "cifar10" --params "utils/cifar10_params.yaml" --aggregation_rule "avg" --device "cuda:1" --attack_mode "NEUROTOXIN" || echo Error ignored

python trainer.py --dataset "cifar10" --params "utils/cifar10_params.yaml" --aggregation_rule "avg" --device "cuda:1" --attack_mode "MR" || echo Error ignored
@REM python trainer.py --dataset "cifar10" --params "utils/cifar10_params.yaml" --aggregation_rule "avg" --device "cuda:1" --attack_mode "MR" || echo Error ignored
@REM python trainer.py --dataset "cifar10" --params "utils/cifar10_params.yaml" --aggregation_rule "avg" --device "cuda:1" --attack_mode "MR" || echo Error ignored

python trainer.py --dataset "cifar10" --params "utils/cifar10_params.yaml" --aggregation_rule "avg" --device "cuda:1" --attack_mode "EDGE_CASE" || echo Error ignored
@REM python trainer.py --dataset "fmnist" --params "utils/fmnist_params.yaml" --aggregation_rule "flame" --device "cuda:1" --attack_mode "EDGE_CASE" || echo Error ignored
@REM python trainer.py --dataset "fmnist" --params "utils/fmnist_params.yaml" --aggregation_rule "fltrust" --attack_mode "EDGE_CASE" || echo Error ignored

@REM python trainer.py --dataset "fmnist" --params "utils/fmnist_params.yaml" --aggregation_rule "flame" --attack_mode "NEUROTOXIN" || echo Error ignored
@REM python trainer.py --dataset "fmnist" --params "utils/fmnist_params.yaml" --aggregation_rule "fltrust" --attack_mode "NEUROTOXIN" || echo Error ignored

@REM python trainer.py --dataset "fmnist" --params "utils/fmnist_params.yaml" --aggregation_rule "avg" --attack_mode "MR" || echo Error ignored
@REM python trainer.py --dataset "fmnist" --params "utils/fmnist_params.yaml" --aggregation_rule "flame" --attack_mode "MR" || echo Error ignored
@REM python trainer.py --dataset "fmnist" --params "utils/fmnist_params.yaml" --aggregation_rule "fltrust" --attack_mode "MR" || echo Error ignored


@REM python trainer.py --dataset "fmnist" --params "utils/fmnist_params.yaml" --aggregation_rule "secfft" --attack_mode "EDGE_CASE" || echo Error ignored
@REM python trainer.py --dataset "fmnist" --params "utils/fmnist_params.yaml" --aggregation_rule "secfft" --attack_mode "NEUROTOXIN" || echo Error ignored
@REM python trainer.py --dataset "fmnist" --params "utils/fmnist_params.yaml" --aggregation_rule "secfft" --attack_mode "MR" || echo Error ignored



echo All commands executed.
pause



