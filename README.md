<!--
 * @Author: LetMeFly
 * @Date: 2024-08-11 10:29:13
 * @LastEditors: LetMeFly
 * @LastEditTime: 2024-08-16 00:32:31
-->
# SecFFT: Safeguarding Federated Fine-tuning for Large Vision Language Models against Stealthy Poisoning Attacks in IoRT Networks

## 前言

LLM的FL安全性相关实验。

## Log

### Log001 - 2024.8.11_10:42-18:45

首先使用[CLIP-Adapter](https://github.com/gaopengcuhk/CLIP-Adapter)的方式进行研究，先将CLIP-Adapter跑起来。

CLIP-Adapter需要先跑起来[CoOp](https://github.com/KaiyangZhou/Dassl.pytorch)，

<details>
<summary>然后就</summary>

```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
Collecting package metadata (current_repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Collecting package metadata (repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Solving environment: \ failed

CondaError: KeyboardInterrupt

(dassl)  ✘ lzy@admin  ~/ltf/Codes/LLM/Dassl.pytorch   master  conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
Collecting package metadata (current_repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Collecting package metadata (repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Solving environment: / 
Found conflicts! Looking for incompatible packages.
This can take several minutes.  Press CTRL-C to abort.
failed                                                                                                                                                  

UnsatisfiableError: The following specifications were found to be incompatible with each other:

Output in format: Requested package -> Available versions

Package openssl conflicts for:
setuptools -> python[version='>=3.9,<3.10.0a0'] -> openssl[version='1.0.*|>=1.0.2m,<1.0.3a|>=1.0.2n,<1.0.3a|>=1.0.2o,<1.0.3a|>=1.0.2p,<1.0.3a|>=1.1.1a,<1.1.2a|>=1.1.1b,<1.1.2a|>=1.1.1c,<1.1.2a|>=1.1.1d,<1.1.2a|>=1.1.1e,<1.1.2a|>=1.1.1g,<1.1.2a|>=1.1.1h,<1.1.2a|>=1.1.1i,<1.1.2a|>=1.1.1j,<1.1.2a|>=1.1.1k,<1.1.2a|>=1.1.1l,<1.1.2a|>=1.1.1n,<1.1.2a|>=1.1.1o,<1.1.2a|>=1.1.1q,<1.1.2a|>=1.1.1s,<1.1.2a|>=1.1.1t,<1.1.2a|>=1.1.1u,<1.1.2a|>=1.1.1v,<1.1.2a|>=3.0.10,<4.0a0|>=3.0.13,<4.0a0|>=3.0.9,<4.0a0|>=3.0.8,<4.0a0|>=3.0.14,<4.0a0|>=3.0.12,<4.0a0|>=3.0.11,<4.0a0|>=1.1.1m,<1.1.2a|>=3.2.1,<4.0a0|>=1.0.2l,<1.0.3a']
openssl
python=3.8 -> openssl[version='>=1.1.1d,<1.1.2a|>=1.1.1e,<1.1.2a|>=1.1.1g,<1.1.2a|>=1.1.1j,<1.1.2a|>=1.1.1k,<1.1.2a|>=1.1.1l,<1.1.2a|>=1.1.1n,<1.1.2a|>=1.1.1q,<1.1.2a|>=1.1.1s,<1.1.2a|>=1.1.1t,<1.1.2a|>=1.1.1u,<1.1.2a|>=1.1.1v,<1.1.2a|>=3.0.10,<4.0a0|>=3.0.13,<4.0a0|>=3.2.1,<4.0a0|>=3.0.9,<4.0a0|>=3.0.8,<4.0a0']
pytorch -> python[version='>=3.8,<3.9.0a0'] -> openssl[version='1.0.*|>=1.0.2m,<1.0.3a|>=1.0.2n,<1.0.3a|>=1.0.2o,<1.0.3a|>=1.0.2p,<1.0.3a|>=1.1.1a,<1.1.2a|>=1.1.1b,<1.1.2a|>=1.1.1c,<1.1.2a|>=1.1.1d,<1.1.2a|>=1.1.1e,<1.1.2a|>=1.1.1g,<1.1.2a|>=1.1.1j,<1.1.2a|>=1.1.1k,<1.1.2a|>=1.1.1l,<1.1.2a|>=1.1.1n,<1.1.2a|>=1.1.1q,<1.1.2a|>=1.1.1s,<1.1.2a|>=1.1.1t,<1.1.2a|>=1.1.1u,<1.1.2a|>=1.1.1v,<1.1.2a|>=3.0.10,<4.0a0|>=3.0.13,<4.0a0|>=3.2.1,<4.0a0|>=3.0.9,<4.0a0|>=3.0.8,<4.0a0|>=1.1.1o,<1.1.2a|>=1.1.1i,<1.1.2a|>=1.1.1h,<1.1.2a|>=1.1.1m,<1.1.2a|>=3.0.14,<4.0a0|>=3.0.12,<4.0a0|>=3.0.11,<4.0a0|>=1.0.2l,<1.0.3a']
torchvision -> ffmpeg[version='>=4.2'] -> openssl[version='1.0.*|>=1.0.2m,<1.0.3a|>=1.0.2n,<1.0.3a|>=1.0.2o,<1.0.3a|>=1.0.2p,<1.0.3a|>=1.1.1a,<1.1.2a|>=1.1.1b,<1.1.2a|>=1.1.1c,<1.1.2a|>=1.1.1d,<1.1.2a|>=3.0.14,<4.0a0|>=3.0.13,<4.0a0|>=3.0.12,<4.0a0|>=3.0.10,<4.0a0|>=1.1.1v,<1.1.2a|>=1.1.1u,<1.1.2a|>=3.0.9,<4.0a0|>=3.0.8,<4.0a0|>=1.1.1t,<1.1.2a|>=1.1.1s,<1.1.2a|>=1.1.1q,<1.1.2a|>=1.1.1o,<1.1.2a|>=1.1.1n,<1.1.2a|>=1.1.1l,<1.1.2a|>=1.1.1k,<1.1.2a|>=1.1.1j,<1.1.2a|>=1.1.1i,<1.1.2a|>=1.1.1h,<1.1.2a|>=3.2.1,<4.0a0|>=1.1.1g,<1.1.2a|>=1.1.1e,<1.1.2a|>=3.0.11,<4.0a0|>=1.1.1m,<1.1.2a|>=1.0.2l,<1.0.3a']
wheel -> python[version='>=3.8'] -> openssl[version='1.0.*|>=1.0.2m,<1.0.3a|>=1.0.2n,<1.0.3a|>=1.0.2o,<1.0.3a|>=1.0.2p,<1.0.3a|>=1.1.1a,<1.1.2a|>=1.1.1b,<1.1.2a|>=1.1.1c,<1.1.2a|>=1.1.1d,<1.1.2a|>=1.1.1e,<1.1.2a|>=1.1.1g,<1.1.2a|>=1.1.1h,<1.1.2a|>=1.1.1i,<1.1.2a|>=1.1.1j,<1.1.2a|>=1.1.1k,<1.1.2a|>=1.1.1l,<1.1.2a|>=1.1.1m,<1.1.2a|>=1.1.1n,<1.1.2a|>=1.1.1q,<1.1.2a|>=1.1.1s,<1.1.2a|>=1.1.1t,<1.1.2a|>=1.1.1u,<1.1.2a|>=1.1.1v,<1.1.2a|>=3.0.10,<4.0a0|>=3.0.11,<4.0a0|>=3.0.12,<4.0a0|>=3.0.13,<4.0a0|>=3.0.14,<4.0a0|>=3.0.9,<4.0a0|>=3.0.8,<4.0a0|>=1.1.1o,<1.1.2a|>=3.2.1,<4.0a0|>=1.0.2l,<1.0.3a']
pip -> python[version='>=3.8'] -> openssl[version='1.0.*|>=1.0.2m,<1.0.3a|>=1.0.2n,<1.0.3a|>=1.0.2o,<1.0.3a|>=1.0.2p,<1.0.3a|>=1.1.1a,<1.1.2a|>=1.1.1b,<1.1.2a|>=1.1.1c,<1.1.2a|>=1.1.1d,<1.1.2a|>=1.1.1e,<1.1.2a|>=1.1.1g,<1.1.2a|>=1.1.1h,<1.1.2a|>=1.1.1i,<1.1.2a|>=1.1.1j,<1.1.2a|>=1.1.1k,<1.1.2a|>=1.1.1l,<1.1.2a|>=1.1.1m,<1.1.2a|>=1.1.1n,<1.1.2a|>=1.1.1q,<1.1.2a|>=1.1.1s,<1.1.2a|>=1.1.1t,<1.1.2a|>=1.1.1u,<1.1.2a|>=1.1.1v,<1.1.2a|>=3.0.10,<4.0a0|>=3.0.11,<4.0a0|>=3.0.12,<4.0a0|>=3.0.13,<4.0a0|>=3.0.14,<4.0a0|>=3.0.9,<4.0a0|>=3.0.8,<4.0a0|>=1.1.1o,<1.1.2a|>=3.2.1,<4.0a0|>=1.0.2l,<1.0.3a']

Package libstdcxx-ng conflicts for:
python=3.8 -> libstdcxx-ng[version='>=11.2.0|>=7.5.0|>=7.3.0']
pytorch -> libstdcxx-ng[version='>=11.2.0|>=9.3.0|>=7.5.0|>=7.3.0|>=5.4.0']
wheel -> python[version='>=3.8'] -> libstdcxx-ng[version='>=11.2.0|>=7.5.0|>=7.3.0|>=7.2.0']
torchvision -> libstdcxx-ng[version='>=11.2.0|>=7.3.0|>=5.4.0']
setuptools -> python[version='>=3.9,<3.10.0a0'] -> libstdcxx-ng[version='>=11.2.0|>=7.5.0|>=7.3.0|>=7.2.0']
pip -> python[version='>=3.8'] -> libstdcxx-ng[version='>=11.2.0|>=7.5.0|>=7.3.0|>=7.2.0']
readline -> ncurses[version='>=6.2,<7.0a0'] -> libstdcxx-ng[version='>=7.2.0|>=7.3.0']
cudatoolkit=10.2 -> libstdcxx-ng[version='>=7.3.0']
pytorch -> python[version='>=3.7,<3.8.0a0'] -> libstdcxx-ng[version='>=7.2.0']
libffi -> libstdcxx-ng[version='>=11.2.0|>=7.5.0|>=7.3.0|>=7.2.0']
ncurses -> libstdcxx-ng[version='>=7.2.0|>=7.3.0']
python=3.8 -> libffi[version='>=3.2.1,<3.3a0'] -> libstdcxx-ng[version='>=7.2.0']
torchvision -> pillow[version='>=5.3.0,!=8.3.*'] -> libstdcxx-ng[version='>=7.2.0|>=7.5.0|>=8.4.0|>=9.3.0']

Package readline conflicts for:
python=3.8 -> readline[version='>=7.0,<8.0a0|>=8.0,<9.0a0|>=8.2,<9.0a0']
torchvision -> python[version='>=3.11,<3.12.0a0'] -> readline[version='7.*|>=7.0,<8.0a0|>=8.0,<9.0a0|>=8.2,<9.0a0|>=8.1.2,<9.0a0']
pip -> python[version='>=3.8'] -> readline[version='7.*|>=7.0,<8.0a0|>=8.0,<9.0a0|>=8.1.2,<9.0a0|>=8.2,<9.0a0']
wheel -> python[version='>=3.8'] -> readline[version='7.*|>=7.0,<8.0a0|>=8.0,<9.0a0|>=8.1.2,<9.0a0|>=8.2,<9.0a0']
pytorch -> python[version='>=3.8,<3.9.0a0'] -> readline[version='7.*|>=7.0,<8.0a0|>=8.0,<9.0a0|>=8.2,<9.0a0|>=8.1.2,<9.0a0']
readline
setuptools -> python[version='>=3.9,<3.10.0a0'] -> readline[version='7.*|>=7.0,<8.0a0|>=8.0,<9.0a0|>=8.1.2,<9.0a0|>=8.2,<9.0a0']

Package libsqlite conflicts for:
setuptools -> python[version='>=3.8,<3.9.0a0'] -> libsqlite[version='>=3.45.2,<4.0a0']
torchvision -> python[version='>=3.8,<3.9.0a0'] -> libsqlite[version='>=3.45.2,<4.0a0']
libsqlite
wheel -> python[version='>=3.8'] -> libsqlite[version='>=3.45.2,<4.0a0']
python=3.8 -> libsqlite[version='>=3.45.2,<4.0a0']
pip -> python[version='>=3.8'] -> libsqlite[version='>=3.45.2,<4.0a0']
pytorch -> python[version='>=3.8,<3.9.0a0'] -> libsqlite[version='>=3.45.2,<4.0a0']

Package _libgcc_mutex conflicts for:
libnsl -> libgcc-ng[version='>=12'] -> _libgcc_mutex[version='*|0.1',build='main|conda_forge|main']
openssl -> libgcc-ng[version='>=12'] -> _libgcc_mutex[version='*|0.1',build='main|conda_forge|main']
libgcc-ng -> _libgcc_mutex[version='*|0.1',build='main|conda_forge|main']
libffi -> libgcc-ng[version='>=11.2.0'] -> _libgcc_mutex[version='*|0.1',build='main|conda_forge|main']
bzip2 -> libgcc-ng[version='>=12'] -> _libgcc_mutex[version='*|0.1',build='main|conda_forge|main']
xz -> libgcc-ng[version='>=11.2.0'] -> _libgcc_mutex[version='*|0.1',build='main|conda_forge|main']
_openmp_mutex -> _libgcc_mutex==0.1[build='main|conda_forge']
readline -> libgcc-ng[version='>=12'] -> _libgcc_mutex[version='*|0.1',build='main|conda_forge|main']
libxcrypt -> libgcc-ng[version='>=12'] -> _libgcc_mutex==0.1=conda_forge
_libgcc_mutex
libzlib -> libgcc-ng[version='>=12'] -> _libgcc_mutex==0.1=conda_forge
torchvision -> libgcc-ng[version='>=11.2.0'] -> _libgcc_mutex[version='*|0.1',build='main|conda_forge|main']
cudatoolkit=10.2 -> libgcc-ng[version='>=7.3.0'] -> _libgcc_mutex[version='*|0.1',build='main|conda_forge|main']
libsqlite -> libgcc-ng[version='>=12'] -> _libgcc_mutex==0.1=conda_forge
libgomp -> _libgcc_mutex==0.1[build='main|conda_forge']
libuuid -> libgcc-ng[version='>=12'] -> _libgcc_mutex[version='*|0.1',build='main|conda_forge|main']
tk -> libgcc-ng[version='>=11.2.0'] -> _libgcc_mutex[version='*|0.1',build='main|conda_forge|main']
ncurses -> libgcc-ng[version='>=12'] -> _libgcc_mutex[version='*|0.1',build='main|conda_forge|main']
python=3.8 -> libgcc-ng[version='>=11.2.0'] -> _libgcc_mutex[version='*|0.1',build='main|conda_forge|main']
pytorch -> _openmp_mutex -> _libgcc_mutex[version='*|0.1',build='main|conda_forge|main']

Package libgcc-ng conflicts for:
libzlib -> libgcc-ng[version='>=12']
tk -> libgcc-ng[version='>=11.2.0|>=12|>=7.5.0|>=7.3.0|>=7.2.0']
libsqlite -> libgcc-ng[version='>=12']
wheel -> python[version='>=3.8'] -> libgcc-ng[version='>=11.2.0|>=7.5.0|>=7.3.0|>=12|>=7.2.0']
pytorch -> libgcc-ng[version='>=11.2.0|>=9.3.0|>=7.5.0|>=7.3.0|>=5.4.0']
openssl -> libgcc-ng[version='>=11.2.0|>=12|>=7.5.0|>=7.3.0|>=7.2.0']
setuptools -> python[version='>=3.9,<3.10.0a0'] -> libgcc-ng[version='>=11.2.0|>=7.5.0|>=7.3.0|>=12|>=7.2.0']
xz -> libgcc-ng[version='>=11.2.0|>=12|>=7.5.0|>=7.3.0|>=7.2.0']
torchvision -> libpng -> libgcc-ng[version='>=12|>=7.2.0|>=7.5.0|>=8.4.0|>=9.3.0']
torchvision -> libgcc-ng[version='>=11.2.0|>=7.3.0|>=5.4.0']
python=3.8 -> libgcc-ng[version='>=11.2.0|>=12|>=7.5.0|>=7.3.0']
libgcc-ng
python=3.8 -> libffi[version='>=3.4,<3.5'] -> libgcc-ng[version='>=7.2.0|>=9.4.0']
libnsl -> libgcc-ng[version='>=11.2.0|>=12']
bzip2 -> libgcc-ng[version='>=11.2.0|>=12|>=7.3.0|>=7.2.0']
libuuid -> libgcc-ng[version='>=11.2.0|>=12|>=7.5.0|>=7.2.0']
pytorch -> python[version='>=3.8,<3.9.0a0'] -> libgcc-ng[version='>=12|>=7.2.0|>=8.2.0']
libxcrypt -> libgcc-ng[version='>=12']
pip -> python[version='>=3.8'] -> libgcc-ng[version='>=11.2.0|>=7.5.0|>=7.3.0|>=12|>=7.2.0']
libffi -> libgcc-ng[version='>=11.2.0|>=9.4.0|>=7.5.0|>=7.3.0|>=7.2.0']
readline -> libgcc-ng[version='>=11.2.0|>=12|>=7.5.0|>=7.3.0|>=7.2.0']
ncurses -> libgcc-ng[version='>=11.2.0|>=12|>=7.5.0|>=7.3.0|>=7.2.0']
cudatoolkit=10.2 -> libgcc-ng[version='>=7.3.0']

Package xz conflicts for:
python=3.8 -> xz[version='>=5.2.10,<6.0a0|>=5.2.6,<6.0a0|>=5.4.6,<6.0a0|>=5.4.2,<6.0a0|>=5.2.5,<6.0a0|>=5.2.4,<6.0a0']
wheel -> python[version='>=3.8'] -> xz[version='>=5.2.10,<6.0a0|>=5.4.2,<6.0a0|>=5.4.5,<6.0a0|>=5.4.6,<6.0a0|>=5.2.8,<6.0a0|>=5.2.6,<6.0a0|>=5.2.5,<6.0a0|>=5.2.4,<6.0a0|>=5.2.3,<6.0a0']
pip -> python[version='>=3.8'] -> xz[version='>=5.2.10,<6.0a0|>=5.4.2,<6.0a0|>=5.4.5,<6.0a0|>=5.4.6,<6.0a0|>=5.2.8,<6.0a0|>=5.2.6,<6.0a0|>=5.2.5,<6.0a0|>=5.2.4,<6.0a0|>=5.2.3,<6.0a0']
torchvision -> ffmpeg[version='>=4.2'] -> xz[version='>=5.2.10,<6.0a0|>=5.4.2,<6.0a0|>=5.4.5,<6.0a0|>=5.4.6,<6.0a0|>=5.2.8,<6.0a0|>=5.2.6,<6.0a0|>=5.2.5,<6.0a0|>=5.2.4,<6.0a0|>=5.2.3,<6.0a0']
xz
setuptools -> python[version='>=3.9,<3.10.0a0'] -> xz[version='>=5.2.10,<6.0a0|>=5.4.2,<6.0a0|>=5.4.6,<6.0a0|>=5.2.8,<6.0a0|>=5.2.6,<6.0a0|>=5.2.5,<6.0a0|>=5.4.5,<6.0a0|>=5.2.4,<6.0a0|>=5.2.3,<6.0a0']
pytorch -> python[version='>=3.8,<3.9.0a0'] -> xz[version='>=5.2.10,<6.0a0|>=5.2.6,<6.0a0|>=5.4.6,<6.0a0|>=5.4.2,<6.0a0|>=5.2.5,<6.0a0|>=5.2.4,<6.0a0|>=5.2.8,<6.0a0|>=5.4.5,<6.0a0|>=5.2.3,<6.0a0']

Package libnsl conflicts for:
libnsl
torchvision -> python[version='>=3.8,<3.9.0a0'] -> libnsl[version='>=2.0.1,<2.1.0a0']
setuptools -> python[version='>=3.8,<3.9.0a0'] -> libnsl[version='>=2.0.1,<2.1.0a0']
wheel -> python[version='>=3.8'] -> libnsl[version='>=2.0.1,<2.1.0a0']
pip -> python[version='>=3.8'] -> libnsl[version='>=2.0.1,<2.1.0a0']
python=3.8 -> libnsl[version='>=2.0.1,<2.1.0a0']
pytorch -> python[version='>=3.8,<3.9.0a0'] -> libnsl[version='>=2.0.1,<2.1.0a0']

Package _openmp_mutex conflicts for:
libffi -> libgcc-ng[version='>=11.2.0'] -> _openmp_mutex[version='>=4.5']
openssl -> libgcc-ng[version='>=12'] -> _openmp_mutex[version='>=4.5']
pytorch -> libgcc-ng[version='>=11.2.0'] -> _openmp_mutex[version='>=4.5']
xz -> libgcc-ng[version='>=11.2.0'] -> _openmp_mutex[version='>=4.5']
bzip2 -> libgcc-ng[version='>=12'] -> _openmp_mutex[version='>=4.5']
readline -> libgcc-ng[version='>=12'] -> _openmp_mutex[version='>=4.5']
libxcrypt -> libgcc-ng[version='>=12'] -> _openmp_mutex[version='>=4.5']
_openmp_mutex
tk -> libgcc-ng[version='>=11.2.0'] -> _openmp_mutex[version='>=4.5']
libsqlite -> libgcc-ng[version='>=12'] -> _openmp_mutex[version='>=4.5']
libzlib -> libgcc-ng[version='>=12'] -> _openmp_mutex[version='>=4.5']
libuuid -> libgcc-ng[version='>=12'] -> _openmp_mutex[version='>=4.5']
python=3.8 -> libgcc-ng[version='>=11.2.0'] -> _openmp_mutex[version='>=4.5']
cudatoolkit=10.2 -> libgcc-ng[version='>=7.3.0'] -> _openmp_mutex[version='>=4.5']
torchvision -> pytorch==2.3.0 -> _openmp_mutex[version='>=4.5']
libnsl -> libgcc-ng[version='>=12'] -> _openmp_mutex[version='>=4.5']
pytorch -> _openmp_mutex
ncurses -> libgcc-ng[version='>=12'] -> _openmp_mutex[version='>=4.5']
libgcc-ng -> _openmp_mutex[version='>=4.5']

Package tk conflicts for:
setuptools -> python[version='>=3.9,<3.10.0a0'] -> tk[version='8.6.*|>=8.6.10,<8.7.0a0|>=8.6.11,<8.7.0a0|>=8.6.12,<8.7.0a0|>=8.6.14,<8.7.0a0|>=8.6.13,<8.7.0a0|>=8.6.8,<8.7.0a0|>=8.6.7,<8.7.0a0']
torchvision -> pillow[version='>=5.3.0,!=8.3.*'] -> tk[version='8.6.*|>=8.6.10,<8.7.0a0|>=8.6.12,<8.7.0a0|>=8.6.8,<8.7.0a0|>=8.6.14,<8.7.0a0|>=8.6.11,<8.7.0a0|>=8.6.13,<8.7.0a0|>=8.6.7,<8.7.0a0']
pytorch -> python[version='>=3.8,<3.9.0a0'] -> tk[version='8.6.*|>=8.6.10,<8.7.0a0|>=8.6.11,<8.7.0a0|>=8.6.12,<8.7.0a0|>=8.6.13,<8.7.0a0|>=8.6.8,<8.7.0a0|>=8.6.14,<8.7.0a0|>=8.6.7,<8.7.0a0']
python=3.8 -> tk[version='>=8.6.10,<8.7.0a0|>=8.6.11,<8.7.0a0|>=8.6.12,<8.7.0a0|>=8.6.13,<8.7.0a0|>=8.6.8,<8.7.0a0']
tk
wheel -> python[version='>=3.8'] -> tk[version='8.6.*|>=8.6.10,<8.7.0a0|>=8.6.11,<8.7.0a0|>=8.6.12,<8.7.0a0|>=8.6.14,<8.7.0a0|>=8.6.13,<8.7.0a0|>=8.6.8,<8.7.0a0|>=8.6.7,<8.7.0a0']
pip -> python[version='>=3.8'] -> tk[version='8.6.*|>=8.6.10,<8.7.0a0|>=8.6.11,<8.7.0a0|>=8.6.12,<8.7.0a0|>=8.6.14,<8.7.0a0|>=8.6.13,<8.7.0a0|>=8.6.8,<8.7.0a0|>=8.6.7,<8.7.0a0']

Package libuuid conflicts for:
setuptools -> python[version='>=3.12,<3.13.0a0'] -> libuuid[version='>=1.0.3,<2.0a0|>=1.41.5,<2.0a0|>=2.38.1,<3.0a0']
pip -> python[version='>=3.8'] -> libuuid[version='>=1.0.3,<2.0a0|>=1.41.5,<2.0a0|>=2.38.1,<3.0a0']
pytorch -> python[version='>=3.8,<3.9.0a0'] -> libuuid[version='>=1.0.3,<2.0a0|>=1.41.5,<2.0a0|>=2.38.1,<3.0a0']
torchvision -> python[version='>=3.11,<3.12.0a0'] -> libuuid[version='>=1.0.3,<2.0a0|>=1.41.5,<2.0a0|>=2.38.1,<3.0a0']
libuuid
python=3.8 -> libuuid[version='>=2.38.1,<3.0a0']
wheel -> python[version='>=3.8'] -> libuuid[version='>=1.0.3,<2.0a0|>=1.41.5,<2.0a0|>=2.38.1,<3.0a0']

Package libzlib conflicts for:
pip -> python[version='>=3.8'] -> libzlib[version='>=1.2.13,<2.0.0a0']
tk -> libzlib[version='>=1.2.13,<2.0.0a0']
python=3.8 -> libzlib[version='>=1.2.13,<2.0.0a0']
libzlib
pytorch -> python[version='>=3.8,<3.9.0a0'] -> libzlib[version='>=1.2.13,<2.0.0a0']
python=3.8 -> libsqlite[version='>=3.45.2,<4.0a0'] -> libzlib[version='>=1.2.13,<2.0a0']
libsqlite -> libzlib[version='>=1.2.13,<2.0a0']
setuptools -> python[version='>=3.8,<3.9.0a0'] -> libzlib[version='>=1.2.13,<2.0.0a0']
wheel -> python[version='>=3.8'] -> libzlib[version='>=1.2.13,<2.0.0a0']
torchvision -> python[version='>=3.8,<3.9.0a0'] -> libzlib[version='>=1.2.13,<2.0.0a0']

Package ncurses conflicts for:
setuptools -> python[version='>=3.9,<3.10.0a0'] -> ncurses[version='6.0.*|>=6.0,<7.0a0|>=6.1,<7.0a0|>=6.2,<7.0a0|>=6.3,<7.0a0|>=6.4,<7.0a0|>=6.4.20240210,<7.0a0']
pip -> python[version='>=3.8'] -> ncurses[version='6.0.*|>=6.0,<7.0a0|>=6.1,<7.0a0|>=6.2,<7.0a0|>=6.3,<7.0a0|>=6.4,<7.0a0|>=6.4.20240210,<7.0a0']
wheel -> python[version='>=3.8'] -> ncurses[version='6.0.*|>=6.0,<7.0a0|>=6.1,<7.0a0|>=6.2,<7.0a0|>=6.3,<7.0a0|>=6.4,<7.0a0|>=6.4.20240210,<7.0a0']
readline -> ncurses[version='6.0.*|>=6.0,<7.0a0|>=6.1,<7.0a0|>=6.2,<7.0a0|>=6.3,<7.0a0']
python=3.8 -> readline[version='>=7.0,<8.0a0'] -> ncurses[version='6.0.*|>=6.0,<7.0a0']
python=3.8 -> ncurses[version='>=6.1,<7.0a0|>=6.2,<7.0a0|>=6.3,<7.0a0|>=6.4,<7.0a0|>=6.4.20240210,<7.0a0']
pytorch -> python[version='>=3.8,<3.9.0a0'] -> ncurses[version='6.0.*|>=6.0,<7.0a0|>=6.1,<7.0a0|>=6.2,<7.0a0|>=6.3,<7.0a0|>=6.4,<7.0a0|>=6.4.20240210,<7.0a0']
ncurses
torchvision -> python[version='>=3.11,<3.12.0a0'] -> ncurses[version='6.0.*|>=6.0,<7.0a0|>=6.1,<7.0a0|>=6.2,<7.0a0|>=6.3,<7.0a0|>=6.4,<7.0a0|>=6.4.20240210,<7.0a0']

Package ld_impl_linux-64 conflicts for:
wheel -> python[version='>=3.8'] -> ld_impl_linux-64[version='>=2.35.1|>=2.36.1']
ld_impl_linux-64
setuptools -> python[version='>=3.9,<3.10.0a0'] -> ld_impl_linux-64[version='>=2.35.1|>=2.36.1']
pip -> python[version='>=3.8'] -> ld_impl_linux-64[version='>=2.35.1|>=2.36.1']
python=3.8 -> ld_impl_linux-64[version='>=2.36.1']
torchvision -> python[version='>=3.11,<3.12.0a0'] -> ld_impl_linux-64[version='>=2.35.1|>=2.36.1']
pytorch -> python[version='>=3.8,<3.9.0a0'] -> ld_impl_linux-64[version='>=2.35.1|>=2.36.1']

Package bzip2 conflicts for:
bzip2
setuptools -> python[version='>=3.12,<3.13.0a0'] -> bzip2[version='>=1.0.8,<2.0a0']
python=3.8 -> bzip2[version='>=1.0.8,<2.0a0']
pytorch -> python[version='>=3.8,<3.9.0a0'] -> bzip2[version='>=1.0.8,<2.0a0']
torchvision -> ffmpeg[version='>=4.2'] -> bzip2[version='>=1.0.8,<2.0a0']
wheel -> python[version='>=3.8'] -> bzip2[version='>=1.0.8,<2.0a0']
pip -> python[version='>=3.8'] -> bzip2[version='>=1.0.8,<2.0a0']

Package certifi conflicts for:
wheel -> setuptools -> certifi[version='>=2016.09|>=2016.9.26']
pip -> setuptools -> certifi[version='>=2016.09|>=2016.9.26']
setuptools -> certifi[version='>=2016.09|>=2016.9.26']
torchvision -> requests -> certifi[version='>=2016.09|>=2016.9.26|>=2017.4.17']

Package ca-certificates conflicts for:
pytorch -> python[version='>=2.7,<2.8.0a0'] -> ca-certificates
openssl -> ca-certificates
torchvision -> python[version='>=2.7,<2.8.0a0'] -> ca-certificates
pip -> python[version='>=2.7,<2.8.0a0'] -> ca-certificates
ca-certificates
wheel -> python -> ca-certificates
python=3.8 -> openssl[version='>=3.0.13,<4.0a0'] -> ca-certificates
setuptools -> python[version='>=2.7,<2.8.0a0'] -> ca-certificates

Package libffi conflicts for:
torchvision -> python[version='>=3.11,<3.12.0a0'] -> libffi[version='3.2.*|>=3.2.1,<3.3a0|>=3.3,<3.4.0a0|>=3.4,<3.5|>=3.4,<4.0a0']
pytorch -> python[version='>=3.8,<3.9.0a0'] -> libffi[version='3.2.*|>=3.2.1,<3.3a0|>=3.3,<3.4.0a0|>=3.4,<3.5|>=3.4,<4.0a0|>=3.3']
setuptools -> python[version='>=3.9,<3.10.0a0'] -> libffi[version='3.2.*|>=3.2.1,<3.3a0|>=3.3,<3.4.0a0|>=3.4,<3.5|>=3.4,<4.0a0']
python=3.8 -> libffi[version='>=3.2.1,<3.3a0|>=3.3,<3.4.0a0|>=3.4,<3.5|>=3.4,<4.0a0']
libffi
wheel -> python[version='>=3.8'] -> libffi[version='3.2.*|>=3.2.1,<3.3a0|>=3.3,<3.4.0a0|>=3.4,<3.5|>=3.4,<4.0a0']
pip -> python[version='>=3.8'] -> libffi[version='3.2.*|>=3.2.1,<3.3a0|>=3.3,<3.4.0a0|>=3.4,<3.5|>=3.4,<4.0a0']

Package libxcrypt conflicts for:
pip -> python[version='>=3.8'] -> libxcrypt[version='>=4.4.36']
python=3.8 -> libxcrypt[version='>=4.4.36']
pytorch -> python[version='>=3.8,<3.9.0a0'] -> libxcrypt[version='>=4.4.36']
torchvision -> python[version='>=3.8,<3.9.0a0'] -> libxcrypt[version='>=4.4.36']
wheel -> python[version='>=3.8'] -> libxcrypt[version='>=4.4.36']
setuptools -> python[version='>=3.8,<3.9.0a0'] -> libxcrypt[version='>=4.4.36']
libxcrypt

Package expat conflicts for:
torchvision -> python[version='>=3.12,<3.13.0a0'] -> expat[version='>=2.5.0,<3.0a0|>=2.6.2,<3.0a0']
pytorch -> python[version='>=3.12,<3.13.0a0'] -> expat[version='>=2.5.0,<3.0a0|>=2.6.2,<3.0a0']
wheel -> python[version='>=3.8'] -> expat[version='>=2.5.0,<3.0a0|>=2.6.2,<3.0a0']
pip -> python[version='>=3.8'] -> expat[version='>=2.5.0,<3.0a0|>=2.6.2,<3.0a0']
setuptools -> python[version='>=3.12,<3.13.0a0'] -> expat[version='>=2.5.0,<3.0a0|>=2.6.2,<3.0a0']

Package wheel conflicts for:
pip -> wheel
wheel
python=3.8 -> pip -> wheel

Package cudatoolkit conflicts for:
torchvision -> pytorch==1.13.1 -> cudatoolkit[version='10.0.*|>=10.1.243,<10.2.0a0|>=11.3.1,<11.4.0a0|9.2.*|>=8.0,<8.1.0a0|9.0.*|8.0.*|7.5.*']
pytorch -> cudatoolkit[version='10.0.*|>=10.0.130,<10.1.0a0|>=10.1,<10.2|>=10.2,<10.3|>=11.3,<11.4|>=11.6,<11.7|>=11.5,<11.6|>=11.1,<11.2|>=11.0,<11.1|>=9.2,<9.3|>=11.8.0,<11.9.0a0|>=11.3.1,<11.4.0a0|>=10.1.243,<10.2.0a0|>=9.2,<9.3.0a0|9.2.*|>=9.0,<9.1.0a0|>=8.0,<8.1.0a0|9.0.*|8.0.*|7.5.*']
pytorch -> cudnn[version='>=8.9,<9.0a0'] -> cudatoolkit[version='11.*|>=11.8.0|8.*|>=10.2.89,<10.3.0a0|>=9.0,<9.1|>=10.0,<10.1']
cudatoolkit=10.2
torchvision -> cudatoolkit[version='11.8.*|>=10.1,<10.2|>=10.2,<10.3|>=11.3,<11.4|>=11.6,<11.7|>=11.5,<11.6|>=11.1,<11.2|>=11.0,<11.1|>=9.2,<9.3|>=11.8.0,<11.9.0a0|>=10.0.130,<10.1.0a0|>=9.2,<9.3.0a0|>=9.0,<9.1.0a0']

Package pytorch conflicts for:
pytorch
torchvision -> pytorch[version='1.1.*|1.10.0|1.10.1|1.10.2|1.11.0|1.12.0|1.12.1|1.13.0|1.13.1|2.0.0|2.0.1|2.1.0|2.1.1|2.1.2|2.2.0|2.2.1|2.2.2|2.3.0|2.3.1|2.4.0|1.9.1|1.9.0|1.8.1|1.8.0|1.7.1|1.7.0|1.6.0|1.5.1|2.3.*|2.3.*|1.7.1.*|1.3.1.*|1.2.0.*|>=0.4|>=0.3',build='*cpu*|*cuda118*']

Package libgomp conflicts for:
_openmp_mutex -> libgomp[version='>=7.5.0']
libgcc-ng -> _openmp_mutex[version='>=4.5'] -> libgomp[version='>=7.5.0']
libgomp
pytorch -> _openmp_mutex -> libgomp[version='>=7.5.0']The following specifications were found to be incompatible with your system:

  - feature:/linux-64::__cuda==11.4=0
  - feature:/linux-64::__glibc==2.31=0
  - feature:|@/linux-64::__cuda==11.4=0
  - feature:|@/linux-64::__glibc==2.31=0
  - bzip2 -> __glibc[version='>=2.17,<3.0.a0']
  - bzip2 -> libgcc-ng[version='>=11.2.0'] -> __glibc[version='>=2.17']
  - cudatoolkit=10.2 -> libgcc-ng[version='>=7.3.0'] -> __glibc[version='>=2.17']
  - libffi -> libgcc-ng[version='>=11.2.0'] -> __glibc[version='>=2.17']
  - libgcc-ng -> __glibc[version='>=2.17']
  - libnsl -> libgcc-ng[version='>=11.2.0'] -> __glibc[version='>=2.17']
  - libuuid -> libgcc-ng[version='>=11.2.0'] -> __glibc[version='>=2.17']
  - ncurses -> libgcc-ng[version='>=11.2.0'] -> __glibc[version='>=2.17']
  - openssl -> __glibc[version='>=2.17,<3.0.a0']
  - openssl -> libgcc-ng[version='>=11.2.0'] -> __glibc[version='>=2.17']
  - python=3.8 -> libgcc-ng[version='>=11.2.0'] -> __glibc[version='>=2.17|>=2.17,<3.0.a0']
  - pytorch -> __cuda[version='>=11.8']
  - pytorch -> libgcc-ng[version='>=11.2.0'] -> __glibc[version='>=2.17']
  - readline -> libgcc-ng[version='>=11.2.0'] -> __glibc[version='>=2.17']
  - tk -> libgcc-ng[version='>=11.2.0'] -> __glibc[version='>=2.17']
  - torchvision -> __cuda[version='>=11.8']
  - torchvision -> __glibc[version='>=2.17,<3.0.a0']
  - torchvision -> libgcc-ng[version='>=11.2.0'] -> __glibc[version='>=2.17']
  - xz -> libgcc-ng[version='>=11.2.0'] -> __glibc[version='>=2.17']

Your installed version is: 2.31

Note that strict channel priority may have removed packages required for satisfiability.
```

</details>

然后准备跑`New version of CLIP-Adapter`：[Tip-Adapter: Training-free CLIP-Adapter](https://github.com/gaopengcuhk/Tip-Adapter)

当前将其添加到了分支[z.001.tip-adapter](https://github.com/LetMeFly666/SecFFT/tree/z.001.tip-adapter)下。

数据集：默认的`ImageNet`有一百多G，决定使用同样受支持的`Flowers102`（oxford_flowers）

### Log002 - 2024.8.12_13:28-18:56

读一下代码结构，融入联邦学习中。

明天大概要开会，先融入了再说。。

结果：先读了一下源码，明确了一下数据类型（[6f08b1..](https://github.com/LetMeFly666/SecFFT/commit/6f08b1cc63cffba4cb91aec910a0c04adf5d965d)）

### Log002 - 2024.8.12_18:56-12:51

加上联邦学习框架。

运行结果

```
round 1's acc: 94.11
round 2's acc: 94.92
round 3's acc: 94.88
round 4's acc: 95.90
round 5's acc: 95.25
round 6's acc: 95.45
round 7's acc: 95.66
round 8's acc: 95.66
round 9's acc: 95.49
round 10's acc: 95.25
```

融入成功：[de604d..](https://github.com/LetMeFly666/SecFFT/commit/de604d63f39300ff0131f8cf1f546a2c0c3472ce)

### Log003 - 2024.8.13_17:02-2024.8.15_21:32

刚开完60min会，周老师准备之后2天开一次会。44GZ44Gn44G/5LyR44GK44Gv6L+R5pyACg==。

把related work写完

场景 - IoT，机器人 具身智能，聚焦大模型微调

针对这些的攻击与防御（最新的攻击、防御:跟Fine-tuning,VLM相关的）

攻击找10来篇，防御找20来篇。

在overleaf里建个表，找到参考文献就引用上，例如(GlobalCome2023 引用, 简介)

最终把搜索关键词由`("federated learning" OR "distributed learning") AND "vision models" AND "fine-tuning attacks" AND (IoT OR robotics OR "embodied intelligence")`简化为了`"vision models" AND "fine-tuning attacks"`，还一共只搜索出来了5篇。

<details><summary>忘记限制“视觉大模型的结果”：</summary>

```bibtex
@article{attack01,
    title   = {Emerging Safety Attack and Defense in Federated Instruction Tuning of Large Language Models},
    author  = {Ye, Rui and Chai, Jingyi and Liu, Xiangrui and Yang, Yaodong and Wang, Yanfeng and Chen, Siheng},
    journal = {arXiv preprint arXiv:2406.10630},
    year    = {2024}
}

@inproceedings{attack02,
    title        = {Adversarial attacks and defenses in large language models: Old and new threats},
    author       = {Schwinn, Leo and Dobre, David and G{\"u}nnemann, Stephan and Gidel, Gauthier},
    booktitle    = {Proceedings on},
    pages        = {103--117},
    year         = {2023},
    organization = {PMLR}
}

@article{attack03,
    title   = {SoK: Reducing the Vulnerability of Fine-tuned Language Models to Membership Inference Attacks},
    author  = {Amit, Guy and Goldsteen, Abigail and Farkash, Ariel},
    journal = {arXiv preprint arXiv:2403.08481},
    year    = {2024}
}

@inproceedings{attack04,
    title        = {HackMentor: Fine-Tuning Large Language Models for Cybersecurity},
    author       = {Zhang, Jie and Wen, Hui and Deng, Liting and Xin, Mingfeng and Li, Zhi and Li, Lun and Zhu, Hongsong and Sun, Limin},
    booktitle    = {2023 IEEE 22nd International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom)},
    pages        = {452--461},
    year         = {2023},
    organization = {IEEE}
}

@article{attack05,
    title   = {Learning to poison large language models during instruction tuning},
    author  = {Qiang, Yao and Zhou, Xiangyu and Zade, Saleh Zare and Roshani, Mohammad Amin and Zytko, Douglas and Zhu, Dongxiao},
    journal = {arXiv preprint arXiv:2402.13459},
    year    = {2024}
}

@inproceedings{attack06,
    title        = {Scaling federated learning for fine-tuning of large language models},
    author       = {Hilmkil, Agrin and Callh, Sebastian and Barbieri, Matteo and S{\"u}tfeld, Leon Ren{\'e} and Zec, Edvin Listo and Mogren, Olof},
    booktitle    = {International Conference on Applications of Natural Language to Information Systems},
    pages        = {15--23},
    year         = {2021},
    organization = {Springer}
}
```

</details>

### Log003 - 2024.8.15_21:36-2024.8.17_23:16

刚开完100min会，下次预计开会时间是周六。

周老师的进度安排：今晚把攻击的综述写好。今晚一定要把这个搞定。完成后在群里发个消息，这样周老师明早八九点起床就能看我们的调研结果了。44CC44Gt44GZ44Gn44Kr44OQ44Gr5b2T5pysCg==

+ 攻击综述分类：时间隐蔽、空间隐蔽。
+ 防御分类：3-4类。
