@echo off
REM 使用 Git Bash 运行 rsync 命令
setlocal enabledelayedexpansion

REM 设置本地目录和远程服务器信息
set LOCAL_DIR=%cd%
set REMOTE_USER=lzy
set REMOTE_HOST=3090.narc.letmefly.xyz
set REMOTE_DIR=/home/lzy/ltf/Codes/LLM/wb2/Codes

REM 进入 Git Bash 环境
@REM bash -c "rsync -avz --delete --exclude-from='%LOCAL_DIR%/.gitignore' -e 'ssh -p 8922' '%LOCAL_DIR%/' '%REMOTE_USER%@%REMOTE_HOST%:%REMOTE_DIR%'"
ssh %REMOTE_USER%@%REMOTE_HOST% "rm -rf %REMOTE_DIR%"
scp -r %LOCAL_DIR% %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_DIR%

endlocal
@REM scp clip-vit-base-patch32 lzy@3090.narc.letmefly.xyz:/home/lzy/ltf/Codes/LLM/wb2/Dataset