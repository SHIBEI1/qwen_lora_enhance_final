# 已验证的环境

`requirements-lock.txt` 根据打包时 `myenv` 内的实际包版本生成，去除了指向开发目录或临时构建目录的 editable/file URL。PyTorch `2.7.1+cu128` 与 torchvision `0.22.1+cu128` 由部署入口从官方 CUDA 12.8 索引单独安装。

`conda-explicit.txt` 记录此次运行机器的 Linux Conda 底层环境，供版本追溯；可迁移部署入口仍为 `scripts/setup_environment.sh`，默认创建或复用 `myenv`。Conda 环境本身不属于项目文件，不复制原机器的绝对路径环境目录。

Musubi Tuner 随项目携带并固定版本；运行入口优先加载项目内源代码。新机器运行部署脚本时会安装当前项目内的后端，而不是回到开发备份目录安装。
