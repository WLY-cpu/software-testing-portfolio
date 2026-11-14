# 在 Windows 上自己搭一次 Linux 环境（WSL）

> 目的：把"只在大赛给好的环境里用过"变成"自己装过、自己配过"。
> 预计 1—2 小时，踩坑是正常的，坑记下来就是你的笔记。

## 1. 开启 WSL

以**管理员身份**打开 PowerShell：

```powershell
wsl --install
```

装完重启电脑。重启后会自动装好 Ubuntu 并让你设置用户名和密码（密码输入时不显示，正常）。

检查：

```powershell
wsl --list --verbose
```

应该看到 `Ubuntu` 和 `Running`。

## 2. 换国内镜像源（不然下载会很慢）

```bash
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
sudo nano /etc/apt/sources.list
```

把 `archive.ubuntu.com` 和 `security.ubuntu.com` 替换成 `mirrors.tuna.tsinghua.edu.cn`（nano 里用 `Ctrl+\` 替换），保存：`Ctrl+O` 回车，退出：`Ctrl+X`。

```bash
sudo apt update && sudo apt upgrade -y
```

## 3. 装 Python 和虚拟环境

```bash
sudo apt install -y python3 python3-pip python3-venv git
python3 --version
git --version
```

## 4. 建一个自己的虚拟环境

```bash
mkdir -p ~/projects && cd ~/projects
python3 -m venv venv
source venv/bin/activate          # 前面会出现 (venv)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy pandas
```

## 5. 跑通一个脚本

```bash
mkdir -p ~/projects/test-run && cd ~/projects/test-run
cat > hello.py << 'EOF'
import sys, platform
print("Python:", sys.version.split()[0])
print("系统:", platform.platform())
EOF
python hello.py
```

看到 Python 版本和系统信息，就说明环境通了。

## 6. 把踩的坑记下来（这一步别省）

装的过程中一定会卡住几次，卡住就是收获。按这个格式记，记满 2 条这份指南才算真的属于你：

| 现象 | 原因 | 怎么解决的 |
|---|---|---|
|  |  |  |
|  |  |  |

## 完成标准

- [ ] `wsl` 能启动 Ubuntu
- [ ] `python3 --version` 有输出
- [ ] 跑通 `hello.py`
- [ ] 上表填了至少 2 条
