# WSL配置深度学习环境


在正式配置之前，不要忘了在win11上下载  Geforce EXperience，然后更新最新的驱动。
> https://www.nvidia.com/en-us/geforce/geforce-experience/
## 安装cuda

### toolkit安装

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/cuda-repo-wsl-ubuntu-12-5-local_12.5.1-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-5-local_12.5.1-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-5-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-5
```
最新版本参见官网：
> https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local

### 环境变量

打开bashrc
```bash
vim ~/.bashrc
```
增加如下内容：

```bash
export PATH=$PATH:/usr/local/cuda-12.5/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.5/lib64
```
检查安装：
```bash
source ~/.bashrc
nvcc -V
```

## 安装cudnn

```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.2.1/local_installers/cudnn-local-repo-ubuntu2004-9.2.1_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2004-9.2.1_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2004-9.2.1/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn-cuda-12
```

## 安装nimiconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
## 创建torch环境
```bash
conda create -n py310 python=3.10
conda activate py310
# conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

## 修改pip源，安装torch
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install torch torchvision torchaudio
```
检验是否安装成功：
```python
import torch
torch.cuda.is_available()
# True 
torch.cuda.get_device_name()
# 'NVIDIA GeForce RTX 2060'
```
thanks to:
> https://cloud.tencent.com/developer/article/1710564

## 将WSL ssh映射到局域网
原理是通过windows端口转发将WSl的22端口转发出去
重装ssh
```bash
sudo apt-get remove openssh-server
sudo apt-get install openssh-server
```
修改配置文件
```bash
sudo vim /etc/ssh/sshd_config
```
修改如下内容
```
Port 22
PermitRootLogin Yes
PasswordAuthentication Yes
```
```bash
sudo vim /etc/hosts.allow
```
添加如下内容
```
sshd:ALL
```

重启ssh
```bash
sudo service ssh --full-restart
```

在windows shell 中运行

```shell
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=22 connectaddress=172.19.247.91 connectport=22
```

其中`172.19.247.91`替换为wsl的地址（ifconfig）

修改winsows防火墙，允许22入规则。

此时可以使用ssh连接WSL,ip地址为windows的ip地址。
```bash
ssh root@192.168.0.2
```

thanks to：

> https://gitcode.csdn.net/65e8400d1a836825ed78b888.html
> https://learn.microsoft.com/zh-cn/windows/wsl/wsl-config#configuration-settings-for-wslconfig
