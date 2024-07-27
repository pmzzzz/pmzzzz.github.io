# 苹果tips

## 软件安装相关
### macOS无法验证此App不包含恶意软件

![](https://cdn.jsdelivr.net/gh/pmzzzz/picbed/img/20210604104324.png)

```
sudo spctl --master-disable
```
```
sudo xattr -r -d com.apple.quarantine /applications/xxx.app
```

## brew相关

### 安装brew

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```


### brew 换源
```
cd “$(brew —repo)”
git remote set-url origin https://mirrors.ustc.edu.cn/brew.git
echo
‘export HOMEBREW_BOTTLE_DOMAIN=https://mirrors.ustc.edu.cn/homebrew-bottle
source~/. bash profile
cd “$(brew —repo)/Library/Taps/homebrew/homebrew-core”
git remote set-url origin https://mirrors.ustc.edu.cn/homebrew-core.git
```


## 命令行相关

### 环境变量编辑
```
code ~/.zshrc
```

## python相关

### jupyter notebook 切换虚拟环境
```

(base) conda install -c conda-forge jupyterlab
(base) conda install Jupyter notebook
 #安装这个之后就可以识别其他环境
(base) conda install nb_conda_kernels   

#在其他环境安装jupyter/ipykernel 就能被识别（ipykernel更轻量，推荐）
(ml) conda install jupyter      
(ml) conda install ipykernel

# 注册新的kernel也可以被识别
pip install ipykernels
python -m ipykernel install —user —name mykernelname
```
