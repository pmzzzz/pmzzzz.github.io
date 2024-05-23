# StableDiffusion-图片预览失效

## 环境
- macos 12.7.2

## 现象
生成图片之后，预览窗口啥也没有，以至于不能进行后续操作
## 原因
代理软件让它需要的一个包下载不下来
## 解决
### 关闭代理
这是一个比较直接的解决方法，但确实很不方便
### 修改配置
在`webui-user.sh`中修改成以下配置:
```shell
export COMMANDLINE_ARGS="--skip-torch-cuda-test --upcast-sampling --no-half-vae --use-cpu interrogate --no-gradio-queue "
```
禁止这个包来避免这个问题
