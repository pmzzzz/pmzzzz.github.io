# youtube-dl以及windows常用命令


免费下载youtube视频的**命令行**工具
<!--more-->

## youtube-dl
youtube-dl用于下载youtube视频
### 指定为最好画质
```bash
yt-dlp -f best 1
```
### 通过文件下载
```bash
yt-dlp -a file.txt 
```

### 提取MP4
```bash
ffmpeg -i "original_filename.mkv" -codec copy output_name.mp4
```

### 提取音频
```bash
ffmpeg -i input.mp4 -vn -c:a mp3 output.mp3
```
### 提取视频
```bash
ffmpeg -i input.mp4 -an -c:a copy output.mp4
```


## windows常用命令
值得注意的一些小tips
### windows包管理器 winget、安装sudo

```bash
winget install gsudo 
```
### 查看命令位置
```bash
Get-Command gsudo 
```


### 查看环境变量的值
```bash
$Env:POSH_THEMES_PATH 
```
### 编辑profile
```bash
code $profile
```
code 指的使用vscode

### posh是美化控制台的程序
```bash
Get-PoshThemes 
gsudo oh-my-posh font install
gsudo set-executionpolicy remotesigned
```
### 控制台优化程序
```
oh-my-posh 
```

### 临时管理员权限
在命令前加上这个可以临时获取管理员权限，跟sodu一样
```bash
gsudo 
```
### 当运行不了脚本的时候使用
```bash
Get-ExecutionPolicy
```

