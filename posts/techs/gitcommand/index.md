# git常用命令


## 创建版本库
1. git init   
初始化仓库
2. git add <file>   
添加到仓库
3. git commit -m "message"  
提交到仓库
## 版本相关
### 版本退回
1. git log --pretty=oneline  
查看版本历史记录
2. git reset --hard HEAD^  
退回上个版本，HEAD表示当前版本，文件也退回
3. git reset --hard <commit id>  
commit id 可以在git log中找到
4. git reflog  
所有命令的记录
### 工作区和暂存区
1. git status  
查看状态

### 管理修改  
1. git diff HEAD -- <filename>
查看工作区和版本库里面最新版本的区别

### 撤销修改
1. git checkout -- file  
丢弃工作区的修改，返回到上一次add或者commit状态
2. git reset HEAD <file>   
撤销掉暂存区的修改，即把暂存区放回工作区

### 删除文件

1. git rm  
从版本库删除文件  
先手动删除**工作目录**的文件，再使用git rm 和gt add的效果是一样的 
如果删错了，可以用 git checkout -- filename 退回版找回来
## 远程仓库
### ssh链接
1. cd ~
去到主目录
1. ssh-keygen -t rsa -C "email"
生成密钥，将.pub的内容分享给别人
### 添加远程仓库
1. git remote add origin git@github.com:pmzzzz/xxxx.git
添加远程库名叫origin，origin是远程库默认名字
3. git branch -M <name>  
命名主节点
2. git push -u origin master
第一次加 -u，master是本地的节点

### 从远程库克隆
1. git clone git@github.com:pmzzzz/xxxx.git
xxxx是自己的仓库名，会在当前目录下载整个仓库  
还可以用https://github.com/michaelliao/gitskills.git这样的地址

## 分支管理
### 创建与合并分支
1. git branch  
查看分支
2. git branch name
创建分支
3. git checkout name /switch name
切换分支
4. git checkout -b name /switch -c name
创建后切换到分支
5. git merge name 
合并分支
6. git branch -d name  
删除分支




 

## 去掉代理设置
git config --global --unset http.proxy

git config --global --unset https.proxy
