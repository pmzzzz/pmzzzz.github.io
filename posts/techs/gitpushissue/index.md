# git push 出现 protocol error


## 问题现象
git push 我的博客的时候，出现如下提示
```bash
fatal: protocol error: bad line length 189
send-pack: unexpected disconnect while reading sideband packet
error: failed to push some refs to 'https://github.com/pmzzzz/myWebsite.git'
Enumerating objects: 32, done.                                                                                              
Counting objects: 100% (32/32), done.
Delta compression using up to 8 threads
Compressing objects: 100% (24/24), done.
```

## 问题原因

我记得我为了重装brew，修改了一个类似https协议“大小”的参数，具体是什么我给忘了，才导致了这个问题---protocol error

## 问题解决
这个问题解决纯粹是头痛医头了
```
git config --global http.postBuffer 524288000
```
这个解决方法是加大git文件到500M

至少现在看来，出现 protocol error: bad line length 修改缓存大小的方式是可行的
