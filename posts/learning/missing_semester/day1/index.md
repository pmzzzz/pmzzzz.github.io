# MissingSemester day1




## 课程概览

1. 本节课主要对整个课程进行了大致的描述，包括设置这门课的意义，以及内容安排。
2. 讲解了SHELL的基本概念和特性，以及一些程序和符号的使用（man、ls、mkdir、｜、<>、etc..）

## 重点

### 什么是SHELL

SHELL是文本交互界面，区别于图形化交互界面，SHELL通过文本与系统交互，比图形界面更能掌控我们的电脑

### shell如何知道你要运行的程序在哪里

shell会通过一个特定的顺序（不同的文件夹目录）来搜寻程序，如果找到了就会运行找到的第一个程序

通过 `$PATH` 能观察shell的搜索路径，下面的例子通过`echo`将它打印了出来

另外，可以使用`which` 命令，来告诉你如果运行某个程序，它具体来自哪个目录,例如`echo`就来自于`/bin/echo`

```shell
missing:~$ echo $PATH
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
missing:~$ which echo
/bin/echo
missing:~$ /bin/echo $PATH
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

### 文件和文件夹的访问权限

使用ls -l ，显示文件夹下文件的详细信息，其中第一列代表了文件/文件夹的权限

```shell
missing:~$ ls -l /home
drwxr-xr-x 1 missing  users  4096 Jun 15  2019 missing
```

- d代表这是一个文件夹
- 后续一共九个字符分为三组，每一组都是按照rwx(读、写、执行)的顺序排列，如果有权限则限制字母，反正则显示`-` 
- 三组字母分别代表用户、组、其他人对该文件\文件夹的权限

{{< admonition tip "tips">}}

文件夹的执行，代表是否能够打开文件夹，文件夹的读，代表能否查看内容，文件夹的写，代表能否删除内部的文件

{{< /admonition >}}

### 连接程序,重定向与管道

通过 `<` ，`>` ，`|` 可以将程序的输入输出流连接起来

首先要知道的是：对于一个程序来说，一般有输入流和输出流，例如从键盘输出，然后打印到屏幕上

```shell
(base) ➜  TheMissingSemester echo hello
hello
```

#### 使用`><`符号将输入输出流重定向：

```shell
missing:~$ echo hello > hello.txt
missing:~$ cat hello.txt
hello
missing:~$ cat < hello.txt
hello
missing:~$ cat < hello.txt > hello2.txt
missing:~$ cat hello2.txt
hello
```

最后一个例子中，使用`cat < hello.txt`读取文件的内容，它会有一个输出流，然后使用`> hello2.txt` 将输出流定向到了`hello2.txt`中

可以发现，重定向是将文件输入到程序，或者将程序的结果输出到文件中。

#### 使用管道连接程序

```shell
missing:~$ ls -l / | tail -n1
```

以上程序会将ls的结果作为tail的输入，效果为只显示当前文件夹最后一个文件。

### 重定向与管道的区别

1、管道命令只处理前一个命令正确输出，不处理错误输出

2、管道命令右边命令，必须能够接收标准输入流命令才行。

```shell
[chengmo@centos5 shell]$ cat test.sh | grep -n 'echo'
5:    echo "very good!";
7:    echo "good!";
9:    echo "pass!";
11:    echo "no pass!";
#读出test.sh文件内容，通过管道转发给grep 作为输入内容
 
[chengmo@centos5 shell]$ cat test.sh test1.sh | grep -n 'echo'
cat: test1.sh: 没有那个文件或目录
5:    echo "very good!";
7:    echo "good!";
9:    echo "pass!";
11:    echo "no pass!";
#cat test1.sh不存在，错误输出打印到屏幕，正确输出通过管道发送给grep 
 
 
[chengmo@centos5 shell]$ cat test.sh test1.sh 2>/dev/null | grep -n 'echo' 
5:    echo "very good!";
7:    echo "good!";
9:    echo "pass!";
11:    echo "no pass!";
#将test1.sh 没有找到错误输出重定向输出给/dev/null 文件，正确输出通过管道发送给grep
 
 
[chengmo@centos5 shell]$ cat test.sh | ls
catfile      httprequest.txt  secure  test            testfdread.sh  testpipe.sh    testsh.sh      testwhile2.sh
envcron.txt  python           sh      testcase.sh     testfor2.sh    testselect.sh  test.txt       text.txt
env.txt      release          sms     testcronenv.sh  testfor.sh     test.sh        testwhile1.sh

```

> **区别是：**
>
> 1、左边的命令应该有标准输出 | 右边的命令应该接受标准输入
>   左边的命令应该有标准输出 > 右边只能是文件
>   左边的命令应该需要标准输入 < 右边只能是文件
>
> 2、**管道**触发两个子进程执行"|"两边的程序；而**重定向**是在一个进程内执行



## 练习与答案

### 题目

1. For this course, you need to be using a Unix shell like Bash or ZSH. If you are on Linux or macOS, you don’t have to do anything special. If you are on Windows, you need to make sure you are not running cmd.exe or PowerShell; you can use [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/) or a Linux virtual machine to use Unix-style command-line tools. To make sure you’re running an appropriate shell, you can try the command `echo $SHELL`. If it says something like `/bin/bash` or `/usr/bin/zsh`, that means you’re running the right program.

2. Create a new directory called `missing` under `/tmp`. 
  ```shell
   mkdir /tmp/missing
  ```
3. Look up the `touch` program. The `man` program is your friend.
	```shell
	man touch
	```

4. Use `touch` to create a new file called `semester` in `missing`.
	```shell
	touch /tmp/missing/semester
	```
5. Write the following into that file, one line at a time:

   ```
   #!/bin/sh
   curl --head --silent https://missing.csail.mit.edu
   ```

   ```shell
   echo '#!/bin/sh' > semester
   echo "curl --head --silent https://missing.csail.mit.edu" >> semester
   # 注意使用单引号来包括#
   ```
   The first line might be tricky to get working. It’s helpful to know that `#` starts a comment in Bash, and `!` has a special meaning even within double-quoted (`"`) strings. Bash treats single-quoted strings (`'`) differently: they will do the trick in this case. See the Bash [quoting](https://www.gnu.org/software/bash/manual/html_node/Quoting.html) manual page for more information.

6. Try to execute the file, i.e. type the path to the script (`./semester`) into your shell and press enter. Understand why it doesn’t work by consulting the output of `ls` (hint: look at the permission bits of the file).

7. Run the command by explicitly starting the `sh` interpreter, and giving it the file `semester` as the first argument, i.e. `sh semester`. Why does this work, while `./semester` didn’t?

8. Look up the `chmod` program (e.g. use `man chmod`).

9. Use `chmod` to make it possible to run the command `./semester` rather than having to type `sh semester`. How does your shell know that the file is supposed to be interpreted using `sh`? See this page on the [shebang](https://en.wikipedia.org/wiki/Shebang_(Unix)) line for more information.
  ```shell
  chmod 755 smester
  ```
10. Use `|` and `>` to write the “last modified” date output by `semester` into a file called `last-modified.txt` in your home directory.
  ```shell
  ./semester | grep last-modified | cut -d ';' -f 1  > ~/last-modified.txt
  ```
11. Write a command that reads out your laptop battery’s power level or your desktop machine’s CPU temperature from `/sys`. Note: if you’re a macOS user, your OS doesn’t have sysfs, so you can skip this exercise.


