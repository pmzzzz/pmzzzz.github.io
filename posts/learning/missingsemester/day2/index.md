# MissingSemester day2

## 课程总结

1. shell脚本的基础知识，变量的使用以及shebang
2. 常用工具的介绍，文件搜索工具和方法

## 重点内容

### shell脚本

#### 变量的定义和赋值

```shell
foo=bar
```

{{< admonition tip "提示">}}

**空格**在shell中十分敏感，切勿随意添加空格，如果写成 `foo = bar`它的意思是运行`foo`这个程序，第二和第三个参数分别为 = 和 bar，这明显不是我们想要的

{{< /admonition >}}

#### 变量的引用

```shell
echo $foo
# 输出 bar
```

使用`$`来引用变量的值

#### 单引号与双引号

```shell
foo=bar
echo "$foo"
# prints bar
echo '$foo'
# prints $foo
```

{{< admonition tip "提示">}}

从上面的例子可以看出，双引号会将其中的变量引用进行解释，而单引号是将其中的内容直接当作一个字符串。

{{< /admonition >}}

#### 特殊变量引用

若`mcd.sh`内容如下：

```shell
mcd () {
    mkdir -p "$1"
    cd "$1"
}
```

该脚本的功能为 新建一个文件夹，然后自动cd进去

```shell
source mcd.sh
mcd test
```

使用source file 将脚本的内容进行加载和执行，上面的例子即是将函数进行了加载

然后使用 `mcd` 命令来调用函数

------

值得注意的是，在脚本中，我们使用了`"$1"`来表示接受到的第一个参数，在这个例子中，我们的参数是`test`,所以它会创建名为`test`的文件夹，然后`cd`进去

**还有其他的特殊变量：**

|    名称     |                             意义                             |
| :---------: | :----------------------------------------------------------: |
|    `$0`     |                            脚本名                            |
| `$1`到 `$9` |          脚本的参数。 `$1` 是第一个参数，依此类推。          |
|    `$@`     |                           所有参数                           |
|    `$#`     |                           参数个数                           |
|    `$?`     |                      前一个命令的返回值                      |
|    `$$`     |                     当前脚本的进程识别码                     |
|    `!!`     | 完整的上一条命令，包括参数。<br />常见应用：当你因为权限不足执行命令失败时，可以使用 `sudo !!`再尝试一次。 |
|    `$_`    | 上一条命令的最后一个参数。<br />如果你正在使用的是交互式 shell，你可以通过按下 `Esc` 之后键入 . 来获取这个值。 |

#### 短接 [short-circuiting](https://en.wikipedia.org/wiki/Short-circuit_evaluation)

与C语言一样，逻辑表达式的执行，采用短接的形式，也就是说：当前面的语句已经能判断真假，后续的语句将不再执行

```shell
false || echo "Oops, fail"
# Oops, fail
true || echo "Will not be printed"
#
true && echo "Things went well"
# Things went well

false && echo "Will not be printed"
#
true ; echo "This will always run"
# This will always run

false ; echo "This will always run"
# This will always run
```

在这个例子中，`||`和`&&`使用了短接，而`;`代表将两个语句分割，没有短接效果。

#### 参数的综合运用

```shell
#!/bin/bash

echo "Starting program at $(date)" # Date will be substituted

echo "Running program $0 with $# arguments with pid $$"

for file in "$@"; do
    grep foobar "$file" > /dev/null 2> /dev/null
    # When pattern is not found, grep has exit status 1
    # We redirect STDOUT and STDERR to a null register since we do not care about them
    if [[ $? -ne 0 ]]; then
        echo "File $file does not have any foobar, adding one"
        echo "# foobar" >> "$file"
    fi
done
```

```shell
(base) ➜  day2 ./example.sh example.sh mcd.sh
Starting program at 2023年12月29日 星期五 23时48分43秒 CST
Running program ./example.sh with 2 arguments with pid 96149
File mcd.sh does not have any foobar, adding one
```

- 在shell中，不同的输出被分到了不同的流，2> 代表了错误流。> 代表标准输出流
- -ne(not equal) 可以通过 `man test` 查看其他比较运算符
- 在进行比较时，推荐使用两个中括号`[[ ]]`

#### 通配符*globbing*

- `*`代表若干字符
- `?`代表一个字符
- `{}`会进行笛卡尔积扩展

例如：

```shell
*.py #所有以.py结尾的文件
?.py #前有一个字符，且以.py 结尾的文件
a.{jpg,png} # a.jpg 和 a.png
{a,b}.{c,d} # a.c、a.d、b.c、b.d 四个文件
```



#### shebang

shebang的意义在于，告诉shell应该用什么解释器来执行某个脚本，它是写在脚本第一行的提示词，常见的shebang为`#!/bin/bash`这表示使用bash来执行这个脚本

`example.py`:

```python
#!/usr/local/bin/python
import sys
for arg in reversed(sys.argv[1:]):
    print(arg)
```

以上为一个python脚本，为了让shell知道使用哪个解释器我们添加了shebang:`#!/usr/local/bin/python`

#### 更好的shebang

```python
#!/usr/bin/env python
```

以上的shebang首先使用了env命令（这在unix like 机器中很常见），然后利用env命令找到这台机器的python解释器（即是环境变量、也是之前提到的path路径）

### shell工具

#### tldr - 简介明了的命令解释

tldr 是man的增强，它以更接近人类的语言描述命令的使用方法（需要第三方安装）

```shell
tldr ffmpeg #查看ffmpeg的使用方法
```

#### find - 查找文件

#### fd-find增强

#### locate-速度极快



## 作业

1. Read [`man ls`](https://www.man7.org/linux/man-pages/man1/ls.1.html) and write an `ls` command that lists files in the following manner

   - Includes all files, including hidden files
   - Sizes are listed in human readable format (e.g. 454M instead of 454279954)
   - Files are ordered by recency
   - Output is colorized

   A sample output would look like this

   ```shell
    -rw-r--r--   1 user group 1.1M Jan 14 09:53 baz
    drwxr-xr-x   5 user group  160 Jan 14 09:53 .
    -rw-r--r--   1 user group  514 Jan 14 06:42 bar
    -rw-r--r--   1 user group 106M Jan 13 12:12 foo
    drwx------+ 47 user group 1.5K Jan 12 18:08 ..
   ```

   ```shell
   ls -a -l -h -t --color=auto
   ```

2. Write bash functions `marco` and `polo` that do the following. Whenever you execute `marco` the current working directory should be saved in some manner, then when you execute `polo`, no matter what directory you are in, `polo` should `cd` you back to the directory where you executed `marco`. For ease of debugging you can write the code in a file `marco.sh` and (re)load the definitions to your shell by executing `source marco.sh`.
   文件`marco.sh`：

     ```shell
   #!/bin/bash
   # Function to save the current directory
   marco() {
       export MARCO_DIR=$(pwd)
       echo "Current directory saved as MARCO_DIR"
   }
   # Function to change directory back to the saved directory
   polo() {
       cd "$MARCO_DIR" || echo "MARCO_DIR is not set"
   }
     ```

     ```shell
   source marco.sh
     ```

3. Say you have a command that fails rarely. In order to debug it you need to capture its output but it can be time consuming to get a failure run. Write a bash script that runs the following script until it fails and captures its standard output and error streams to files and prints everything at the end. Bonus points if you can also report how many runs it took for the script to fail.

   ```shell
    #!/usr/bin/env bash
   
    n=$(( RANDOM % 100 ))
   
    if [[ n -eq 42 ]]; then
       echo "Something went wrong"
       >&2 echo "The error was using magic numbers"
       exit 1
    fi
   
    echo "Everything went according to plan"
   ```

   ```shell
   #!/usr/bin/env bash 
   
   # Output and error log files
   output_log="output.log"
   error_log="error.log"
   counter=0
   
   # Function to run the given script until failure
   run_script() {
       while true; do
           if ! ./quz3.sh >> "$output_log" 2>> "$error_log"; then
               break
           fi
           ((counter++))
       done
   }
   
   # Run the script and capture output/errors
   run_script
   
   # Print captured logs
   echo "Final Output:"
   cat "$output_log"
   echo
   echo "Final Errors:"
   cat "$error_log"
   echo
   echo "Number of runs until failure: $counter"
   ```

4. As we covered in the lecture `find`’s `-exec` can be very powerful for performing operations over the files we are searching for. However, what if we want to do something with **all** the files, like creating a zip file? As you have seen so far commands will take input from both arguments and STDIN. When piping commands, we are connecting STDOUT to STDIN, but some commands like `tar` take inputs from arguments. To bridge this disconnect there’s the [`xargs`](https://www.man7.org/linux/man-pages/man1/xargs.1.html) command which will execute a command using STDIN as arguments. For example `ls | xargs rm` will delete the files in the current directory.

   Your task is to write a command that recursively finds all HTML files in the folder and makes a zip with them. Note that your command should work even if the files have spaces (hint: check `-d` flag for `xargs`).

   ```shell
   find /your/directory/path -type f -name '*.html' -print0 | xargs -0 zip html_files.zip
   ```

   > -print0 和-0表示以\0为分隔符而不是空格，这样就能保证有空格的文件能被正确识别
   >
   > -dX 表示以X为分隔符

   If you’re on macOS, note that the default BSD `find` is different from the one included in [GNU coreutils](https://en.wikipedia.org/wiki/List_of_GNU_Core_Utilities_commands). You can use `-print0` on `find` and the `-0` flag on `xargs`. As a macOS user, you should be aware that command-line utilities shipped with macOS may differ from the GNU counterparts; you can install the GNU versions if you like by [using brew](https://formulae.brew.sh/formula/coreutils).

5. (Advanced) Write a command or script to recursively find the most recently modified file in a directory. More generally, can you list all files by recency?

   ```shell
    ls -t -1 -r /your/directory/path | head -n 1
   ```

      ```shell
   ls -t -1 -r /your/directory/path
      ```


- -t表示按照时间排序
- -1表示一行显示一个文件
- -r表示递归


