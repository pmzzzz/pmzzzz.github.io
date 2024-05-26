# MissingSemester day3

## 课程总结
1. vim 编辑器的设计思想
2. vim 编辑器的基本使用

## 重点内容
### vim的哲学
vim 被称为*多模态*(modal)编辑器:它对于插入文字和操纵文字有不同的模式。Vim 是可编程的（可以使用 Vimscript 或者像 Python 一样的其他程序语言），Vim 的接口本身也是一个程序语言：键入操作（以及其助记名） 是命令，这些命令也是可组合的。Vim 避免了使用鼠标，因为那样太慢了；Vim 甚至避免用 上下左右键因为那样需要太多的手指移动。

### vim的模式
- Normal: for moving around a file and making edits
- Insert: for inserting text
- Replace: for replacing text
- Visual (plain, line, or block): for selecting blocks of text
- Command-line: for running a command

### 移动 movement
即光标的移动。在normal模式和Visual模式都能用。
- 基本移动： `hjkl` --> 左下上右
- 单词移动：`w`后一个单词的开始，`b`前一个单词的开始，`e`单词的结尾
- 行内移动：`0`行首，`^`首个非空字符,`$`行尾
- 屏幕移动： `HML` --> 屏幕最高、中间、最低行 high，middle、low
- 翻页    ：`<C-D>`、`<C-U>` ---> 向上、向下翻页
- 文件    : `G`文件末尾，`gg`文件开头 (g is for go)
- 行号    ： `{行号}G` 去行号(23G),或者`:{行号}` 
- 行内查找：`t{char}`、`f{char}`向后查找,`T{char}`、`F{char}`向前查找 【f到该字符上，t在该字符前面】
	- `,`、`;`查找结果跳转--->前、后
- 搜索: `/{正则表达式}`, `n` / `N` 用于导航匹配

### 选择 Selection

配合前面的移动来选择字符

- `v`选择模式,普通模式
- `V`行选择，一定是选择几行的内容
- `<C-V>`块选择，按照矩形框进行选择

### 编辑 edit
1. 在普通模式下进行简单编辑
  - 删除`d{移动}`: 例如 `dw` 即删除一个单词，`d$`删除这行后面的内容
  - 删除`x`：删除单个字符 == `dl`
  - 删除一行：`dd`
  - 替换单个字符：`r`
  - 替换字符再插入： `s` == `xi`
  - 改变字符 :`c{移动}` : `cw` 改变一个单词 == `dwi` 
  - 后/前增一行:`o`/`O`
  - 复制粘贴：`y` -- `p` 注意`d`删除的文字也能粘贴
  - 撤销重做: `u`,`<C-R>`
  - 改变大小写: `~`
### 多窗口
- 用 `:sp` / `:vsp` 来分割窗口
- 同一个缓存可以在多个窗口中显示。
- `<C-w{hjkl}>` 移动窗口、`<C-w>w`回到上一个窗口

### 搜索和替换
- 搜索: `/{string}`向下搜索字符串，`?{string}`向上
- 使用`n`/`N` 跳转搜到的词
- 替换 `:s/a/b/g`、`:%s/a/b/g`【行内/全文替换a --> b】
### vim 命令
普通模式下输入`:`进入vim命令
- `:r`:读取
- `:w`:写入
- `:q`:退出当前窗口

## 课外补充

### 复制到系统剪切板
`"+y`:将选择的内容复制到`+`寄存器中。

> 详见-->[vim与系统剪切板之间的复制粘贴](https://www.cnblogs.com/gmpy/p/11177719.html)

### 使用shell命令

`:!cmd`:`:!echo hello`

### vimtutor

边做边学文档：[vimtutor](https://github.com/vim/vim/blob/master/runtime/tutor/tutor)

------------

Edit with VIM!


