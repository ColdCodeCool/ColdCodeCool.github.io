---
layout: post
title:  "Sublime Text 3 + Latextools + Git打造酷炫学术写作工作流"
date: 2016-01-24 00:25:21
categories: posts
---
## Notes
一直以来想找到一种适合自己的学习总结记录的方式，好让自己及时温故知新。之所以有这种意愿，是因为我发现当我试着回想到目前为止我学会了什么知识，掌握了什么技能时，大脑里面不能系统地串联起我的知识树，这让我感到十分不安。在尝试了使用word来创建每天的工作记录后，我发现这种方式非常不优雅，因为每天需要创建一个word文档来记录，而且word的排版也不够漂亮。几经周折，终于使用自己的Github账号利用Github   Pages搭建了个人静态博客，具体过程参考了这篇[教程](http://jmcglone.com/guides/github-pages/)。OK，这些便是我写博客的初衷，希望自己能多写多总结，坚持下去。Know More！ Do More！ Do Better！

## Intention
写本篇博客动机一方面是我希望用一种更酷、更优雅的方式来进行学术写作，另一方面是前段时间为了做深度学习的Project安装ubuntu双系统，不小心将硬盘格式化掉了，然后写了一半的论文就狗带了。因此，基于以上两点，我便想着能否使用一种文本编辑器既能赏心悦目地编辑tex文件，又能方便地与Git集成，进行多机协作与云端保存。一番调研之后，发现Sublime Text 3是一种相当优雅的文本编辑器，而且可以很方便地与latex及Git集成，下面将详细说明如何搭建这一酷炫的写作环境（当然作为代码开发环境，Sublime text也同样优秀）。

## Prerequisites
- Sublime Text 3. 作为Sublime Text 2的演进，Sulime Text 3在功能上做了一些加强和改进，目前处于测试版本，在[这里](http://www.sublimetext.com/3)可以下载对应于你电脑操作系统的版本。
- Texlive. 既然我们要写tex文件，我们就必须拥有可以编译tex文件的编译器，通过Sublime Text设置快捷键组合，我们可以很方便的编译tex文件，并实时预览pdf。在这里我选用的是texlive，因为它是跨平台的，但我们仍然可以使用Miktex编译器，两种都OK。
- LatexTools. 这是一款针对在Sublime Text上进行学术写作开发的一款插件，其功能十分强大，正是由于这款插件的存在，Sublime Text才得以与Texlive完美配合，构成一种全新的、酷炫的学术写作环境。
- Git. 相信喜爱编程，喜爱开源项目的童鞋们对此工具都不陌生。不过在这里，Git有两层含义，一是Git工具本身，二是Sublime Text中的一款插件，这款插件的作用就是可以使我们在Sublime Text中直接调用Git的命令，而不必切换到命令行操作，将工作流的两步合二为一。以下为配置成功之后的截图：![image](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/2016-01-24.png)

## Setup
下面将以windows操作系统为例，step by step地介绍如何搭建写作环境。

### 安装Sublime Text 3 与 LatexTools
本软件为收费软件，下载安装之后需要注册激活，因为注册码不与MAC绑定，所以一个码可以在多台机器上安装，可以在网络上找到共享的激活码，但是建议大家手头宽裕的话购买正版注册码。当然，这里依然为大家提供一个可用的[注册码](http://my.oschina.net/xldc/blog/486654)。

- 安装Package Control. 安装好Sublime Text 3之后，还需要安装一个必不可少的插件Package Control，通过这一插件我们可以很方便地安装其他插件。Package Control的安装方法见此[链接](http://jingyan.baidu.com/article/f71d60379b20071ab641d181.html)。此插件安装好之后，重启sublime，启用快捷键组合“Ctrl+Shift+p”打开命令面板，键入“pci”，这时出现下拉菜单，包含一系列的Package Control指令。
- 通过Package Control安装LatexTools. 我们选择第一个Package Control:Install来安装我们的LatexTools插件，在命令面板输入latextools，不出意外，第一个便是LatexTools插件，选择并回车，这时Package Control会自动从互联网上搜索并安P装此插件，同样地，如果需要安装其他插件，也可遵循此步骤。指定tex文件编译器(MikTex或Texlive)路径，方法是在Sublime上侧导航栏依次选择`Prefrences`,`PackageSettings`,`LatexTools`。这时我们会看到下图菜单:![](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/Menu_002.png),其中`Settings-default(read-only)`为插件默认设置，一般情况下我们不可以修改这里的配置，我们需要进行的自定义配置是在`Settings-User`中进行。我们进入其中，找到如图配置选项:![](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/Selection_002.png)。
- 设置tex编译器路径与PDF预览器路径. 在`Textpath`一栏，填入你预先下载好的Texlive或MikTex编译器的路径，如“C:\\\texlive\\\2011\\\bin\\\win32;$PATH”，在此之前我们需要先设置Texlive的环境变量。我们可以参考这个[教程](http://jingyan.baidu.com/article/d2b1d10288bcb25c7f37d467.html),只需要将教程中的环境变量改为你安装在电脑中的texlive的路径即可，如C:\\texlive\\2011\\bin\\win32。这步完成之后，我们就可以使用sublime text 3编辑tex文件了，编译的默认快捷键组合为“Ctrl+b”。但是，此时我们还需要一个PDF预览器，这里我们使用SumatraPDF。同时我们也可以看到图片当中有“sumatra”这一栏，这一栏我们需要将SumatraPDF的exe完整路径填入双引号中。因此，我们也需要预先下载SumatraPDF。
- 设置反向搜索. 反向搜索是指在编译tex文件生成pdf之后，点击pdf当中的段落，自动跳转到tex文件对应的段落,设置教程请看[这里](http://blog.yuelong.info/post/st-latextools-readme-2.html)。
至此，我们已经完整配置好了Sublime Text 3下的latex文件编写的环境，可以愉快地进行latex写作了。但是，这仍然是一个本地编写环境，下面我将介绍如何将此环境与Git集成。

## Sublime Text 3与Git集成
在进行这一步之前，我们首先需要注册一个GitHub账号，以及在我们的电脑中安装Git。对开源社区有一定了解的童鞋相信对GitHub都不陌生，如果从来没接触过，也没有关系，下面我还是以windows系统为例，一步步将Sublime Text与Git集成起来。

- 注册GitHub账号并创建第一个代码仓库（repository）. 首先，我们登陆GitHub的[官网](https://github.com),注册并登陆GitHub。然后，我们创建你的第一个repository。点击如下图所示的`New repository`按钮:![](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/GitHub - Mozilla Firefox_004.png)，然后在repository name当中填入你自定义的代码仓库名称，因为这里我们是要将论文同步到GitHub，所以我们不妨将名字起为*Articles*。由于是写论文，我们在发表之前肯定不希望别人看到我们的idea，我们本应该将这个代码仓库的属性选择为private，但是GitHub的免费账号是没有权限将代码仓库属性设置为private的，所以我们只能选择public。但是我们要注意隐私保护，尽量不要让其他人知道你的github地址，这样其他人将无法在GitHub上看到你的论文手稿除了你自己。
- 在电脑上安装Git，并将在GitHub上刚刚创建的Articles仓库克隆到本地. 在[这里](http://www.git-scm.com/)，下载对应系统版本(32或64bit)的Git安装包，将其安装在自己熟悉的路径下，并在桌面创建Git的快捷方式。然后我们就可以运行桌面的Git bash，然后依次运行指令`$ git config --global user.name "xxx"//给自己起个用户名`，`$ git config --globla user.email  "yourname@gmail.com"//填写自己的邮箱`，然后运行指令`git clone git@github.com:yourusername/Articles.git E:/Articles`，你需要将指令中的yourusername替换成你的GitHub账号名称。上述指令将在你的电脑E盘创建一个名为*Articles*的文件夹，并将你在GitHub上创建的代码仓库复制到本地。实际上，你的代码仓库地址可以直接由图中![](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/ColdCodeCool-article - Mozilla Firefox_004.png)红色方框中的地址获得。注意，这里一定要选择SSH地址，因为我们在后续本地写好论文手稿后push到GitHub保存时，SSH方式只需要进行一次设置，而HTTPS方式则在我们每次push时，都需要你输入用户名和密码。因此，切记这一步我们在git clone指令后，填上图中的SSH地址。
- 安装SidebarEnhancement. 在上一步完成之后，我们便可以打开Sublime Text 3，从导航栏选择`File`，然后选择`Open Folder`，选择打开我们上一步中的Articles文件夹。这时我们将在Sublime左侧看到文件目录，然后我们需要下载一个插件SideBarEnhancement，打开Package Control，选择Install，输入sidebar，安装搞定。这个插件的作用是增强sublime左侧目录边栏的功能，安装了这个插件之后，我们将可以在左侧边栏点击鼠标右键，创建或删除文件。然后我们开始创建一个xx.tex文件，可以愉快地papering了。:)
- 安装Git插件. 注意，这里的Git插件，并不是我们第二步下载的Git。这里的Git插件，是一个能让Sublime Text 3在自己的命令面板中调用Git指令的工具。同样地，我们通过Package Control安装这个插件，通过快捷键组合“Ctrl+Shift+p”调出sublime命令面板，输入“pci”调用Package Control Install功能，输入Git搜索“Git”插件，另外一个选择是“SublimeGit”，两个插件都具备完整的Git功能。安装完之后，我们写完了tex，要将其同步到GitHub保存，但我们还需要将本地电脑与GitHub服务器产生映射。
- 生成密钥对，并将公钥添加到GitHub. 在桌面打开Git bash，输入指令`$ ssh-keygen -t rsa -C "your_email@youremail.com"`。指令中的youremail需要替换成你注册GitHub的邮箱，这一指令将在你的C盘当前用户即User文件夹中生成一个.ssh文件夹，里面包含了一对密钥，将后缀为pub的公钥文件用记事本打开，复制里面的所以内容。然后登陆GitHub，点击右上角自己的头像，选择Settings，然后选择左侧导航栏的`SSH keys`，在出现的右侧选择`Add SSH key`，将刚才复制的内容粘贴进去，保存。
- Git status 与 Git Push. 至此，我们已经可以将自己修改过的tex文件同步到GitHub远程仓库了。所有的Git指令都可以在Sublime Text 3内部完成，方法仍然是按快捷键组合'Ctrl+Shift+p'，输入“gs”出现如图控制面板:![](https://github.com/ColdCodeCool/ColdCodeCool.github.io/raw/master/images/Menu_005.png)。“Git status”展示了你对文档所进行的修改，按“s”将保存这些修改，然后按“c”将提交这些修改到Git的工作空间。接下来，我们继续在我们编辑的tex页面下使用“Git Push”，依然是在命令面板中输入“gp”，然后回车，这一动作将把你修改过的tex最新版本同步到GitHub保存。
- 多机协作. 若你需要在另一台电脑继续写作论文，可以按照以上步骤在这台机器上配置相同的写作环境，然后在Sublime Text 3的命令面板输入“Git Pull”，这一指令将会把你最后一次提交到GitHub上的文件克隆到该机，并与当前版本merge。

## Conclusion
好了，以上内容是对最近一周配置写作环境的总结，当然，如何配置本博客的环境本文并没有展开。不可避免地，受个人思路限制，本文难免忽略了某些具体细节，如果对本文所述内容有任何疑惑，欢迎给我发邮件*liyangsh48@gmail.com*交流。
