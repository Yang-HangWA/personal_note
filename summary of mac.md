# Summary of mac

### mac没有声音
   - 有时候 Mac 从睡眠状态恢复之后没有声音，这是 Mac OS X 系统的一个 Bug。这是因为 Mac OS X 的核心音频守护进程「coreaudiod」出了问题，虽然简单的重启电脑就能解决，但是如果此时开启了很多程序后者有其他情况不想重启电脑的话，可以按照下面的方法解决此问题。
    - 操作步骤：
        - 1、在 Mac 中打开活动监视器（在 Finder 的「应用程序」中搜索「活动监视器」可以找到）。
        - 2、在「活动监视器」窗口右上角的搜索框里输入「audio」，此时可以搜索到「coreaudiod」进程。
        - 3、选中「coreaudiod」进程，点击「活动监视器」窗口左上角的「退出进程」按钮，在弹出的对话框中点击「退出」。
        - 4、「coreaudiod」进程退出后会自动重启，这时声音就恢复了。 
    
    - 更简单的做法是 @张宁的方法
       - 在终端输入sudo killall coreaudiod命令并输入密码即可。终端在“应用程序-实用工具-终端”。特好使~


### mac git clone 提速
- 解决链接：https://blog.csdn.net/Carty090616/article/details/98598933

* 1.设置git的代理
    ```
    git config --global http.proxy 'socks5://127.0.0.1:1086' # 1086是根据自己电脑查出来的
    git config --global https.proxy 'socks5://127.0.0.1:1086'

    sudo killall -HUP mDNSResponder
    ```

    - 当1086显示refused后，换用1080又可以了，感觉好奇怪

    - 实际例子：
    ```
    yanghang@192  ~/paper  git clone https://github.com/wzhe06/Reco-papers.git                ✔  1310  15:57:41
    Cloning into 'Reco-papers'...
    fatal: unable to access 'https://github.com/wzhe06/Reco-papers.git/': Failed to connect to 127.0.0.1 port 1086: Connection refused
    yanghang@192  ~/paper  git config --global https.proxy 'socks5://127.0.0.1:1080'      128 ↵  1311  15:58:14
    yanghang@192  ~/paper  git config --global http.proxy 'socks5://127.0.0.1:1080'           ✔  1312  15:58:40
    yanghang@192  ~/paper  git clone https://github.com/wzhe06/Reco-papers.git                ✔  1313  15:58:45
    Cloning into 'Reco-papers'...
    remote: Enumerating objects: 229, done.
    Receiving objects:  19% (44/229), 20.52 MiB | 567.00 KiB/s
    ```
* 2.完成上述步骤之后clone的速度就会变快了，但只限于GitHub

* 3.关闭git代理
    ```
    git config --global --unset http.proxy
    git config --global --unset https.proxy
    ```

* mac合上盖子保持不休眠
https://www.zhihu.com/question/312052061

    - 给它喝咖啡（注射咖啡因）
    在终端当中输入
    caffeinate -t 3600
    就可以让它保持一小时清醒。（3600）秒。


- pip安装通过清华镜像：https://blog.csdn.net/furzoom/article/details/53897318
- 例如：
```
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow
pip3 install -i http://mirrors.aliyun.com/pypi/simple  --trusted-host mirrors.aliyun.com gensim
```
* 当ubuntu下apt相关安装命令过慢时，替换sources.list
https://blog.csdn.net/TotoroCyx/article/details/79517202

- problem1: I use with data_set argument. I got
FileNotFoundError: [Errno 2] No such file or directory: 'gem/intermediate/karate_gf.graph'
I couldn't see any gem/intermediate folder
    - solved: https://github.com/palash1992/GEM/issues/78

- Problem2:matplotlib 3.0: module 'matplotlib.cbook' has no attribute 'is_string_like' in 

    - solved: https://github.com/palash1992/GEM/issues/51
matplotlib版本过高，降低版本即可
pip install matplotlib==2.2.3.

- Problem3:_tkinter.TclError: no display name and no $DISPLAY environment variable

    - Solved:https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable You can solve it by adding these two lines in the VERY beginning of your .py script.
    ```
    import matplotlib
    matplotlib.use('Agg')
    ```
    - PS: The error will still exists if these two lines are not added in the very beginning of the source code.