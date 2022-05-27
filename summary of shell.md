# Summary of shell

* `/dev/null `： 这个伪设备经常被称为黑洞（black hole），因此它的目的是忽略发送给它的一切数据。向它写数据时，它始终报告写操作时成功的：向它读取数据时，该设备总是返回没有数据。

* 磁盘剩余空间查询： df   -h     
* 查看某个目录下文件大小分布 du -sh ./*

* supervisorctl status & supervisorctl reload & supervisorctl update

* [secureCRT 字体颜色、文件夹和文件显示的颜色区别开解决办法](https://blog.csdn.net/qq_22122811/article/details/77978442)


### secureCRT设置
- 1.多窗口并排
    - window -> Tile Vertically

- 2.不要总在上面
    - view -> always on top 去掉勾

- 3.单行字数设置
    - Step 1: Options --> Global Options --> Terminal --> Appearance
        - 找到Maximum columns 设置每行的最大长度，推荐256
    - Step 2: Options --> Global Options --> Default Session --> Edit Default Settings
        - 然后是 Terminal --> Emulation
        - 找到Logical columns 虽然可以设置成255，我设成175，这样可以不用左右拖拉
- 4.不要每次都换两行设置，去掉line wrap or new line mode