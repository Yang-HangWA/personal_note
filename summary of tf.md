# Summary of tf 

* bug a: `{ “error”: “Serving signature name: ”serving_default“ not found in signature def” }`

* solve a: 
```
saved_model_cli show --dir saved_model/1/ --all
查看秘钥具体是什么,tf默认是 serving_default
signature_def[‘helloworld’]: 中的helloworld 就是秘钥
```
<br>

* 启动容器： 限制GPU 内存 jupyter
```
docker run -d --user root  -m 64g --memory-swap 64g -e GRANT_SUDO=yes --runtime=nvidia -it  -e NVIDIA_VISIBLE_DEVICES=1 -p 8800:8888 -p 6000:6006  -v /logs/yanghang02_workspace/data/:/tf -w /tf  tensorflow/tensorflow:1.15.0-gpu-py3-jupyter
```
* 注：jupyter网页关闭，后台进程存在，需要kill掉。另外可以点击jupter重启服务按钮释放资源。进程会重启，后台进程id变更。
<br>

* log等级设置
```
tensorflow中可以通过配置环境变量 'TF_CPP_MIN_LOG_LEVEL' 的值，控制tensorflow是否屏蔽通知信息、警告、报错等输出信息。

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

使用方法：

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

TF_CPP_MIN_LOG_LEVEL 取值 0 ： 0也是默认值，输出所有信息

TF_CPP_MIN_LOG_LEVEL 取值 1 ： 屏蔽通知信息

TF_CPP_MIN_LOG_LEVEL 取值 2 ： 屏蔽通知信息和警告信息

TF_CPP_MIN_LOG_LEVEL 取值 3 ： 屏蔽通知信息、警告信息和报错信息
```
<br>

* .ipynb 清理输出（下面例子可以删除掉安装的输出信息）
```
!sudo apt-get install -y xvfb ffmpeg
!pip install 'gym==0.10.11'
!pip install imageio
!pip install PILLOW
!pip install 'pyglet==1.3.2'
!pip install pyvirtualdisplay


!pip install dm-acme
!pip install dm-acme[reverb]
!pip install dm-acme[tf]
!pip install dm-acme[envs]


from IPython.display import clear_output
clear_output()
```
<br>

* tfrecord & estimator
  - [tfrecord制作和读取](https://zhuanlan.zhihu.com/p/128975161)
  - [estimator模型示例](https://zhuanlan.zhihu.com/p/129018863)

* [libstdc++.so.6: version `GLIBCXX_3.4.22' not found](https://github.com/lhelontra/tensorflow-on-arm/issues/13)

* [Error when checking model input: expected convolution2d_input_1 to have shape (None, 3, 32, 32) but got array with shape (50000, 32, 32, 3)](https://stackoverflow.com/questions/41771965/error-when-checking-model-input-expected-convolution2d-input-1-to-have-shape-n)