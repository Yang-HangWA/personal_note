# others

- [md 公式生成](https://editor.codecogs.com/)

- [Sampling matters in deep embedding learning](arxiv.org/pdf/1706.0756) --- [知乎](https://zhuanlan.zhihu.com/p/27748177)
    - for 语音分类 or 人脸分类 
    - key word: distance weighted sampling   margin-based loss
    - [code](https://github.com/chaoyuaw/incubator-mxnet/tree/master/example/gluon/embedding_learning)
    - [语音mfcc](https://zhuanlan.zhihu.com/p/88625876)
 
- for embedding for device or others
    - [Faiss](https://cloud.tencent.com/developer/article/1077741) 
    - [Faiss wiki from github](https://github.com/facebookresearch/faiss/wiki)
    - [HashingVectorizer](https://zhuanlan.zhihu.com/p/33779124)

- [白话强化学习系列](https://zhuanlan.zhihu.com/c_1215667894253830144)
    - on-policy 与 off-policy区别
其实就是只有一句话: 更新值函数时是否只使用当前策略所产生的样本.
    - Q-learning, Deterministic policy gradient是Off-police算法,这是因为他们更新值函数时,不一定使用当前策略 产生的样本. 可以回想DQN算法,其包含一个replay memory.这个经验池中存储的是很多历史样本(包含的样本 ),而更新Q函数时的target用的样本是从这些样本中采样而来,因此,其并不一定使用当前策略的样本.

    - REINFORCE, TRPO, SARSA都是On-policy,这是因为他们更新值函数时,只能使用当前策略产生的样本.具体的,REINFORCE的梯度更新公式中 ,这里的R就是整个episode的累积奖赏,它用到的样本必然只是来自与
- ![reinforcement pic](https://github.com/Yang-HangWA/DailyNote/blob/master/pic/reinforcement.png)


-  谷歌公司在2003年到2004年公布了关于GFS、MapReduce和BigTable的3篇技术论文，成为后来云计算和Hadoop项目的重要基石。如今，谷歌在后Hadoop时代的新“三驾马车”——Caffeine、Dremel和Pregel，再一次影响着全球大数据技术的发展潮流。Caffeine主要为谷歌网络搜索引擎提供支持，使谷歌能够更迅速地添加新的链接（包括新闻报道以及博客文章等）到自身大规模的网站索引系统中。Dremel是一种可扩展的、交互式的实时查询系统，用于只读嵌套数据的分析。通过结合多级树状执行过程和列式数据结构，它能做到几秒内完成对万亿张表的聚合查询。系统可以扩展到成千上万的CPU上，满足谷歌上万用户操作PB级的数据，并且可以在2秒～3秒钟完成PB级别数据的查询。Pregel是一种基于BSP模型实现的并行图处理系统。为了解决大型图的分布式计算问题，Pregel搭建了一套可扩展的、有容错机制的平台，该平台提供了一套非常灵活的API，可以描述各种各样的图计算。Pregel作为分布式图计算的计算框架，主要用于图遍历、最短路径、PageRank计算等。

- [奇异值分解(SVD)原理与在降维中的应用](https://www.cnblogs.com/pinard/p/6251584.html)

- EM算法
    - 可以采用的求解方法是EM算法——将求解分为两步：第一步是假设我们知道各个高斯模型的参数（可以初始化一个，或者基于上一步迭代结果），去估计每个高斯模型的权值；第二步是基于估计的权值，回过头再去确定高斯模型的参数。重复这两个步骤，直到波动很小，近似达到极值（注意这里是个极值不是最值，EM算法会陷入局部最优）。

- [常见的六大聚类算法](https://blog.csdn.net/Katherine_hsr/article/details/79382249)  
- [外文]( https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68)
- [聚类算法总结](https://blog.csdn.net/u012050154/article/details/50524884)

- horovod issue
    - WARNING: One or more tensors were submitted to be reduced, gathered or broadcasted bysubset of ranks and are waiting for remainder of ranks for more than 60 seconds. This mayindicate that different ranks are trying to submit different tensors or that only subset of ranks issubmitting tensors, which will cause deadlock.
    - https://github.com/uber/horovod/issues/403
    - https://github.com/uber/horovod/issues/100
    - https://pastebin.com/tFWFGTgF

- [清华大学开源软件镜像站](https://mirrors.tuna.tsinghua.edu.cn)

- [Anaconda 官方源下载很慢，有没有好的解决办法](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)

- [python3 csv 空行解决](https://blog.csdn.net/chuan_yu_chuan/article/details/53671587)


### 机器视觉
* [[译] 必读的计算机视觉开创性论文](https://mp.weixin.qq.com/s/RBIr3Bf2vqTK4t3Q38vlcQ)

* pydoop安装
    - 1. 配置JAVA_HOME 及 HADOOP_HOME
    - 2. 设置HADOOP_VERSIONexport HADOOP_VERSION=2.7.2
    - 3. 设置CLASSPATH，需要导入hadoop hdfs所需的全部jarexport CLASSPATH=$($HADOOP_HOME/bin/hdfs classpath --glob)
    - 4. pip安装pydooppip install pydoop5. 若过程中出现，无Python.h文件时，安装python开发依赖python-develyum install python-devel （需要root权限）


### 使用示例
* 0. 导入module
```
import pydoop.hdfs as hdfs
```
* 1. 枚举目录文件
```
for i in hdfs.ls("/home/pydoopDemo"):
print i
```
* 2. 判断目录(/home/pydoopDemo/test)是否存在，若不存在，则创建
```
if not hdfs.path.exists("/home/pydoopDemo/test") :
    hdfs.mkdir("/home/pydoopDemo/test")
else :
    print("/home/pydoopDemo/test exists.")
```
* 3. 上传本地文件至hdfs
```
hdfs.put("/home/pydoopDemo/pydoopDemo.py", "/home/pydoopDemo/test/")
```
* 4. 下载hdfs文件至本地目录
```
hdfs.get("/home/pydoopDemo/hello.txt", "/home/pydoopDemo/")
```
* 5. 复制hdfs文件至hdfs路径
```
hdfs.cp("/home/pydoopDemo/hello.txt", "/home/pydoopDemo/test/")
```
* 6. 移动hdfs文件至hdfs路径
```
hdfs.move("/home/pydoopDemo/hello.txt", "/home/pydoopDemo/test/hello.txt.mv")
```
* 7. 写入hdfs文件
```
hdfs.dump("pydoop write Test", "/home/pydoopDemo/test/test.txt")
with hdfs.open("/home/pydoopDemo/test/test.txt", "w") as f:
f.write("test\n")
```
* 8. 读取hdfs文件
```
f=hdfs.open("/home/pydoopDemo/hello.txt")
读取指定长度内容，未指定，则输出全部内容
f.read()
读取行数据
f.readline()
跳至文件指定位置
f.seek(0)
关闭文件
f.close()
```
* 9. 删除hdfs文件
```
hdfs.rmr("/home/pydoopDemp/test")10. hdfs文件信息获取

hdfs文件的全路径
hdfs.path.abspath("/home/pydoopDemo/test")
hdfs文件所在目录
hdfs.path.dirname("/home/pydoopDemo/test")
```

- [1ADAC驱动](hav.update.sony.net/MDR/drivers/Driver_v2.23.0.exe)

- [播放DSD软件](hav.update.sony.net/APPS/Hi-ResAudioPlayer/Win/Hi-ResAudioPlayer_1.2.0.exe)
<br>

- jetbrain 系列激活 <= 2019.1.3 的版本都能用
   - 直接整个压缩包拖进去就行 jetbrain 系列的都可以用  
   - [jetbrains-agent-latest.zip](https://github.com/Yang-HangWA/DailyNote/blob/master/file/jetbrains-agent-latest.zip)
<br>

- windows 搜索神器 [Everything.exe](https://github.com/Yang-HangWA/DailyNote/blob/master/file/Everything64_1.4.1.895.exe)
* `pip freeze | tee requirements.txt` # 输出本地包环境至文件
* `pip install -r requirements.txt` # 根据文件进行包安装
<br>
### pycharm
- shift+ctrl+F10 运行当前文件
- python环境添加包： 
    - 第一步： setting
    - 第二步：Project Interpreter 

### RANK

- learning to rank / RankNet / LambdaRank

    - Learning to Rank就是一类目前最常用的，通过机器学习实现步骤②的算法。它主要包含单文档方法（pointwise）、文档对方法（pairwise）和文档列表（listwise）三种类型。pointwise单文档方法顾名思义：对于某一个query，它将每个doc分别判断与这个query的相关程度，由此将docs排序问题转化为了分类（比如相关、不相关）或回归问题（相关程度越大，回归函数的值越大）。但是pointwise方法只将query与单个doc建模，建模时未将其他docs作为特征进行学习，也就无法考虑到不同docs之间的顺序关系。而排序学习的目的主要是对搜索结果中的docs根据用户点击的可能性概率大小进行排序，所以pointwise势必存在一些缺陷。


- 向量相似性搜索
    - ann benchmarks :    
        - http://github.com/erikbern/ann-benchmarks/
	- 1. [HNSW](https://zhuanlan.zhihu.com/p/80552211)
	- 2. [SCANN](https://zhuanlan.zhihu.com/p/164971599)

- [ nlp中的词向量对比：word2vec/glove/fastText/elmo ... - 知乎专栏 ](https://zhuanlan.zhihu.com/p/56382372)

### NGS
```

ngs model 2017-11-04
'''
kmer(k=35) 预测第36个位置
input_sequence_lenth:35
sub_kmer(seq_length=k=5): seq_num=7 sub_kmer
embedding_dim=128
rnn_input_shape:(sampe,7,128)

'''

import numpy as np
from pywt import wavedec
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from keras.layers.merge import concatenate
from keras.layers import Dense, Activation, LSTM, merge
from keras.layers.core import Reshape
from keras.engine.topology import *
#from keras.layers import containers
from keras import layers
import keras
import os
from test import *
CURRENT_PATH = os.path.split(os.path.realpath(__file__))[0] + "\\"
dic_kmer = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
seq_length = 5
seq_num = 7
n_classes = 4
file = open(os.path.join(CURRENT_PATH,"result_record.txt"),"a+")
file.write("test!!")
file.close()
# 把数据集划分为训练数据和测试数据x_train, x_test, y_train, y_test
def train_test_split(train, label, train_size=0.8):
np.random.seed(42)
n = len(train)
idx = np.arange(n)
np.random.shuffle(idx)
train = train[idx]
label = label[idx]
m = int(n * train_size)
return (train[:m], train[m:], label[:m], label[m:])


# transform string kmer into a integer ,ensure that k<10
def kmer2int(kmer):
r = 0
for ch in kmer:
r = r * n_classes + dic_kmer[ch]
return r


# def map_fun(number):
# s4 = number%4
# s3 = ((number-s4)/4)%4
# s2 = ((number - s3) / 4) % 4
# s1 = ((number - s2) / 4) % 4
def gen_data(path):
dataX = []
dataY = []
for line in open(path):
dataX2 = []
n_chars = len(line.strip())
for i in range(n_chars - seq_length):
kmer = line[i:i + seq_length]
dataX2.append(kmer2int(kmer))
for i in range(seq_length * seq_num, n_chars):
dataY.append(dic_kmer[line[i]])
for i in range(n_chars - seq_length * seq_num):
list_x = dataX2[i:i + seq_length * seq_num:seq_length]
dataX.append(list_x)
dataX = np.asarray(dataX)
dataY = np.asarray(dataY)
dataY = keras.utils.to_categorical(dataY, n_classes)
return train_test_split(dataX, dataY)

def Net_Graph():
input_1 = keras.layers.Input(shape=(seq_num,))
x = keras.layers.Embedding(input_dim=n_classes**seq_num, output_dim=128, input_length=seq_num)(input_1)
#print("x", x)
model_1 = Dense(128)(x)
model_2 = Dense(128)(x)
model_3 = Dense(128)(x)
model_4 = Dense(128)(x)
model_5 = Dense(128)(x)
model_6 = Dense(128)(x)
model_7 = Dense(128)(x)
#print("model:", model_1)
reshape = Reshape(target_shape=(128,7,))
combined = merge([reshape(model_1), reshape(model_2), reshape(model_3), reshape(model_4), reshape(model_5), reshape(model_6), reshape(model_7)], mode='concat')
lstm_1 = keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.3, return_sequences=True)(combined)
lstm_2 = keras.layers.LSTM(64, dropout=0.3, recurrent_dropout=0.2)(lstm_1)
dense_1 = keras.layers.Dense(64, activation="relu")(lstm_2)
dropout_1 = keras.layers.Dropout(0.5)(dense_1)
output = keras.layers.Dense(n_classes, activation="softmax")(dropout_1)
model = keras.models.Model(input_1, output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
to_file_dir = os.path.join(CURRENT_PATH, "sub_section.png")
keras.utils.plot_model(model, to_file=to_file_dir, show_shapes=True)

path = os.path.join(CURRENT_PATH, "chr22.1024.fa")
x_train, x_test, y_train, y_test = gen_data(path)
print(x_train.shape)
print(x_test.shape)
batch_value = 64
file = open(os.path.join(CURRENT_PATH, "result_record_one.txt"), "a+")
epochs_size = 100
print(batch_value, epochs_size)
str_s = "batch_value,epochs_size:" + str(batch_value) + " " + str(epochs_size) + "\n"
file.write(str_s)
model.fit(x_train, y_train, batch_size=batch_value, epochs=epochs_size, shuffle=True, verbose=2)
loss1, accuracy1 = model.evaluate(x_test, y_test)
loss2, accuracy2 = model.evaluate(x_train, y_train)
print('\n')
print("what1:", loss1, accuracy1)
print("what2", loss2, accuracy2)
title = "input_dim = 4"
file.write(title + "\n")
file.write("loss1,accuracy1:" + str(loss1) + " " + str(accuracy1) + "\n")
file.write("loss2,accuracy2:" + str(loss2) + " " + str(accuracy2) + "\n")
file.close()

Net_Graph()
```

### PHP动态调整
```
#! env bash
set -e
set +o noglob
WORKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
conf_dir="$WORKDIR/dynamic_consumer.conf"
#日志存放的文件夹
LOG_DIR="/data/logs"
#log存放路径
log_file="$LOG_DIR/dynamic_adjust_eventhandle.log"
log_max_size=$((300*1024*1024))
function log()
{
    [ ! -d $LOG_DIR ] && mkdir -p $LOG_DIR
    echo "`date` $1" >> $log_file
}

#检查配置文件：若果配置文件不存在，则sleep 10，退出
if [ ! -f $conf_dir ];then
   log "The file $conf_dir is not exist!"
   sleep 10
   exit
fi

#让配置文件生效
source $conf_dir

#赋予disque客户端权限
chmod 777 $DISQUE

#检查配置文件中的参数是否设置正常,数值型赋值为默认值，路径检查不存在则退出程序
if [[ $M_P -lt 0 || ! $M_P ]];then
    log "The value of M_P is not correct,exec:M_P=3!"
    M_P=3
fi

if [[ $N_P -lt 0 || ! $N_P ]];then
    log "The value of N_P is not correct,exec:N_P=2!"
    N_P=2
fi

if [[ $SPEED_EVE -lt 0 || ! $SPEED_EVE ]];then
    SPEED_EVE=200
    log "The value of SPEED_EVE is not correct,exec:SPEED_EVE=${SPEED_EVE}!"
fi  

if [[ $SPEED_NGX -lt 0 || ! $SPEED_NGX ]];then
    SPEED_NGX=200
    log "The value of SPEED_NGX is not correct,exec:SPEED_NGX=${SPEED_NGX}!"
fi

if [[ $TIME_EVE -lt 0 || ! $TIME_EVE ]];then
    TIME_EVE=30
    log "The value of TIME_EVE is not correct,exec:TIME_EVE=${TIME_EVE}!"
fi

if [[ $TIME_NGX -lt 0 || ! $TIME_NGX ]];then
    TIME_NGX=180
    log "The value of TIME_NGX is not correct,exec:TIME_NGX=${TIME_NGX}!"
fi

if [[ ! -f $PROC_CONF_PATH_EVE || ! $PROC_CONF_PATH_EVE ]];then
    log "The file ${PROC_CONF_PATH_EVE} is not exist,the procs will exit after 10s!"
    sleep 10
    exit
fi   

if [[ ! -f $PROC_CONF_PATH_NGX || ! $PROC_CONF_PATH_NGX ]];then
    log "The file ${PROC_CONF_PATH_NGX} is not exist,the procs will exit after 10s!"
    sleep 10
    exit
fi

if [[ ! -f $DISQUE || ! $DISQUE ]];then
    log "The file disque client is not exist,the procs will exit after 10s!"
    sleep 10
    exit
fi

if [[ ! -f $DISQUE_CONF || ! $DISQUE_CONF ]];then
    log "The php configure file .env is not exist,the procs will exit after 10s!"
    sleep 10
    exit
fi

if [[ $INIT_STATUS -ne 0 && $INIT_STATUS -ne 1 ]];then
    log "The value of INIT_STATUS is not correct,,the procs will exit after 10s!"
    sleep 10
    exit
fi

function adjust_conf()
{
    local eventhandle="/etc/supervisord.d/eventhandle.ini"
    local ngxmsgconsume="/etc/supervisord.d/ngxmsgconsume.ini"
    local dyn_eventhandle="/etc/supervisord.d/dyn_eventhandle.ini"
    local dyn_ngxmsgconsume="/etc/supervisord.d/dyn_ngxmsgconsume.ini"
    local dynamic_consumer_conf="/data/scripts/dynamic_consumer.conf"
    local org_procs=$(cat /proc/cpuinfo |grep processor | wc -l)
    procs=$((org_procs/2))
    max_procs=$((org_procs*3/2))
    if [ -f "$dynamic_consumer_conf" ]
    then
        source "$dynamic_consumer_conf"
        max_procs_dyn=$((org_procs*M_P/N_P))
        if [ $max_procs_dyn -lt 2 ]
        then
            max_procs_dyn=$max_procs
        fi
        max_procs=$max_procs_dyn
    fi
    log "org_procs: $org_procs, procs: $procs, max_procs: $max_procs"
    if [ $procs -gt 2 ]; then
        if [ -e "$eventhandle" ]
        then
            log "change $eventhandle"
            sed -i -e "s/numprocs=.*/numprocs=$procs/g" "$eventhandle"
        fi
        if [ -e "$ngxmsgconsume" ]
        then
            log "change $ngxmsgconsume"
            sed -i -e "s/numprocs=.*/numprocs=$procs/g" "$ngxmsgconsume"
        fi
    fi
    dyn_procs=$((max_procs-procs))
    if [ -e "$dyn_eventhandle" ]
    then
        log "change $dyn_eventhandle"
        sed -i -e "s/numprocs=.*/numprocs=$dyn_procs/g" "$dyn_eventhandle"
    fi
    if [ -e "$dyn_ngxmsgconsume" ]
    then
        log "change $dyn_ngxmsgconsume"
        sed -i -e "s/numprocs=.*/numprocs=$dyn_procs/g" "$dyn_ngxmsgconsume"
    fi
}

#进行初始化
if [ $INIT_STATUS -eq 0 ];then
   adjust_conf
   supervisorctl reread
   supervisorctl update
   sed -i -e "s/INIT_STATUS=.*/INIT_STATUS=1/g" "$conf_dir"
   log "init done!"
fi


#单独求某一个队列的长度
function queryforlen()
{
  echo $($1 -h $2 qlen $3)
}

# 获取当前消息队列长度,参数是channel,2018/9/28将queue的长度获取调整为三台机获取的qlen长度之和
function getQLen()
{
    local test_str=$(cat $DISQUE_CONF | grep "DISQUE_HOST")
    local port_str=$(cat $DISQUE_CONF | grep "DISQUE_PORT")
    local disque_port=${port_str#*=}
    local ip_str=${test_str#*=}
    local disque_ip_arr=(${ip_str//;/ })
    local total=0
    for ip in ${disque_ip_arr[@]}
    do  #ping不通的ip执行循环会超过20秒
    log "function getQLen():Try to connect to ${ip} to get disque info!"
    (echo > /dev/tcp/${ip}/$disque_port) >/dev/null 2>&1
    result=$?
    local queue_total=0
    if [[ $result -eq 0 ]];then
        log "function getQLen():Success to connect to ${ip}!!!"
        local queue_str=$1
        local queue_arr=${queue_str//;/}
        for queue in ${queue_arr[@]}
        do
            queue_total=$[$(queryforlen $DISQUE ${ip} $queue)+$queue_total]
        done
    fi
    total=$[total+queue_total]
    done    
    echo $total
}

# 获取CPU内核数
function getCPUCores()
{
    local procs=$(cat /proc/cpuinfo |grep processor | wc -l)
    echo "$procs"
}

#获取初始进程数，为CPU数的1/2
function getInitProcsNum()
{
    local procs=$[ ($(getCPUCores)/2) ]
    if [ $procs -le 2 ];then
        echo "2"
    else
       echo "$procs"  
    fi
}
#获取进程上限值为cpu的1.5倍
function getUplimitNum()
{
    local uplimit_num=$((($(getCPUCores)*$M_P)/$N_P-$init_num))
    #log "function getUplimitNum():The uplimit number of procs is:$uplimit_num"
    echo "$uplimit_num"
}

#获取当前有多少消费进程,参数是进程名
function getConsumers()
{
    local cur_proc_num=$(supervisorctl status | grep $1* | grep RUNNING | wc -l  )
    echo "$cur_proc_num"
}

# 计算阈值，需要消费速度，初始进程数，和可堆积时间  $1 SPEED $2 TIME
function calcThreshold()
{   
    local speed=$1
    local t=$2
    local result=$[speed*init_num*t]
    echo "$result"
}



function needModify()
{
    local speed=$1
    local t=$2
    local cur_queue_len=$(getQLen $3)
    local cpu_uplimit_num=$(getUplimitNum)
    local init_procs_num=$init_num
    local cur_procs=$(getConsumers $4)
    local min_adjust_len=$(calcThreshold $1 $2)
    log "function needModify():cur_queue_len:$cur_queue_len,init_procs_num:$init_procs_num,cpu_uplimit_num:$cpu_uplimit_num,cur_procs:$cur_procs,min_adjust_len:$min_adjust_len"
    if [[ "$cur_queue_len" -gt "$min_adjust_len" ]];then
        local need_incs_procs=$[(cur_queue_len-min_adjust_len)/speed/t]
        local need_consume=$[cur_queue_len-min_adjust_len]
        #判断取ceil值
        local consume_queue_len=$[need_incs_procs*t*speed]
        if [[ $consume_queue_len -lt $need_consume ]];then
            need_incs_procs=$[need_incs_procs+1]
        fi

        if [ $need_incs_procs -gt $cpu_uplimit_num ]
        then
            need_incs_procs=$cpu_uplimit_num
        fi
        echo "$need_incs_procs"
    else
        echo "0"
    fi
}
#修改进程数，$1为最终进程数，$2为配置文件位置
function modifyProcs()
{
    sed -i -e "s/numprocs=.*/numprocs=$1/g" "$2"
    log "numprocs change to $1 in $2"
}

function consume_process_running()
{
    local label=$(supervisorctl status | grep $1 | grep RUNNING | wc -l)
    if [ "$label" -ge "1" ];then
        echo "1"
    else
        echo "0"
    fi
}

#检查并确保启动的进程数达标，参数为需要重启的进程数组和进程数目
function restartProcs()
{
    local start_time=$(date +%s)
    local arr=$1
    local number=$2

    local consumer=$(getConsumers $3)

    while [ $consumer -lt $number ]
    do
        for item in ${arr[@]}
        do
           local label=$(consume_process_running $item)
           if [ "$label" -eq "0" ];then
                log "supervisorctl start ${item}"
                supervisorctl start ${item}
                local pid=`supervisorctl pid ${item}`
                if [ "$pid" -gt "0" ]
                then
                    echo "start ${item} succ..."
                else
                    echo "start ${item} failed..."
                    continue
                fi
            fi
            
            consumer=$(getConsumers $3)
            if [ $consumer -ge $number ]
            then
                log "$consumer is greater or equal $number, break"
                break 2
            fi
        done
        local use_time=$(checktime $start_time)
        if [[ $use_time -gt 20 ]];then
            log "Start action takes 20 seconds,there is something wrong with the restart!"
            break
        fi
    done
}

#参数为name
function getProcArr()
{
    local name=$1
    local proc_num=$(getUplimitNum)
    local proc_name_arr=()
    local count=2
    proc_name_arr[1]=${name}":"${name}"_00"
    local end_tag=$[proc_num-1]
    if [[ $end_tag -ge 1 ]];then
        for i in $(seq 1 $end_tag)
        do
            if [[ ${i} -lt 10 ]];then
                local final_name=${name}":"${name}"_0"${i}
            else
                local final_name=${name}":"${name}"_"${i}
            fi    
            proc_name_arr[$count]=$final_name
            count=$[count+1]        
       done
    fi
    echo "${proc_name_arr[@]}"
}


function checktime()
{
  local END=$(date +%s)
  local DIFF=$(($END-$1))
  echo "$DIFF"
}

init_num=$(getInitProcsNum)
dyn_eve_proc_arr=$(getProcArr "dyn_eventhandle")
dyn_ngx_proc_arr=$(getProcArr "dyn_ngxmsgconsume")
count=0
start_eve=$(date +%s)
start_ngx=$(date +%s)
diff_eve=0
diff_ngx=0
while [ $count -lt 600 ]
do
    count=$[count+1]

    #evethandle监控模块
    if [[ $diff_eve -ge $TIME_EVE || "$diff_eve" -eq "0"  ]];then
       log "------Start to check eventhandle!!!------"
       chang_num_eve=$(needModify $SPEED_EVE $TIME_EVE $DISQUE_CHANNEL_EVE $PROC_NAME_EVE)
       log "main:  The change number is $chang_num_eve!"
        if [[ $chang_num_eve -gt 0 ]] ;then
            restartProcs "${dyn_eve_proc_arr[*]}" $chang_num_eve $PROC_NAME_EVE
        fi  
        start_eve=$(date +%s)
        log "------End of the check eventhandle!!!------"
    fi
    #ngxmsgconsume监控模块
    if [[ $diff_ngx -ge $TIME_NGX || "$diff_ngx" -eq 0 ]];then
       log "------Start to check ngxmsgconsume!!!------"
       chang_num_ngx=$(needModify $SPEED_NGX $TIME_NGX $DISQUE_CHANNEL_NGX $PROC_NAME_NGX)
       log "main:  The change number is $chang_num_ngx!"
       if [[ $chang_num_ngx -ne 0 ]] ;then
            restartProcs "${dyn_ngx_proc_arr[*]}" $chang_num_ngx $PROC_NAME_NGX
        fi  
        log "------End of the check ngxmsgconsume!!!------"
        start_ngx=$(date +%s)
        log_size=`ls -l $log_file | awk '{print $5}'`
        if [ $log_size -gt $log_max_size ];then
            rm -f $log_file
        fi
    fi
    
    sleep 5

    diff_eve=$(checktime $start_eve)
    diff_ngx=$(checktime $start_ngx)
    
done
```

```
文件：train_by_csv_horovod.py
import sys,os  
sys.path.append(os.getcwd())
from os.path import join
import argparse
import warnings
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from malconv import Malconv_3
import time
import datetime
import horovod.keras as hvd
from keras import optimizers
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
os.environ["HADOOP_VERSION"] = "2.7.2"
import pydoop.hdfs as hdfs
warnings.filterwarnings("ignore")

hvd.init()

parser = argparse.ArgumentParser(description='word2vec_cnn classifier training')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--limit', type=float, default=0., help="limit gpy memory percentage")
parser.add_argument('--max_len', type=int, default=200000, help="model input legnth")
#parser.add_argument('--val_size', type=float, default=0.1, help="validation percentage")
parser.add_argument('--save_path', type=str, default='./data/', help='Directory to save model and log')
parser.add_argument('--model_path', type=str, default='./saved/malconv.h5', help="model to resume")
parser.add_argument('--resume',type=int,default=1,help="Retrain or continue last model parm")
#--csv is training data path,--valid_csv is test data path
parser.add_argument('--csv',type=str,default='hdfs://HACluster/zhuhai/pefile/data1/csv_train_data/train_data_1.csv',help="The csv file for training!")
parser.add_argument('--valid_csv',type=str,default='hdfs://HACluster/zhuhai/pefile/data1/csv_test_data/test_0-56.csv',help="The csv file for testing!")
parser.add_argument('--train_number',type=int,default=100,help="To set the number  for training!")
parser.add_argument('--test_number',type=int,default=100,help="To set the number for testing!")

args = parser.parse_args()

#Avoid receiving warning message
TF_CPP_MIN_LOG_LEVEL=2

from keras import backend as K
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

#make the list str to int list
def str_to_intlist(data):
    list_str = []
    num_list = data.split('[')[1].split(']')[0].split(',')
    for num in num_list:
        list_str.append(int(num))
    return list_str

#To generator data for training or testing!
def data_generator(reader, max_len=200000, batch_size=64, shuffle =True):
    with tf.device('/cpu:0'):
        while True:
            chunks = []
            chunk = reader.get_chunk(batch_size)
            chunks.append(chunk)
            df = pd.concat(chunks,ignore_index=True)
            label,data = df[1].values,df[2].values
            data_xx = []
            data_yy = []
            length = len(label)
            idx = np.arange(length)
            np.random.shuffle(idx)
            try:
                for i in idx:
                    xx = str_to_intlist(data[i])
                    data_xx.append(xx)
                    data_yy.append(label[i])
                xx = pad_sequences(data_xx, maxlen=max_len, padding='post', truncating='post')
                yy = data_yy
            except:
                print("get xx or yy error!!!!!")
            yield (xx, yy)

if __name__ == '__main__':
    save_best, verbose = True,True
    batch_size, max_len, save_path, epochs = args.batch_size, args.max_len, args.save_path, args.epochs
    
    if args.resume==1:
        model = load_model(args.model_path)
        print("=============load model!===============")
    else:
        model = Malconv_3(args.max_len)
        opt = keras.optimizers.Adadelta(1.0 * hvd.size())
        opt = hvd.DistributedOptimizer(opt)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
        print("=============compile model!==============")
    #model message
    model.summary()
    #get training data
    #with tf.device('/cpu:0'):
    f = hdfs.open(args.csv)
    row_number = args.train_number
    reader = pd.read_csv(f,header=None,sep=',',iterator=True,nrows=row_number)
    #get testing data
    valid_f = hdfs.open(args.valid_csv)
    valid_number = args.test_number
    valid_reader = pd.read_csv(valid_f,header=None,sep=',',iterator=True,nrows=valid_number)

    callbacks_list= [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),
    keras.callbacks.ReduceLROnPlateau(patience=5, verbose=1),
    ]   

    model_save = os.path.join(save_path,'malconv.h5')
    #if hvd.rank() == 0:
    with tf.device('/cpu:0'):
       callbacks_list.append(keras.callbacks.ModelCheckpoint(model_save))
    
    model.fit_generator(
        data_generator(reader, max_len, batch_size, shuffle=True),
        steps_per_epoch=(row_number//batch_size + 1)//hvd.size(),
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks_list,
        validation_data=data_generator(valid_reader, max_len, batch_size),
        validation_steps=(valid_number//batch_size + 1)//hvd.size())

```