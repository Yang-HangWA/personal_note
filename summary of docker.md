# Summary of docker

* 在docker镜像中加入环境变量
  - 1.使用docker run --env VARIABLE=VALUE image:tag直接添加变量，适用于直接用docker启动的项目
   ```
   root@ubuntu:/home/vickey/test_build# docker run --rm -it --env TEST=2 ubuntu:latest
   root@2bbe75e5d8c7:/# env |grep "TEST"
   ```
   - 2.使用dockerfile的ARG和ENV添加变量，适用于不能用docker run命令启动的项目，如k8sARG只在构建docker镜像时有效（dockerfile的RUN指令等），在镜像创建了并用该镜像启动容器后则无效（后面有例子验证）。但可以配合ENV指令使用使其在创建后的容器也可以生效。
     - `example`
    ```
    From *
    ARG RSYNC_PASSWORD
    ARG REMOTE_MODEL_PATH

    RUN apt-get install -y rsync tar
    ENV LOCAL_MODEL_PATH="/home/model"

    RUN mkdir -p ${LOCAL_MODEL_PATH} && cd ${LOCAL_MODEL_PATH} && rsync -avz rsync://******/ai_models/${REMOTE_MODEL_PATH}/* ./ && tar -zxvf *.tar.gz

    ENTRYPOINT ["sh","-c","sleep 1000000"]

    build.sh
    cd `dirname $0`

    cd ../../

    docker build \
    -t hub.lzhaohao.info/infer_platform_garbage_account \
    -f docker/garbage_account/Dockerfile \
    --build-arg RSYNC_PASSWORD="****" \
    --build-arg REMOTE_MODEL_PATH="garbage_account" \
    .

    #docker push hub.lzhaohao.info/infer_platform_garbage_account
    ```

* [docker核心技术和实现原理](https://draveness.me/docker)
* [docker -- 从入门到实践](https://yeasy.gitbook.io/docker_practice/)
* 在线安装docker 
```bash
    curl -fsSL get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
```

* 安装Docker Compose
```bash
    $ sudo curl -L https://github.com/docker/compose/releases/download/1.17.0/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose

    $ sudo chmod +x /usr/local/bin/docker-compose

    $ docker-compose --version # 安装正确能正常显示版本号
```