
### note


#### 添加docker用户
     - 1. 创建docker用户组
         ```
         sudo groupadd docker
         ```

     - 2. 应用用户加入docker用户组
        ```
        sudo usermod -aG docker ${USER}
        ```
        
    - 3. 重启docker服务
       ```
       sudo systemctl restart docker
       ```
       
#### conlab
    - 显卡显存不足，调小参数重新测试。
    - 拆分测试数据，分批导入测试。。
    
    
#### cproflie 分析函数时间
```
python -m cProfile -s cumulative cprofile_test.py
python -m cProfile -s cumulative predict.py --model_path xlnet_0_gector.th --input_file test.txt --output_file out --transformer_model xlnet --special_tokens_fix 0


python -m cProfile -o result.out -s cumulative step.py  //性能分析, 分析结果保存到 result.out 文件；
python gprof2dot.py -f pstats result.out | dot -Tpng -o result.png   //gprof2dot 将 result.out 转换为 dot 格式；再由 graphvix 转换为 png 图形格式。
```

#### Mysql 大坑：Mysql 无法连接
```
https://www.jianshu.com/p/d501af0f127c

https://segmentfault.com/a/1190000022319230
注意：不要映射：/etc/mysql/my.cnf，该文件是个软连。即使映射成功配置文件是不会生效的。
当前页可以新建一个配置文件映射到：/etc/mysql/conf.d/ 目录下

├─mysql5.7
│      ├─conf
│      │  └─mysql.conf.d
│      │          mysqld.cnf
│      └─scripts
│              run.sh

-v="$dirpath"/conf/mysql.conf.d/mysqld.cnf:/etc/mysql/mysql.conf.d/mysqld.cnf 
```

#### confluence迁移测试
    - 1.confluence数据迁出
      ``` 
      mysqldump confluence -uroot -p  > confluence.sql
      ```
    - 2.docker cp 数据到新的mysql容器
    - 3.合并现有特征测试

#### confluence 搭建相关
```
atlassian-extras-decoder-v2-3.2.jar  
#下载地址：
https://pan.baidu.com/s/1eRKDDOA 获取密码：mbjp
atlassian-universal-plugin-manager-plugin-2.22.jar
#下载地址：
https://pan.baidu.com/s/1o7Lfv6M 提取密码：1i3y
```
```
mv /opt/atlassian/confluence/confluence/WEB-INF/lib/atlassian-extras-decoder-v2-3.4.1.jar /mnt
mv /opt/atlassian/confluence/confluence/WEB-INF/atlassian-bundled-plugins/atlassian-universal-plugin-manager-plugin-4.2.6.jar /mnt

docker cp confluence/atlassian-extras-decoder-v2-3.2.jar confluence:/opt/atlassian/confluence/confluence/WEB-INF/lib/
docker cp confluence/atlassian-universal-plugin-manager-plugin-2.22.jar  confluence:/opt/atlassian/confluence/confluence/WEB-INF/atlassian-bundled-plugins/
```
#### 1. mysql使用5.7版本
```
mysql的配置文件需要修改，Confluence官方推荐MySQL配置如下：
character-set-server=utf8mb4
collation-server=utf8mb4_bin
default-storage-engine=INNODB
max_allowed_packet=256M
innodb_log_file_size=2GB
transaction-isolation=READ-COMMITTED
binlog_format=row
```

- 2.使用： https://wandouduoduo.github.io/articles/f9f96949.html 操作，下载破解包可以按照上面两个百度云盘地址下载

```
docker run -d --name confluence -p 8090:8090 -v /opt/data/hangyang/confluence:/var/atlassian/confluence --link mysql_v2 --user root:root cptactionhank/atlassian-confluence:latest
```
