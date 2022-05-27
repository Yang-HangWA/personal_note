#  SummaryNote
For myself note something 

## 基本用户添加
```
1.useradd usrname
2.passwd usrname
3.mkdir /home/usrname
4.chown usrname:usrname /home/usrname
5.chmod 0700 /home/usrname
6.vim /etc/passwd  改sh为bash
```
## 用户docker添加
```
usermod -aG docker usrname
```

##待使用并行
```
 def get_info_every_type(self, json_file, get_type=0):
        """
        :param paral_num:
        :return:id_map_feature_train = {item_id_1:train_feautre_1, item_id_2:train_feature_2...}
        """
        from pyspark import SparkConf, SparkContext
        lines = codecs.open(json_file, 'r', 'utf-8').readlines()
        conf = SparkConf().setAppName("App")
        conf = (conf.setMaster('local[*]').set('spark.executor.memory', '10G')
                .set('spark.driver.memory', '10G').set('spark.driver.maxResultSize', '10G'))
        sc = SparkContext(conf=conf)
        result = sc.parallelize(lines, self.paral_num).map(
            lambda line: self.transform_json_to_feature(line, get_type=0)).collect()
        sc.stop()
        train_feature, y_list = [], []
        for item in result:
            train_feature.append(item[0])
            y_list.append(item[1])
        return train_feature, y_list
```
