##零售商品分类
### 1.运行环境
* pytorch1.0.1
### 2.数据说明
数据主要包含三部分：data文件夹(train)、test文件夹(test)、data.csv(train信息)  
`.csv`

- 键值说明

|名称|类型|描述|
| --- | --- | --- |
|ImageName|string|图片文件名|
|CategoryId|int|商品分类 ID，取值范围 [1, 200]|

### 3. 文件说明
* dataset.py  
dataset.py改写pytorch中的DATASET类来读取数据
* model.py  
model.py为本次分类所采用的模型InceptionResNetV2
* rpi_define.py  
常量定义，resize图片大小以及分类数
* train.py   
对模型进行训练
* test 
对test数据进行测试