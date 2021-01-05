# tftools

## 代码结构

--FCGen          数据读取模块\<br>  
----FCGen.py       数据读取接口函数\<br>  
----Generator.py   提供各种数据类型的处理（例如ID特征hash分桶、embeding等）\<br>  
--Trainer        模型训练模块\<br>  
----Evaluation.py  模型评估方法封装\<br>  
----InputFn.py     模型输入方法封装\<br>  
----Training.py    模型训练方法封装\<br>  

## 使用原理
特征的配置和特征间的组合在底层委托给[tf.feature_column](https://www.tensorflow.org/api_docs/python/tf/feature_column)来完成，这个模块用起来非常方便，和tfrecords格式的训练样本也很好配合。
参数配置的字段基本对应于不同类型feature_column的参数，以下是样例中的特征配置文件`Projects/Example/feature_spec.ini`



