# 【A/B测试】支付宝营销策略效果分析

## 1. 数据来源
数据集来自阿里云天池：
[阿里云天池 - Audience Expansion Dataset](https://tianchi.aliyun.com/dataset/dataDetail?dataId=50893&lang=zh-cn)

该数据集包含三张表，分别记录了支付宝两组营销策略的活动情况：

+ emb_tb_2.csv: 用户特征数据集
+ effect_tb.csv: 广告点击情况数据集
+ seed_cand_tb.csv: 用户类型数据集
本分析报告主要使用广告点击情况数据，涉及字段如下：

+ dmp_id：营销策略编号（源数据文档未作说明，这里根据数据情况设定为1：对照组，2：营销策略一，3：营销策略二）
+ user_id：支付宝用户ID
+ label：用户当天是否点击活动广告（0：未点击，1：点击）


```python
import pandas as pd
import numpy as np
```

## 2. 数据处理

### 2.1 数据导入和清洗


```python
# 加载数据，自定义原始数据 header
data = pd.read_csv('https://ds-1300369208.cos.ap-shanghai.myqcloud.com/effect_tb.csv',header = None)
data.columns = ["dt","user_id","label","dmp_id"]

# 日志天数属性用不上，删除该列
data = data.drop(columns = "dt")
data.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>label</th>
      <th>dmp_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000004</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000004</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 描述性统计
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>label</th>
      <th>dmp_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.645958e+06</td>
      <td>2.645958e+06</td>
      <td>2.645958e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.112995e+06</td>
      <td>1.456297e-02</td>
      <td>1.395761e+00</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.828262e+06</td>
      <td>1.197952e-01</td>
      <td>6.920480e-01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.526772e+06</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.062184e+06</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.721132e+06</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.265402e+06</td>
      <td>1.000000e+00</td>
      <td>3.000000e+00</td>
    </tr>
  </tbody>
</table>
</div>



### 重复值处理


```python
data.shape
```




    (2645958, 3)



数据行数与独立用户数不统一，检查是否存在重复行：


```python
data[data.duplicated(keep = False)].sort_values(by = ["user_id"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>label</th>
      <th>dmp_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8529</th>
      <td>1027</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1485546</th>
      <td>1027</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1579415</th>
      <td>1471</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>127827</th>
      <td>1471</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>404862</th>
      <td>2468</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1382121</th>
      <td>6264633</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1382245</th>
      <td>6264940</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2575140</th>
      <td>6264940</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1382306</th>
      <td>6265082</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2575171</th>
      <td>6265082</td>
      <td>0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>25966 rows × 3 columns</p>
</div>




```python
data.nunique()
```




    user_id    2410683
    label            2
    dmp_id           3
    dtype: int64



删除重复行，并再次检查


```python
# drop duplicate
data = data.drop_duplicates()

# check if any duplicates left
data[data.duplicated(keep = False)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>label</th>
      <th>dmp_id</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



### 空值处理


```python
# check null values
data.info(show_counts = True)
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2632975 entries, 0 to 2645957
    Data columns (total 3 columns):
     #   Column   Non-Null Count    Dtype
    ---  ------   --------------    -----
     0   user_id  2632975 non-null  int64
     1   label    2632975 non-null  int64
     2   dmp_id   2632975 non-null  int64
    dtypes: int64(3)
    memory usage: 80.4 MB
    

数据集无空值，无需进行处理。

空值的处理流程：[pandas dataframe空值的处理方法](https://zhuanlan.zhihu.com/p/35321806)

### 异常值处理

通过透视表检查各属性字段是否存在不合理取值，同时查看每个分析组别的样本数量


```python
data.pivot_table(index = "dmp_id", columns = "label", values = "user_id",
                aggfunc = "count", margins = True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>label</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>dmp_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1881745</td>
      <td>23918</td>
      <td>1905663</td>
    </tr>
    <tr>
      <th>2</th>
      <td>404811</td>
      <td>6296</td>
      <td>411107</td>
    </tr>
    <tr>
      <th>3</th>
      <td>307923</td>
      <td>8282</td>
      <td>316205</td>
    </tr>
    <tr>
      <th>All</th>
      <td>2594479</td>
      <td>38496</td>
      <td>2632975</td>
    </tr>
  </tbody>
</table>
</div>



属性字段未发现异常取值，无需进行处理。

### 2.2 样本容量检验

在进行A/B测试前，需检查样本容量是否满足试验所需最小值。这里借助Evan Miller的样本量计算工具：[Sample Size Calculator](https://www.evanmiller.org/ab-testing/sample-size.html)

首先需要设定点击率基准线以及最小提升比例，我们将对照组的点击率设为基准线.


```python
# 计算对照组的点击率
data[data["dmp_id"] == 1]["label"].mean()
```




    0.012551012429794775



对照组点击率为1.25%，假定我们希望新的营销策略能让广告点击率至少提升1个百分点，则算得所需最小样本量为：2152。

[计算流程](https://www.evanmiller.org/ab-testing/sample-size.html#!1.25;80;5;1;0)

[A/B实验样本量计算原理](https://blog.csdn.net/weixin_41744624/article/details/109840263)


```python
data["dmp_id"].value_counts()
```




    1    1905663
    2     411107
    3     316205
    Name: dmp_id, dtype: int64



两组营销活动的样本量分别为41.11万和31.62万，满足最小样本量需求。


```python
# 保存清洗好的数据备用
data.to_csv("data/output.csv", index = False)
```
