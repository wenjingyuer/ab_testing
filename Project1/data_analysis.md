## 推论统计分析

首先观察几组试验的点击率情况。

```python
import pandas as pd
import numpy as np
import statsmodels.stats.proportion as sp
from scipy.stats import norm

data = pd.read_csv("../data/output.csv")
print("对  照  组： ", data[data["dmp_id"] == 1]["label"].mean())
print("营销策略一： ", data[data["dmp_id"] == 2]["label"].mean())
print("营销策略二： ", data[data["dmp_id"] == 3]["label"].mean())
```

    对  照  组：  0.012551012429794775
    营销策略一：  0.015314747742072015
    营销策略二：  0.026191869198779274
    

##1.设计假设
可以看到策略一和策略二相较对照组在点击率上都有不同程度提升。

其中策略一提升0.2个百分点，策略二提升1.3个百分点，只有策略二满足了前面我们对点击率提升最小值（1%）的要求。

接下来需要进行假设检验，看策略二点击率的提升是否显著。

**零假设和备择假设**
记对照组点击率为p1，策略二点击率为p2，则：

零假设,策略二组相比于对照组没有优势， H0： p1 ≥ p2

备择假设，策略二组效果好于对照组 H1： p1 ＜ p2


## 2. 确定分布类型、检验类型、显著性水平和功效

样本服从二点分布，独立双样本，样本大小n＞30，总体均值和标准差未知，所以采用Z检验，显著性水平α取0.05，功效取0.8。

由H1得应该使用单尾检验


```python
# 用户数
n_old = data.query('dmp_id == 1').shape[0]  # 对照组
n_new = data.query('dmp_id == 3').shape[0]  # 策略二

# 点击数
c_old = data.query('dmp_id == 1 & label == 1').shape[0]
c_new = data.query('dmp_id == 3 & label == 1').shape[0]

# 计算点击率
r_old = c_old / n_old
r_new = c_new / n_new

# 总和点击率
r = (c_old + c_new) / (n_old + n_new)
```

## 3. 计算z值和p值
直接用python statsmodels包计算z值和p值。

`proportions_ztest([对照组分子,实验组分子],[对照组分母,实验组分母],alternative='smaller')`

备择假设中，对照组比率表现<实验组，则alternative='smaller'


```python
z_score, p = sp.proportions_ztest([c_old, c_new],[n_old, n_new], alternative = "smaller")
print("检验统计量z：",z_score,"，p值：", p)
```

    检验统计量z： -59.44168632985996 ，p值： 0.0
    

**|z| > 2.58,因此实验效果非常显著**

**p值约等于0，p ＜ α(0.05)**，拒绝原假设。策略二的点击率好于原方案



## 4. 根据cohen's d看效应如何

首先求出对照组和实验组的标准差和效应量Cohen's d


```python
#对照组标准差
std1=data[data.dmp_id==1].label.std()
# 实验组（策略二）标准差
std2=data[data.dmp_id==3].label.std()
#联合标准差
se=np.sqrt(((n_old-1)*std1**2+(n_new-1)*std2**2)/(n_old+n_new-2))
# 效应量Cohen's d
cohen = (r_new-r_old)/se
print("效应量Cohen's d:",cohen)
```

    效应量Cohen's d: 0.11423211767783437
    

随后计算MDE


```python
# 显著性水平α,对应z分位数
z_alpha = norm.ppf(0.05)

# 统计功效1-β,对应z分位数
z_beta = norm.ppf(0.2)

MDE = (np.abs(z_alpha)+np.abs(z_beta))*np.sqrt((std1**2)/n_old+(std2**2)/n_new)
print("MDE:",MDE)
```

    MDE: 0.0007341047609042006
    

**由于Cohen's d > MDE,所以2种方案间有差异，而且非常显著。**

## 5. 计算置信区间

最后计算2组方案点击率差异的置信区间


```python
CI_a = (r_new-r_old) - z_score*np.sqrt((std1**2)/n_old+(std2**2)/n_new)
CI_b = (r_new-r_old) + z_score*np.sqrt((std1**2)/n_old+(std2**2)/n_new)
print("置信区间:[%f,%f]"%(np.minimum(CI_a,CI_b),np.maximum(CI_a,CI_b)))
```

    置信区间:[-0.003909,0.031190]
    

## 总结

在两种营销策略中，策略二对广告点击率有显著提升效果，且相较于对照组点击率提升了近一倍，因而在两组营销策略中应选择第二组进行推广。


```python
print("检验统计量z：",z_score)
print("p值：", p)
print("置信区间:[%f,%f]"%(np.minimum(CI_a,CI_b),np.maximum(CI_a,CI_b)))
```

    检验统计量z： -59.44168632985996
    p值： 0.0
    置信区间:[-0.003909,0.031190]
    
