```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

```

# Data Set Up
Welcome to my final project assignment for COGS118A. 



```python
bank_df = pd.read_csv("Bank/bank-full.csv", sep = ";")
display(bank_df)
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
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>2143</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>261</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>technician</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>29</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>151</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33</td>
      <td>entrepreneur</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2</td>
      <td>yes</td>
      <td>yes</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>76</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>unknown</td>
      <td>no</td>
      <td>1506</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>92</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>unknown</td>
      <td>single</td>
      <td>unknown</td>
      <td>no</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>198</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>45206</th>
      <td>51</td>
      <td>technician</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>825</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>17</td>
      <td>nov</td>
      <td>977</td>
      <td>3</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>45207</th>
      <td>71</td>
      <td>retired</td>
      <td>divorced</td>
      <td>primary</td>
      <td>no</td>
      <td>1729</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>17</td>
      <td>nov</td>
      <td>456</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>45208</th>
      <td>72</td>
      <td>retired</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>5715</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>17</td>
      <td>nov</td>
      <td>1127</td>
      <td>5</td>
      <td>184</td>
      <td>3</td>
      <td>success</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>45209</th>
      <td>57</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>668</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>17</td>
      <td>nov</td>
      <td>508</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>45210</th>
      <td>37</td>
      <td>entrepreneur</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2971</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>17</td>
      <td>nov</td>
      <td>361</td>
      <td>2</td>
      <td>188</td>
      <td>11</td>
      <td>other</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
<p>45211 rows × 17 columns</p>
</div>



```python
mushroom_df = pd.read_csv("Mushroom/agaricus-lepiota.data")
display(mushroom_df)
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
      <th>p</th>
      <th>x</th>
      <th>s</th>
      <th>n</th>
      <th>t</th>
      <th>p.1</th>
      <th>f</th>
      <th>c</th>
      <th>n.1</th>
      <th>k</th>
      <th>...</th>
      <th>s.2</th>
      <th>w</th>
      <th>w.1</th>
      <th>p.2</th>
      <th>w.2</th>
      <th>o</th>
      <th>p.3</th>
      <th>k.1</th>
      <th>s.3</th>
      <th>u</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>y</td>
      <td>t</td>
      <td>a</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>g</td>
    </tr>
    <tr>
      <th>1</th>
      <td>e</td>
      <td>b</td>
      <td>s</td>
      <td>w</td>
      <td>t</td>
      <td>l</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>m</td>
    </tr>
    <tr>
      <th>2</th>
      <td>p</td>
      <td>x</td>
      <td>y</td>
      <td>w</td>
      <td>t</td>
      <td>p</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <th>3</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>g</td>
      <td>f</td>
      <td>n</td>
      <td>f</td>
      <td>w</td>
      <td>b</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>e</td>
      <td>n</td>
      <td>a</td>
      <td>g</td>
    </tr>
    <tr>
      <th>4</th>
      <td>e</td>
      <td>x</td>
      <td>y</td>
      <td>y</td>
      <td>t</td>
      <td>a</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>n</td>
      <td>g</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8118</th>
      <td>e</td>
      <td>k</td>
      <td>s</td>
      <td>n</td>
      <td>f</td>
      <td>n</td>
      <td>a</td>
      <td>c</td>
      <td>b</td>
      <td>y</td>
      <td>...</td>
      <td>s</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>b</td>
      <td>c</td>
      <td>l</td>
    </tr>
    <tr>
      <th>8119</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>n</td>
      <td>f</td>
      <td>n</td>
      <td>a</td>
      <td>c</td>
      <td>b</td>
      <td>y</td>
      <td>...</td>
      <td>s</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>o</td>
      <td>p</td>
      <td>b</td>
      <td>v</td>
      <td>l</td>
    </tr>
    <tr>
      <th>8120</th>
      <td>e</td>
      <td>f</td>
      <td>s</td>
      <td>n</td>
      <td>f</td>
      <td>n</td>
      <td>a</td>
      <td>c</td>
      <td>b</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>b</td>
      <td>c</td>
      <td>l</td>
    </tr>
    <tr>
      <th>8121</th>
      <td>p</td>
      <td>k</td>
      <td>y</td>
      <td>n</td>
      <td>f</td>
      <td>y</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>b</td>
      <td>...</td>
      <td>k</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>e</td>
      <td>w</td>
      <td>v</td>
      <td>l</td>
    </tr>
    <tr>
      <th>8122</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>n</td>
      <td>f</td>
      <td>n</td>
      <td>a</td>
      <td>c</td>
      <td>b</td>
      <td>y</td>
      <td>...</td>
      <td>s</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>o</td>
      <td>c</td>
      <td>l</td>
    </tr>
  </tbody>
</table>
<p>8123 rows × 23 columns</p>
</div>



```python
income_df = pd.read_csv("Income/adult.data", delimiter=",")
display(income_df)
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
      <th>39</th>
      <th>State-gov</th>
      <th>77516</th>
      <th>Bachelors</th>
      <th>13</th>
      <th>Never-married</th>
      <th>Adm-clerical</th>
      <th>Not-in-family</th>
      <th>White</th>
      <th>Male</th>
      <th>2174</th>
      <th>0</th>
      <th>40</th>
      <th>United-States</th>
      <th>&lt;=50K</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37</td>
      <td>Private</td>
      <td>284582</td>
      <td>Masters</td>
      <td>14</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>32555</th>
      <td>27</td>
      <td>Private</td>
      <td>257302</td>
      <td>Assoc-acdm</td>
      <td>12</td>
      <td>Married-civ-spouse</td>
      <td>Tech-support</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32556</th>
      <td>40</td>
      <td>Private</td>
      <td>154374</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>32557</th>
      <td>58</td>
      <td>Private</td>
      <td>151910</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>Adm-clerical</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32558</th>
      <td>22</td>
      <td>Private</td>
      <td>201490</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32559</th>
      <td>52</td>
      <td>Self-emp-inc</td>
      <td>287927</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>15024</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
  </tbody>
</table>
<p>32560 rows × 15 columns</p>
</div>


###  One-Hot Encoding 

While our data looks nice and formatted, we sadly cannot work with categorical labels to perform our model training. To resolve this, we utilize one-hot encoding to transform the set of each possible nominal values from every column into a new column span that treats each label as a 0 or 1. For other columns with numerical values, we simply treat their values the same. Thankfully, numpy comes with a method to do this for us automatically for each of our datasets


```python
bank_encoded = pd.get_dummies(bank_df)
bank_encoded = bank_encoded.replace({True: 1, False: 0})

mushroom_encoded = pd.get_dummies(mushroom_df)
mushroom_encoded = mushroom_encoded.replace({True: 1, False: 0})

income_encoded = pd.get_dummies(income_df)
income_encoded = income_encoded.replace({True: 1, False: 0})

```


```python
display(bank_encoded)
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
      <th>age</th>
      <th>balance</th>
      <th>day</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>job_admin.</th>
      <th>job_blue-collar</th>
      <th>job_entrepreneur</th>
      <th>...</th>
      <th>month_may</th>
      <th>month_nov</th>
      <th>month_oct</th>
      <th>month_sep</th>
      <th>poutcome_failure</th>
      <th>poutcome_other</th>
      <th>poutcome_success</th>
      <th>poutcome_unknown</th>
      <th>y_no</th>
      <th>y_yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58</td>
      <td>2143</td>
      <td>5</td>
      <td>261</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>29</td>
      <td>5</td>
      <td>151</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33</td>
      <td>2</td>
      <td>5</td>
      <td>76</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47</td>
      <td>1506</td>
      <td>5</td>
      <td>92</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>1</td>
      <td>5</td>
      <td>198</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>45206</th>
      <td>51</td>
      <td>825</td>
      <td>17</td>
      <td>977</td>
      <td>3</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>45207</th>
      <td>71</td>
      <td>1729</td>
      <td>17</td>
      <td>456</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>45208</th>
      <td>72</td>
      <td>5715</td>
      <td>17</td>
      <td>1127</td>
      <td>5</td>
      <td>184</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>45209</th>
      <td>57</td>
      <td>668</td>
      <td>17</td>
      <td>508</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45210</th>
      <td>37</td>
      <td>2971</td>
      <td>17</td>
      <td>361</td>
      <td>2</td>
      <td>188</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>45211 rows × 53 columns</p>
</div>



```python
display(mushroom_encoded)
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
      <th>p_e</th>
      <th>p_p</th>
      <th>x_b</th>
      <th>x_c</th>
      <th>x_f</th>
      <th>x_k</th>
      <th>x_s</th>
      <th>x_x</th>
      <th>s_f</th>
      <th>s_g</th>
      <th>...</th>
      <th>s.3_s</th>
      <th>s.3_v</th>
      <th>s.3_y</th>
      <th>u_d</th>
      <th>u_g</th>
      <th>u_l</th>
      <th>u_m</th>
      <th>u_p</th>
      <th>u_u</th>
      <th>u_w</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8118</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8119</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8120</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8121</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8122</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>8123 rows × 119 columns</p>
</div>



```python
display(income_encoded)
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
      <th>39</th>
      <th>77516</th>
      <th>13</th>
      <th>2174</th>
      <th>0</th>
      <th>40</th>
      <th>State-gov_ ?</th>
      <th>State-gov_ Federal-gov</th>
      <th>State-gov_ Local-gov</th>
      <th>State-gov_ Never-worked</th>
      <th>...</th>
      <th>United-States_ Scotland</th>
      <th>United-States_ South</th>
      <th>United-States_ Taiwan</th>
      <th>United-States_ Thailand</th>
      <th>United-States_ Trinadad&amp;Tobago</th>
      <th>United-States_ United-States</th>
      <th>United-States_ Vietnam</th>
      <th>United-States_ Yugoslavia</th>
      <th>&lt;=50K_ &lt;=50K</th>
      <th>&lt;=50K_ &gt;50K</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50</td>
      <td>83311</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38</td>
      <td>215646</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>234721</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28</td>
      <td>338409</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37</td>
      <td>284582</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>32555</th>
      <td>27</td>
      <td>257302</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32556</th>
      <td>40</td>
      <td>154374</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>32557</th>
      <td>58</td>
      <td>151910</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32558</th>
      <td>22</td>
      <td>201490</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32559</th>
      <td>52</td>
      <td>287927</td>
      <td>9</td>
      <td>15024</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>32560 rows × 110 columns</p>
</div>


### Converting to Matrices
Before we rush ahead, we need to convert each dataframe into a form that we can easily work with so we can utilize complex mathematical operations. We'll extract each value in the dataframes and transfer them over to a numpy array


```python
bank = bank_encoded.values
mushroom = mushroom_encoded.values
income = income_encoded.values

# Verifying that the shape matches their dataframe shape
print(bank.shape)
print(mushroom.shape)
print(income.shape)
```

    (45211, 53)
    (8123, 119)
    (32560, 110)


## Cleaning Up the Data
With the data mostly done, we'll focus on setting up the classifiers. We also have to address the fact that we can't keep the columns that classify our observations since they are not features. Since we already have each table in matrix form, we can easily remove and extract our labels for each data set! Additionally, we must shuffle the data.


```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
```


```python

# Extract the corresponding column for the labels, then remove the last two columns 
bank_labels = (bank[:,-1]).reshape(-1,1).astype(float)
bank = bank[:,: -2]


mushroom_labels = (mushroom[:,1]).reshape(-1,1).astype(float)
mushroom = mushroom[:, 2:]

income_labels = (income[:,-1]).reshape(-1,1).astype(float)
income = income[:,:-2]

print('Bank Shape: {}'.format(bank.shape))
print('Mushroom Shape: {}'.format(mushroom.shape))
print('Income Shape: {}'.format(income.shape))

# print(bank_labels.shape)
# print(mushroom_labels.shape)
# print(income_labels.shape)


# Convert every false (0) to -1 in our labels arrays
unique, counts = np.unique(mushroom_labels, return_counts=True)
count_dict = dict(zip(unique, counts))
print(count_dict) # {0: 7, 1: 4, 2: 1, 3: 2, 4: 1}

bank_labels[bank_labels == 0] = -1
mushroom_labels[mushroom_labels == 0] = -1
income_labels[income_labels == 0] = -1


# print(bank_labels) 
# print(mushroom_labels) 
# print(income_labels)


# Stack the labels with their original tables to shuffle 
bank = np.hstack((bank, bank_labels)) 
mushroom = np.hstack((mushroom, mushroom_labels)) 
income = np.hstack((income, income_labels)) 

np.random.seed(1)          
np.random.shuffle(bank) 
np.random.shuffle(mushroom) 
np.random.shuffle(income) 


```

    Bank Shape: (45211, 51)
    Mushroom Shape: (8123, 117)
    Income Shape: (32560, 108)
    {0.0: 4208, 1.0: 3915}



```python
# For class weights 
bank_unique, bank_counts = np.unique(bank_labels, return_counts=True)
bank_count_dict = dict(zip(bank_unique, bank_counts))

income_unique, income_counts = np.unique(income_labels, return_counts=True)
income_count_dict = dict(zip(income_unique, income_counts))

mushroom_unique, mushroom_counts = np.unique(mushroom_labels, return_counts=True)
mushroom_count_dict = dict(zip(mushroom_unique, mushroom_counts))



# Extract features and labels for calculating proportions of each class
bank_x = bank[:,:-1]
bank_y = bank[:,-1]

income_x = income[:,:-1]
income_y = income[:,-1]

mushroom_x = mushroom[:,:-1]
mushroom_y = mushroom[:,-1]


# Create a dict of counts of each class for each dataset
bank_weight_1_neg = len(bank_x)/bank_count_dict.get(-1)
bank_weight_1_pos = len(bank_x)/bank_count_dict.get(1)
bank_class_weights = {-1: bank_weight_1_neg , 1: bank_weight_1_pos }  

income_weight_1_neg = len(income_x)/income_count_dict.get(-1)
income_weight_1_pos = len(income_x)/income_count_dict.get(1)
income_class_weights = {-1: income_weight_1_neg , 1: income_weight_1_pos } 


mushroom_weight_1_neg = len(mushroom_x)/mushroom_count_dict.get(-1)
mushroom_weight_1_pos = len(mushroom_x)/mushroom_count_dict.get(1)
mushroom_class_weights = {-1: mushroom_weight_1_neg , 1: mushroom_weight_1_pos } 

```

# Support Vector Machines


We won't use the default SVM library but rather SVCLinear. It is similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.The main differences between LinearSVC and SVC lie in the loss function used by default, and in the handling of intercept regularization between those two implementations.

The goal here is to find the best hyperparameter C 


```python
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

```


```python
# Hyperparamter list
C_list = [0.1, 1, 10, 100, 1000, 10000]
```


```python
# Draw heatmaps for result of grid search.
def draw_heatmap(errors, param_list, title):
    plt.figure(figsize = (2,4))
    ax = sns.heatmap(errors, annot=True, fmt='.4f', yticklabels=param_list, xticklabels=[])
    ax.collections[0].colorbar.set_label('error')
    ax.set(ylabel='hyper parameter')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title(title)
    plt.show()

```


```python

def calcSVCMetrics(X_train,X_test, Y_train,Y_test, C_List, class_weights):

    clf = LinearSVC(dual=False, class_weight=class_weights)
    param_grid = {'C': C_list}

    
    # Perform 3-Fold cross validation for each hyperparameter
    grid_search = GridSearchCV(clf, param_grid, cv=3, return_train_score=True )  
    
    # Fit the model
    grid_search.fit(X_train, Y_train)    
    
    # Gather the results
    opt_C = grid_search.best_params_['C']  
                                                 
                                        
    cross_validation_accuracies =  grid_search.cv_results_['mean_test_score']
    cross_validation_errors = 1 - cross_validation_accuracies.reshape(-1,1)
    
    mean_training_accuracies = grid_search.cv_results_['mean_train_score']
    mean_training_errors = 1 - mean_training_accuracies.reshape(-1,1)
    
    Y_pred = grid_search.best_estimator_.predict(X_test)
    test_accuracy = accuracy_score(Y_pred, Y_test)
    test_error = 1 - sum(Y_pred == Y_test) / len(X_test)

    
    return opt_C,cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error
```

## SVM for BANK 


```python
# Array to track accuracies for each partition
svm_bank_accs = []

# 20% Training and 80% Testing 
bank_x_train_20, bank_x_test_80, bank_y_train_20, bank_y_test_80 = train_test_split(bank_x, bank_y, test_size=0.8)
opt_C, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcSVCMetrics(bank_x_train_20, bank_x_test_80, bank_y_train_20, bank_y_test_80, C_list, bank_class_weights)

draw_heatmap(cross_validation_errors, C_list, title='cross-validation error w.r.t C')
print("Best C: {}".format(opt_C))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per C: {}".format(mean_training_errors))
print("Avg training accuracies per C: {}".format(mean_training_accuracies))

svm_bank_accs.append(test_accuracy)
```


    
![png](output_22_0.png)
    


    Best C: 0.1
    Test error: 0.15969476623627965
    Test accuracy: 0.8403052337637203
    Avg training error per C: [[0.14902676]
     [0.14924795]
     [0.14891617]
     [0.14875028]
     [0.14897147]
     [0.14869498]]
    Avg training accuracies per C: [0.85097324 0.85075205 0.85108383 0.85124972 0.85102853 0.85130502]



```python
# 50% Training and 50% Testing 
bank_x_train_50, bank_x_test_50, bank_y_train_50, bank_y_test_50 = train_test_split(bank_x, bank_y, test_size=0.5)
opt_C, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcSVCMetrics(bank_x_train_50, bank_x_test_50, bank_y_train_50, bank_y_test_50, C_list, bank_class_weights )

draw_heatmap(cross_validation_errors, C_list, title='cross-validation error w.r.t C')
print("Best C: {}".format(opt_C))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per C: {}".format(mean_training_errors))
print("Avg training accuracies per C: {}".format(mean_training_accuracies))
svm_bank_accs.append(test_accuracy)
```


    
![png](output_23_0.png)
    


    Best C: 10
    Test error: 0.14987171547376799
    Test accuracy: 0.850128284526232
    Avg training error per C: [[0.14353019]
     [0.14350807]
     [0.14361867]
     [0.14346384]
     [0.14346384]
     [0.14359655]]
    Avg training accuracies per C: [0.85646981 0.85649193 0.85638133 0.85653616 0.85653616 0.85640345]



```python
# 80% Training and 20% Testing 
bank_x_train_80, bank_x_test_20, bank_y_train_80, bank_y_test_20 = train_test_split(bank_x, bank_y, test_size=0.2)
opt_C, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcSVCMetrics(bank_x_train_80, bank_x_test_20, bank_y_train_80, bank_y_test_20, C_list, bank_class_weights  )

draw_heatmap(cross_validation_errors, C_list, title='cross-validation error w.r.t C')
print("Best C: {}".format(opt_C))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per C: {}".format(mean_training_errors))
print("Avg training accuracies per C: {}".format(mean_training_accuracies))

svm_bank_accs.append(test_accuracy)
```


    
![png](output_24_0.png)
    


    Best C: 0.1
    Test error: 0.15348888643149394
    Test accuracy: 0.8465111135685061
    Avg training error per C: [[0.15011889]
     [0.15017419]
     [0.15034008]
     [0.15039538]
     [0.15011889]
     [0.15028478]]
    Avg training accuracies per C: [0.84988111 0.84982581 0.84965992 0.84960462 0.84988111 0.84971522]


## SVM For INCOME



```python
svm_income_accs = []
# 20% Training and 80% Testing 
income_x_train_20, income_x_test_80, income_y_train_20, income_y_test_80 = train_test_split(income_x, income_y, test_size=0.8)
opt_C, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcSVCMetrics(income_x_train_20, income_x_test_80, income_y_train_20, income_y_test_80, C_list, income_class_weights)

draw_heatmap(cross_validation_errors, C_list, title='cross-validation error w.r.t C')
print("Best C: {}".format(opt_C))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per C: {}".format(mean_training_errors))
print("Avg training accuracies per C: {}".format(mean_training_accuracies))

svm_income_accs.append(test_accuracy)
```


    
![png](output_26_0.png)
    


    Best C: 1000
    Test error: 0.1992859336609336
    Test accuracy: 0.8007140663390664
    Avg training error per C: [[0.19878695]
     [0.20845929]
     [0.19794204]
     [0.20124251]
     [0.19172213]
     [0.20170371]]
    Avg training accuracies per C: [0.80121305 0.79154071 0.80205796 0.79875749 0.80827787 0.79829629]



```python
# 50% Training and 50% Testing 
income_x_train_50, income_x_test_50, income_y_train_50, income_y_test_50 = train_test_split(income_x, income_y, test_size=0.5)
opt_C, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcSVCMetrics(income_x_train_50, income_x_test_50, income_y_train_50, income_y_test_50, C_list, income_class_weights)

draw_heatmap(cross_validation_errors, C_list, title='cross-validation error w.r.t C')
print("Best C: {}".format(opt_C))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per C: {}".format(mean_training_errors))
print("Avg training accuracies per C: {}".format(mean_training_accuracies))

svm_income_accs.append(test_accuracy)
```


    
![png](output_27_0.png)
    


    Best C: 0.1
    Test error: 0.18906633906633907
    Test accuracy: 0.8109336609336609
    Avg training error per C: [[0.19367318]
     [0.1941646 ]
     [0.19702097]
     [0.19490163]
     [0.19656026]
     [0.19241396]]
    Avg training accuracies per C: [0.80632682 0.8058354  0.80297903 0.80509837 0.80343974 0.80758604]



```python
# 80% Training and 20% Testing 
income_x_train_80, income_x_test_20, income_y_train_80, income_y_test_20 = train_test_split(income_x, income_y, test_size=0.2)
opt_C, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcSVCMetrics(income_x_train_80, income_x_test_20, income_y_train_80, income_y_test_20, C_list, income_class_weights)

draw_heatmap(cross_validation_errors, C_list, title='cross-validation error w.r.t C')
print("Best C: {}".format(opt_C))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per C: {}".format(mean_training_errors))
print("Avg training accuracies per C: {}".format(mean_training_accuracies))

svm_income_accs.append(test_accuracy)
```


    
![png](output_28_0.png)
    


    Best C: 0.1
    Test error: 0.1904176904176904
    Test accuracy: 0.8095823095823096
    Avg training error per C: [[0.19890202]
     [0.20199257]
     [0.2000922 ]
     [0.19859488]
     [0.20155109]
     [0.20160863]]
    Avg training accuracies per C: [0.80109798 0.79800743 0.7999078  0.80140512 0.79844891 0.79839137]


## SVM For MUSHROOM


```python
svm_mushroom_accs = []

# 20% Training and 80% Testing 
mushroom_x_train_20, mushroom_x_test_80, mushroom_y_train_20, mushroom_y_test_80 = train_test_split(income_x, income_y, test_size=0.8)
opt_C, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcSVCMetrics(mushroom_x_train_20, mushroom_x_test_80, mushroom_y_train_20, mushroom_y_test_80, C_list, mushroom_class_weights)

draw_heatmap(cross_validation_errors, C_list, title='cross-validation error w.r.t C')
print("Best C: {}".format(opt_C))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per C: {}".format(mean_training_errors))
print("Avg training accuracies per C: {}".format(mean_training_accuracies))
svm_mushroom_accs.append(test_accuracy)
```


    
![png](output_30_0.png)
    


    Best C: 0.1
    Test error: 0.2021652334152334
    Test accuracy: 0.7978347665847666
    Avg training error per C: [[0.19702089]
     [0.19702089]
     [0.19702089]
     [0.19702089]
     [0.19702089]
     [0.19702089]]
    Avg training accuracies per C: [0.80297911 0.80297911 0.80297911 0.80297911 0.80297911 0.80297911]



```python
# 50% Training and 50% Testing 
mushroom_x_train_50, mushroom_x_test_50, mushroom_y_train_50, mushroom_y_test_50 = train_test_split(income_x, income_y, test_size=0.5)
opt_C, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcSVCMetrics(mushroom_x_train_50, mushroom_x_test_50, mushroom_y_train_50, mushroom_y_test_50, C_list, mushroom_class_weights)

draw_heatmap(cross_validation_errors, C_list, title='cross-validation error w.r.t C')
print("Best C: {}".format(opt_C))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per C: {}".format(mean_training_errors))
print("Avg training accuracies per C: {}".format(mean_training_accuracies))
svm_mushroom_accs.append(test_accuracy)
```


    
![png](output_31_0.png)
    


    Best C: 0.1
    Test error: 0.20399262899262904
    Test accuracy: 0.796007371007371
    Avg training error per C: [[0.20042998]
     [0.20049141]
     [0.20021501]
     [0.20042998]
     [0.20042998]
     [0.20042998]]
    Avg training accuracies per C: [0.79957002 0.79950859 0.79978499 0.79957002 0.79957002 0.79957002]



```python
# 80% Training and 20% Testing 
mushroom_x_train_80, mushroom_x_test_20, mushroom_y_train_80, mushroom_y_test_20 = train_test_split(income_x, income_y, test_size=0.2)
opt_C, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcSVCMetrics(mushroom_x_train_80, mushroom_x_test_20, mushroom_y_train_80, mushroom_y_test_20, C_list, mushroom_class_weights)

draw_heatmap(cross_validation_errors, C_list, title='cross-validation error w.r.t C')
print("Best C: {}".format(opt_C))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per C: {}".format(mean_training_errors))
print("Avg training accuracies per C: {}".format(mean_training_accuracies))
svm_mushroom_accs.append(test_accuracy)
```


    
![png](output_32_0.png)
    


    Best C: 1
    Test error: 0.2010135135135135
    Test accuracy: 0.7989864864864865
    Avg training error per C: [[0.20181971]
     [0.2018389 ]
     [0.20166614]
     [0.20168533]
     [0.20155097]
     [0.20181971]]
    Avg training accuracies per C: [0.79818029 0.7981611  0.79833386 0.79831467 0.79844903 0.79818029]


## SVM Results

Now that we've performed training and testing on all three datasets and partitions, let's average all the test accuracies. We'll use this to help with our comparison against Caruana's findings. 




```python
average_accuracy = np.sum([a + b + c for a, b, c in zip(svm_bank_accs, svm_income_accs, svm_mushroom_accs)]) / 9
print(svm_bank_accs)
print(svm_income_accs)
print(svm_mushroom_accs)
print("Average SVM accuracy {}".format(average_accuracy))

```

    [0.8403052337637203, 0.850128284526232, 0.8465111135685061]
    [0.8007140663390664, 0.8109336609336609, 0.8095823095823096]
    [0.7978347665847666, 0.796007371007371, 0.7989864864864865]
    Average SVM accuracy 0.8167781436435688


# K Nearest Neighbors


```python
import scipy
from matplotlib.colors import ListedColormap
from functools import partial
from sklearn.neighbors import KNeighborsClassifier

# Hyperparameter list of possible K's
# Becuase of the scope of the project, I capped it at 15 since KNN involves expensive operations
k_range = list(range(1, 16))
```


```python

def calcKNNMetrics(X_train,X_test, Y_train,Y_test,k_range):

    param_grid = dict(n_neighbors=k_range)
    clf = KNeighborsClassifier(algorithm = 'kd_tree', weights='distance')        

    grid_search = GridSearchCV(clf, param_grid, cv=3, return_train_score=True,verbose=1, )

    # Fit the model
    grid_search.fit(X_train, Y_train)    
    
    # Gather the results
    opt_K = grid_search.best_params_
                                                 
                                        
    cross_validation_accuracies =  grid_search.cv_results_['mean_test_score']
    cross_validation_errors = 1 - cross_validation_accuracies.reshape(-1,1)
    
    mean_training_accuracies = grid_search.cv_results_['mean_train_score']
    mean_training_errors = 1 - mean_training_accuracies.reshape(-1,1)
    
    Y_pred = grid_search.best_estimator_.predict(X_test)
    test_accuracy = accuracy_score(Y_pred, Y_test)
    test_error = 1 - sum(Y_pred == Y_test) / len(X_test)

    
    return opt_K,cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error
```

## KNN for Bank


```python
knn_bank_accs = []
# 20% Training and 80% Testing 
bank_x_train_20, bank_x_test_80, bank_y_train_20, bank_y_test_80 = train_test_split(bank_x, bank_y, test_size=0.8)
opt_K, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcKNNMetrics(bank_x_train_20, bank_x_test_80, bank_y_train_20, bank_y_test_80, k_range)

draw_heatmap(cross_validation_errors, k_range, title='cross-validation error w.r.t K')
print("Best C: {}".format(opt_K))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per K: {}".format(mean_training_errors))
print("Avg training accuracies per K: {}".format(mean_training_accuracies))

knn_bank_accs.append(test_accuracy)
```

    Fitting 3 folds for each of 15 candidates, totalling 45 fits



    
![png](output_39_1.png)
    


    Best C: {'n_neighbors': 15}
    Test error: 0.11565152478641927
    Test accuracy: 0.8843484752135807
    Avg training error per K: [[0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]]
    Avg training accuracies per K: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]



```python
# 50% Training and 50% Testing 
bank_x_train_50, bank_x_test_50, bank_y_train_50, bank_y_test_50 = train_test_split(bank_x, bank_y, test_size=0.5)
opt_K, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcKNNMetrics(bank_x_train_50, bank_x_test_50, bank_y_train_50, bank_y_test_50, k_range)

draw_heatmap(cross_validation_errors, k_range, title='cross-validation error w.r.t K')
print("Best C: {}".format(opt_K))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per K: {}".format(mean_training_errors))
print("Avg training accuracies per K: {}".format(mean_training_accuracies))

knn_bank_accs.append(test_accuracy)
```

    Fitting 3 folds for each of 15 candidates, totalling 45 fits



    
![png](output_40_1.png)
    


    Best C: {'n_neighbors': 14}
    Test error: 0.11448288065115453
    Test accuracy: 0.8855171193488455
    Avg training error per K: [[0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]]
    Avg training accuracies per K: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]



```python
bank_x_train_80, bank_x_test_20, bank_y_train_80, bank_y_test_20 = train_test_split(bank_x, bank_y, test_size=0.2)
opt_K, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcKNNMetrics(bank_x_train_80, bank_x_test_20, bank_y_train_80, bank_y_test_20,k_range)

draw_heatmap(cross_validation_errors, k_range, title='cross-validation error w.r.t K')
print("Best C: {}".format(opt_K))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per K: {}".format(mean_training_errors))
print("Avg training accuracies per K: {}".format(mean_training_accuracies))

knn_bank_accs.append(test_accuracy)
```

    Fitting 3 folds for each of 15 candidates, totalling 45 fits



    
![png](output_41_1.png)
    


    Best C: {'n_neighbors': 15}
    Test error: 0.10969810903461241
    Test accuracy: 0.8903018909653876
    Avg training error per K: [[0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]]
    Avg training accuracies per K: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]


## KNN For INCOME


```python
knn_income_accs = []
# 20% Training and 80% Testing 
income_x_train_20, income_x_test_80, income_y_train_20, income_y_test_80 = train_test_split(income_x, income_y, test_size=0.8)
opt_K, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcKNNMetrics(income_x_train_20, income_x_test_80, income_y_train_20, income_y_test_80, k_range)

draw_heatmap(cross_validation_errors, k_range, title='cross-validation error w.r.t K')
print("Best C: {}".format(opt_K))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per K: {}".format(mean_training_errors))
print("Avg training accuracies per K: {}".format(mean_training_accuracies))

knn_income_accs.append(test_accuracy)
```

    Fitting 3 folds for each of 15 candidates, totalling 45 fits



    
![png](output_43_1.png)
    


    Best C: {'n_neighbors': 14}
    Test error: 0.22934582309582308
    Test accuracy: 0.7706541769041769
    Avg training error per K: [[0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]]
    Avg training accuracies per K: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]



```python
income_x_train_50, income_x_test_50, income_y_train_50, income_y_test_50 = train_test_split(income_x, income_y, test_size=0.5)
opt_K, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcKNNMetrics(income_x_train_50, income_x_test_50, income_y_train_50, income_y_test_50, k_range)

draw_heatmap(cross_validation_errors, k_range, title='cross-validation error w.r.t K')
print("Best C: {}".format(opt_K))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per K: {}".format(mean_training_errors))
print("Avg training accuracies per K: {}".format(mean_training_accuracies))

knn_income_accs.append(test_accuracy)
```

    Fitting 3 folds for each of 15 candidates, totalling 45 fits



    
![png](output_44_1.png)
    


    Best C: {'n_neighbors': 15}
    Test error: 0.2147420147420147
    Test accuracy: 0.7852579852579853
    Avg training error per K: [[0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]]
    Avg training accuracies per K: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]



```python
income_x_train_80, income_x_test_20, income_y_train_80, income_y_test_20 = train_test_split(income_x, income_y, test_size=0.2)
opt_K, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcKNNMetrics( income_x_train_80, income_x_test_20, income_y_train_80, income_y_test_20 , k_range)

draw_heatmap(cross_validation_errors, k_range, title='cross-validation error w.r.t K')
print("Best C: {}".format(opt_K))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per K: {}".format(mean_training_errors))
print("Avg training accuracies per K: {}".format(mean_training_accuracies))

knn_income_accs.append(test_accuracy)
```

    Fitting 3 folds for each of 15 candidates, totalling 45 fits



    
![png](output_45_1.png)
    


    Best C: {'n_neighbors': 15}
    Test error: 0.2071560196560197
    Test accuracy: 0.7928439803439803
    Avg training error per K: [[1.91945948e-05]
     [1.91945948e-05]
     [1.91945948e-05]
     [1.91945948e-05]
     [1.91945948e-05]
     [1.91945948e-05]
     [1.91945948e-05]
     [1.91945948e-05]
     [1.91945948e-05]
     [1.91945948e-05]
     [1.91945948e-05]
     [1.91945948e-05]
     [1.91945948e-05]
     [1.91945948e-05]
     [1.91945948e-05]]
    Avg training accuracies per K: [0.99998081 0.99998081 0.99998081 0.99998081 0.99998081 0.99998081
     0.99998081 0.99998081 0.99998081 0.99998081 0.99998081 0.99998081
     0.99998081 0.99998081 0.99998081]


## KNN For MUSHROOM


```python
knn_mushroom_accs = []

# 20% Training and 80% Testing 
mushroom_x_train_20, mushroom_x_test_80, mushroom_y_train_20, mushroom_y_test_80 = train_test_split(income_x, income_y, test_size=0.8)
opt_K, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error   =  calcKNNMetrics(mushroom_x_train_20, mushroom_x_test_80, mushroom_y_train_20, mushroom_y_test_80,  k_range)


draw_heatmap(cross_validation_errors, k_range, title='cross-validation error w.r.t K')
print("Best K: {}".format(opt_K))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per K: {}".format(mean_training_errors))
print("Avg training accuracies per K: {}".format(mean_training_accuracies))

knn_mushroom_accs.append(test_accuracy)
```

    Fitting 3 folds for each of 15 candidates, totalling 45 fits



    
![png](output_47_1.png)
    


    Best K: {'n_neighbors': 15}
    Test error: 0.2264665233415234
    Test accuracy: 0.7735334766584766
    Avg training error per K: [[0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]
     [0.]]
    Avg training accuracies per K: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]



```python
# 50% Training and 50% Testing 
mushroom_x_train_50, mushroom_x_test_50, mushroom_y_train_50, mushroom_y_test_50 = train_test_split(income_x, income_y, test_size=0.5)
opt_K, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcKNNMetrics(mushroom_x_train_50, mushroom_x_test_50, mushroom_y_train_50, mushroom_y_test_50, k_range)

draw_heatmap(cross_validation_errors, k_range, title='cross-validation error w.r.t K')
print("Best K: {}".format(opt_K))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per K: {}".format(mean_training_errors))
print("Avg training accuracies per K: {}".format(mean_training_accuracies))

knn_mushroom_accs.append(test_accuracy)
```

    Fitting 3 folds for each of 15 candidates, totalling 45 fits



    
![png](output_48_1.png)
    


    Best K: {'n_neighbors': 15}
    Test error: 0.21517199017199018
    Test accuracy: 0.7848280098280098
    Avg training error per K: [[6.14241183e-05]
     [6.14241183e-05]
     [6.14241183e-05]
     [6.14241183e-05]
     [6.14241183e-05]
     [6.14241183e-05]
     [6.14241183e-05]
     [6.14241183e-05]
     [6.14241183e-05]
     [6.14241183e-05]
     [6.14241183e-05]
     [6.14241183e-05]
     [6.14241183e-05]
     [6.14241183e-05]
     [6.14241183e-05]]
    Avg training accuracies per K: [0.99993858 0.99993858 0.99993858 0.99993858 0.99993858 0.99993858
     0.99993858 0.99993858 0.99993858 0.99993858 0.99993858 0.99993858
     0.99993858 0.99993858 0.99993858]



```python
# 80% Training and 20% Testing 
mushroom_x_train_80, mushroom_x_test_20, mushroom_y_train_80, mushroom_y_test_20 = train_test_split(income_x, income_y, test_size=0.2)
opt_C, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcKNNMetrics(mushroom_x_train_80, mushroom_x_test_20, mushroom_y_train_80, mushroom_y_test_20,k_range)

draw_heatmap(cross_validation_errors, k_range, title='cross-validation error w.r.t K')
print("Best K: {}".format(opt_K))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per K: {}".format(mean_training_errors))
print("Avg training accuracies per K: {}".format(mean_training_accuracies))

knn_mushroom_accs.append(test_accuracy)
```

    Fitting 3 folds for each of 15 candidates, totalling 45 fits



    
![png](output_49_1.png)
    


    Best K: {'n_neighbors': 15}
    Test error: 0.20915233415233414
    Test accuracy: 0.7908476658476659
    Avg training error per K: [[3.8390295e-05]
     [3.8390295e-05]
     [3.8390295e-05]
     [3.8390295e-05]
     [3.8390295e-05]
     [3.8390295e-05]
     [3.8390295e-05]
     [3.8390295e-05]
     [3.8390295e-05]
     [3.8390295e-05]
     [3.8390295e-05]
     [3.8390295e-05]
     [3.8390295e-05]
     [3.8390295e-05]
     [3.8390295e-05]]
    Avg training accuracies per K: [0.99996161 0.99996161 0.99996161 0.99996161 0.99996161 0.99996161
     0.99996161 0.99996161 0.99996161 0.99996161 0.99996161 0.99996161
     0.99996161 0.99996161 0.99996161]



```python
average_knn_accuracy = np.sum([a + b + c for a, b, c in zip(knn_bank_accs, knn_income_accs, knn_mushroom_accs)]) / 9
print(average_knn_accuracy)
```

    0.8175703089297898


# Decision Tree


```python
import seaborn as sns
from sklearn import tree

D_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```


```python
def calcDTMetrics(X_train,X_test, Y_train,Y_test, D_list):

    estimator = tree.DecisionTreeClassifier(criterion='entropy',random_state = 1 )

    param_grid = {'max_depth': D_list}
    grid_search = GridSearchCV(estimator, param_grid, cv=3, return_train_score=True)

    # Fit the model
    grid_search.fit(X_train, Y_train)    
    
    # Gather the results
    opt_D = grid_search.best_params_['max_depth']
                                                 
                                        
    cross_validation_accuracies =  grid_search.cv_results_['mean_test_score']
    cross_validation_errors = 1 - cross_validation_accuracies.reshape(-1,1)
    
    mean_training_accuracies = grid_search.cv_results_['mean_train_score']
    mean_training_errors = 1 - mean_training_accuracies.reshape(-1,1)
    
    Y_pred = grid_search.best_estimator_.predict(X_test)
    test_accuracy = accuracy_score(Y_pred, Y_test)
    test_error = 1 - sum(Y_pred == Y_test) / len(X_test)

    
    return opt_D ,cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error
```

## Decision Tree for BANK


```python
dt_bank_accs = []
# 20% Training and 80% Testing 
bank_x_train_20, bank_x_test_80, bank_y_train_20, bank_y_test_80 = train_test_split(income_x, income_y, test_size=0.8)
opt_D, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcDTMetrics(bank_x_train_20, bank_x_test_80, bank_y_train_20, bank_y_test_80 , D_list)

draw_heatmap(cross_validation_errors, D_list, title='cross-validation error w.r.t D')
print("Best D: {}".format(opt_D))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per D: {}".format(mean_training_errors))
print("Avg training accuracies per D: {}".format(mean_training_accuracies))

dt_bank_accs.append(test_accuracy)
```


    
![png](output_55_0.png)
    


    Best D: 5
    Test error: 0.15582770270270274
    Test accuracy: 0.8441722972972973
    Avg training error per D: [[0.24646806]
     [0.18542694]
     [0.16369683]
     [0.16369683]
     [0.15041451]
     [0.14534714]
     [0.13951186]
     [0.13075888]
     [0.12231271]
     [0.11425062]]
    Avg training accuracies per D: [0.75353194 0.81457306 0.83630317 0.83630317 0.84958549 0.85465286
     0.86048814 0.86924112 0.87768729 0.88574938]



```python
# 50% Training and 50% Testing 
bank_x_train_50, bank_x_test_50, bank_y_train_50, bank_y_test_50 = train_test_split(income_x, income_y, test_size=0.5)
opt_D, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcDTMetrics(bank_x_train_50, bank_x_test_50, bank_y_train_50, bank_y_test_50, D_list)

draw_heatmap(cross_validation_errors, D_list, title='cross-validation error w.r.t D')
print("Best D: {}".format(opt_D))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per D: {}".format(mean_training_errors))
print("Avg training accuracies per D: {}".format(mean_training_accuracies))

dt_bank_accs.append(test_accuracy)
```


    
![png](output_56_0.png)
    


    Best D: 7
    Test error: 0.14533169533169532
    Test accuracy: 0.8546683046683047
    Avg training error per D: [[0.24146192]
     [0.17195945]
     [0.15795454]
     [0.15795454]
     [0.15540538]
     [0.14763514]
     [0.14345823]
     [0.13799139]
     [0.13175673]
     [0.12610566]]
    Avg training accuracies per D: [0.75853808 0.82804055 0.84204546 0.84204546 0.84459462 0.85236486
     0.85654177 0.86200861 0.86824327 0.87389434]



```python
# 80% Training and 20% Testing 
bank_x_train_80, bank_x_test_20, bank_y_train_80, bank_y_test_20 = train_test_split(income_x, income_y, test_size=0.2)
opt_D, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcDTMetrics(bank_x_train_80, bank_x_test_20, bank_y_train_80, bank_y_test_20, D_list)

draw_heatmap(cross_validation_errors, D_list, title='cross-validation error w.r.t D')
print("Best D: {}".format(opt_D))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per D: {}".format(mean_training_errors))
print("Avg training accuracies per D: {}".format(mean_training_accuracies))

dt_bank_accs.append(test_accuracy)
```


    
![png](output_57_0.png)
    


    Best D: 8
    Test error: 0.14296683046683045
    Test accuracy: 0.8570331695331695
    Avg training error per D: [[0.24105498]
     [0.18673203]
     [0.1561156 ]
     [0.15600043]
     [0.15189261]
     [0.14630682]
     [0.14294764]
     [0.13918534]
     [0.13484716]
     [0.13087369]]
    Avg training accuracies per D: [0.75894502 0.81326797 0.8438844  0.84399957 0.84810739 0.85369318
     0.85705236 0.86081466 0.86515284 0.86912631]


## Decision Tree for INCOME


```python
dt_income_accs = []
# 20% Training and 80% Testing 
income_x_train_20, income_x_test_80, income_y_train_20, income_y_test_80 = train_test_split(income_x, income_y, test_size=0.8)
opt_D, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcDTMetrics(income_x_train_20, income_x_test_80, income_y_train_20, income_y_test_80, D_list)

draw_heatmap(cross_validation_errors, D_list, title='cross-validation error w.r.t D')
print("Best D: {}".format(opt_D))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per D: {}".format(mean_training_errors))
print("Avg training accuracies per D: {}".format(mean_training_accuracies))

dt_income_accs.append(test_accuracy)
```


    
![png](output_59_0.png)
    


    Best D: 6
    Test error: 0.1491477272727273
    Test accuracy: 0.8508522727272727
    Avg training error per D: [[0.24017199]
     [0.17813322]
     [0.15555905]
     [0.15356258]
     [0.14933948]
     [0.14419494]
     [0.13651711]
     [0.1305282 ]
     [0.12215894]
     [0.11417378]]
    Avg training accuracies per D: [0.75982801 0.82186678 0.84444095 0.84643742 0.85066052 0.85580506
     0.86348289 0.8694718  0.87784106 0.88582622]



```python
# 50% Training and 50% Testing 
income_x_train_50, income_x_test_50, income_y_train_50, income_y_test_50 = train_test_split(income_x, income_y, test_size=0.5)
opt_D, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcDTMetrics(income_x_train_50, income_x_test_50, income_y_train_50, income_y_test_50, D_list)

draw_heatmap(cross_validation_errors, D_list, title='cross-validation error w.r.t D')
print("Best D: {}".format(opt_D))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per D: {}".format(mean_training_errors))
print("Avg training accuracies per D: {}".format(mean_training_accuracies))

dt_income_accs.append(test_accuracy)
```


    
![png](output_60_0.png)
    


    Best D: 7
    Test error: 0.14502457002457003
    Test accuracy: 0.85497542997543
    Avg training error per D: [[0.24183047]
     [0.19582305]
     [0.1579238 ]
     [0.15740168]
     [0.14944703]
     [0.14376527]
     [0.14078622]
     [0.13713144]
     [0.13280101]
     [0.12616711]]
    Avg training accuracies per D: [0.75816953 0.80417695 0.8420762  0.84259832 0.85055297 0.85623473
     0.85921378 0.86286856 0.86719899 0.87383289]



```python
# 80% Training and 20% Testing 
income_x_train_80, income_x_test_20, income_y_train_80, income_y_test_20 = train_test_split(income_x, income_y, test_size=0.2)
opt_D, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcDTMetrics(income_x_train_80, income_x_test_20, income_y_train_80, income_y_test_20 , D_list)

draw_heatmap(cross_validation_errors, D_list, title='cross-validation error w.r.t D')
print("Best D: {}".format(opt_D))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per D: {}".format(mean_training_errors))
print("Avg training accuracies per D: {}".format(mean_training_accuracies))

dt_income_accs.append(test_accuracy)
```


    
![png](output_61_0.png)
    


    Best D: 8
    Test error: 0.14388820638820643
    Test accuracy: 0.8561117936117936
    Avg training error per D: [[0.24078624]
     [0.18629052]
     [0.15644192]
     [0.15626916]
     [0.15256445]
     [0.14590369]
     [0.14175751]
     [0.13724662]
     [0.13331161]
     [0.12922302]]
    Avg training accuracies per D: [0.75921376 0.81370948 0.84355808 0.84373084 0.84743555 0.85409631
     0.85824249 0.86275338 0.86668839 0.87077698]


## Decision Tree for Mushroom


```python
dt_mushroom_accs = []

# 20% Training and 80% Testing 
mushroom_x_train_20, mushroom_x_test_80, mushroom_y_train_20, mushroom_y_test_80 = train_test_split(income_x, income_y, test_size=0.8)
opt_D, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcDTMetrics(mushroom_x_train_20, mushroom_x_test_80, mushroom_y_train_20, mushroom_y_test_80, D_list)

draw_heatmap(cross_validation_errors, D_list, title='cross-validation error w.r.t D')
print("Best D: {}".format(opt_D))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per D: {}".format(mean_training_errors))
print("Avg training accuracies per D: {}".format(mean_training_accuracies))

dt_mushroom_accs.append(test_accuracy)
```


    
![png](output_63_0.png)
    


    Best D: 6
    Test error: 0.1515663390663391
    Test accuracy: 0.8484336609336609
    Avg training error per D: [[0.23141892]
     [0.18481252]
     [0.15164315]
     [0.15018428]
     [0.14189218]
     [0.13928161]
     [0.13490495]
     [0.13022131]
     [0.12238957]
     [0.11463453]]
    Avg training accuracies per D: [0.76858108 0.81518748 0.84835685 0.84981572 0.85810782 0.86071839
     0.86509505 0.86977869 0.87761043 0.88536547]



```python
# 50% Training and 50% Testing 
mushroom_x_train_50, mushroom_x_test_50, mushroom_y_train_50, mushroom_y_test_50 = train_test_split(income_x, income_y, test_size=0.5)
opt_D, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcDTMetrics(mushroom_x_train_50, mushroom_x_test_50, mushroom_y_train_50, mushroom_y_test_50, D_list)

draw_heatmap(cross_validation_errors, D_list, title='cross-validation error w.r.t D')
print("Best D: {}".format(opt_D))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per D: {}".format(mean_training_errors))
print("Avg training accuracies per D: {}".format(mean_training_accuracies))

dt_mushroom_accs.append(test_accuracy)
```


    
![png](output_64_0.png)
    


    Best D: 8
    Test error: 0.149017199017199
    Test accuracy: 0.850982800982801
    Avg training error per D: [[0.23937346]
     [0.17801008]
     [0.15605039]
     [0.15595826]
     [0.15205775]
     [0.14404186]
     [0.13977278]
     [0.13445947]
     [0.12954551]
     [0.12361802]]
    Avg training accuracies per D: [0.76062654 0.82198992 0.84394961 0.84404174 0.84794225 0.85595814
     0.86022722 0.86554053 0.87045449 0.87638198]



```python
# 80% Training and 20% Testing 
mushroom_x_train_80, mushroom_x_test_20, mushroom_y_train_80, mushroom_y_test_20 = train_test_split(income_x, income_y, test_size=0.2)
opt_D, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcDTMetrics(mushroom_x_train_80, mushroom_x_test_20, mushroom_y_train_80, mushroom_y_test_20, D_list)

draw_heatmap(cross_validation_errors, D_list, title='cross-validation error w.r.t D')
print("Best D: {}".format(opt_D))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per D: {}".format(mean_training_errors))
print("Avg training accuracies per D: {}".format(mean_training_accuracies))

dt_mushroom_accs.append(test_accuracy)
```


    
![png](output_65_0.png)
    


    Best D: 10
    Test error: 0.14619164619164615
    Test accuracy: 0.8538083538083538
    Avg training error per D: [[0.24132371]
     [0.19560043]
     [0.15694102]
     [0.15669149]
     [0.15091377]
     [0.14640284]
     [0.14171914]
     [0.13849433]
     [0.13363791]
     [0.12828241]]
    Avg training accuracies per D: [0.75867629 0.80439957 0.84305898 0.84330851 0.84908623 0.85359716
     0.85828086 0.86150567 0.86636209 0.87171759]



```python
average_dt_accuracy = np.sum([a + b + c for a, b, c in zip(dt_bank_accs, dt_income_accs, dt_mushroom_accs)]) / 9
print(average_dt_accuracy)
```

    0.8509836541086542



```python

data = {
    'SVM': [svm_bank_accs, svm_income_accs, svm_mushroom_accs],
    'KNN': [knn_bank_accs, knn_income_accs, knn_mushroom_accs],
    'Decision Tree': [dt_bank_accs, dt_income_accs, dt_mushroom_accs],
}



# Data preparation
datasets = ['Bank', 'Income', 'Mushroom']
classifiers = ['SVM', 'KNN', 'Decision Tree']

# Plotting
for i, dataset in enumerate(datasets):
    plt.figure(figsize=(10, 6))
    plt.title(f'Classifier Accuracies for {dataset} Dataset')
    plt.xlabel('Training Split')
    plt.ylabel('Accuracy')

    for classifier in classifiers:
        accs = data[classifier][i]
        splits = ['20/80', '50/50', '80/20']
        plt.plot(splits, accs, label=classifier)

    plt.legend()
    plt.show()
```


    
![png](output_67_0.png)
    



    
![png](output_67_1.png)
    



    
![png](output_67_2.png)
    



```python
bar_width = 0.2  # Width of each bar
bar_positions = np.arange(len(datasets))
colors = plt.cm.Set3(np.linspace(0, 1, len(classifiers)))

for i, split in enumerate(['20/80', '50/50', '80/20']):
    plt.figure(figsize=(10, 6))
    plt.title(f'Classifier Accuracies for {split} Training Split')
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.xticks(bar_positions, datasets)
    
    for j, (classifier, color) in enumerate(zip(classifiers, colors)):
        accs = [data[classifier][k][i] for k in range(len(datasets))]
        plt.bar(bar_positions + j * bar_width, accs, width=bar_width, label=classifier,  color=color)
        
        avg_acc = np.mean(accs)
        print("Average Accuracy for {}: {}".format(classifier, avg_acc))

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=len(classifiers))
  
    plt.grid(True)
    plt.show()
```

    Average Accuracy for SVM: 0.8129513555625177
    Average Accuracy for KNN: 0.8095120429254115
    Average Accuracy for Decision Tree: 0.8478194103194102



    
![png](output_68_1.png)
    


    Average Accuracy for SVM: 0.8190231054890879
    Average Accuracy for KNN: 0.8185343714782801
    Average Accuracy for Decision Tree: 0.8535421785421785



    
![png](output_68_3.png)
    


    Average Accuracy for SVM: 0.8183599698791006
    Average Accuracy for KNN: 0.8246645123856778
    Average Accuracy for Decision Tree: 0.8556511056511056



    
![png](output_68_5.png)
    



```python
average_accuracies = [np.mean(np.array(data[classifier])) for classifier in classifiers]


df_data = pd.DataFrame({
    'Classifier': classifiers,
    'Average Accuracy': average_accuracies
})

# Create a horizontal bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Average Accuracy', y='Classifier', data=df_data, palette='viridis', ci=None)
plt.title('Average Accuracies of Classifiers')
plt.xlabel('Average Accuracy')
plt.ylabel('Classifier')

# Add labels for each point
for i, v in enumerate(average_accuracies):
    plt.text(v + 0.01, i, f'{v:.3f}', color='black', va='center', fontweight='bold')

plt.show()
```


    
![png](output_69_0.png)
    



```python
# Convert accuracy data to a 2D NumPy array
heatmap_data = np.array([[data[classifier][i][j] for j in range(len(datasets))] for i in range(len(datasets))])

# Plotting the heatmap
plt.figure(figsize=(10, 6))
plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest', aspect='auto')

plt.colorbar(label='Accuracy')
plt.title('Classifier Accuracies Heatmap')
plt.xlabel('Dataset')
plt.ylabel('Classifier')
plt.xticks(np.arange(len(datasets)), datasets)
plt.yticks(np.arange(len(classifiers)), classifiers)

plt.show()
```


    
![png](output_70_0.png)
    



```python

```
