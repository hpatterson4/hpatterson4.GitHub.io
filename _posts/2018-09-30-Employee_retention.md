<br>

# EMPLOYEE_RETENTION
LEFT OR EMPLOYEED

# IMPORT 


```python
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split # Scikit-Learn 0.18+
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
```

# EXAMINE DATA


```python
df = pd.read_csv('../Employee_retention/employee_data.csv')
df.head()
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
      <th>avg_monthly_hrs</th>
      <th>department</th>
      <th>filed_complaint</th>
      <th>last_evaluation</th>
      <th>n_projects</th>
      <th>recently_promoted</th>
      <th>salary</th>
      <th>satisfaction</th>
      <th>status</th>
      <th>tenure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221</td>
      <td>engineering</td>
      <td>NaN</td>
      <td>0.932868</td>
      <td>4</td>
      <td>NaN</td>
      <td>low</td>
      <td>0.829896</td>
      <td>Left</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>232</td>
      <td>support</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>NaN</td>
      <td>low</td>
      <td>0.834544</td>
      <td>Employed</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>184</td>
      <td>sales</td>
      <td>NaN</td>
      <td>0.788830</td>
      <td>3</td>
      <td>NaN</td>
      <td>medium</td>
      <td>0.834988</td>
      <td>Employed</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>206</td>
      <td>sales</td>
      <td>NaN</td>
      <td>0.575688</td>
      <td>4</td>
      <td>NaN</td>
      <td>low</td>
      <td>0.424764</td>
      <td>Employed</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>249</td>
      <td>sales</td>
      <td>NaN</td>
      <td>0.845217</td>
      <td>3</td>
      <td>NaN</td>
      <td>low</td>
      <td>0.779043</td>
      <td>Employed</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
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
      <th>avg_monthly_hrs</th>
      <th>department</th>
      <th>filed_complaint</th>
      <th>last_evaluation</th>
      <th>n_projects</th>
      <th>recently_promoted</th>
      <th>salary</th>
      <th>satisfaction</th>
      <th>status</th>
      <th>tenure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14244</th>
      <td>178</td>
      <td>IT</td>
      <td>NaN</td>
      <td>0.735865</td>
      <td>5</td>
      <td>NaN</td>
      <td>low</td>
      <td>0.263282</td>
      <td>Employed</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>14245</th>
      <td>257</td>
      <td>sales</td>
      <td>NaN</td>
      <td>0.638604</td>
      <td>3</td>
      <td>NaN</td>
      <td>low</td>
      <td>0.868209</td>
      <td>Employed</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>14246</th>
      <td>232</td>
      <td>finance</td>
      <td>1.0</td>
      <td>0.847623</td>
      <td>5</td>
      <td>NaN</td>
      <td>medium</td>
      <td>0.898917</td>
      <td>Left</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>14247</th>
      <td>130</td>
      <td>IT</td>
      <td>NaN</td>
      <td>0.757184</td>
      <td>4</td>
      <td>NaN</td>
      <td>medium</td>
      <td>0.641304</td>
      <td>Employed</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>14248</th>
      <td>159</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.578742</td>
      <td>3</td>
      <td>NaN</td>
      <td>medium</td>
      <td>0.808850</td>
      <td>Employed</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (14249, 10)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14249 entries, 0 to 14248
    Data columns (total 10 columns):
    avg_monthly_hrs      14249 non-null int64
    department           13540 non-null object
    filed_complaint      2058 non-null float64
    last_evaluation      12717 non-null float64
    n_projects           14249 non-null int64
    recently_promoted    300 non-null float64
    salary               14249 non-null object
    satisfaction         14068 non-null float64
    status               14249 non-null object
    tenure               14068 non-null float64
    dtypes: float64(5), int64(2), object(3)
    memory usage: 1.1+ MB



```python
df.describe()
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
      <th>avg_monthly_hrs</th>
      <th>filed_complaint</th>
      <th>last_evaluation</th>
      <th>n_projects</th>
      <th>recently_promoted</th>
      <th>satisfaction</th>
      <th>tenure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>14249.000000</td>
      <td>2058.0</td>
      <td>12717.000000</td>
      <td>14249.000000</td>
      <td>300.0</td>
      <td>14068.000000</td>
      <td>14068.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>199.795775</td>
      <td>1.0</td>
      <td>0.718477</td>
      <td>3.773809</td>
      <td>1.0</td>
      <td>0.621295</td>
      <td>3.497228</td>
    </tr>
    <tr>
      <th>std</th>
      <td>50.998714</td>
      <td>0.0</td>
      <td>0.173062</td>
      <td>1.253126</td>
      <td>0.0</td>
      <td>0.250469</td>
      <td>1.460917</td>
    </tr>
    <tr>
      <th>min</th>
      <td>49.000000</td>
      <td>1.0</td>
      <td>0.316175</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.040058</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>155.000000</td>
      <td>1.0</td>
      <td>0.563866</td>
      <td>3.000000</td>
      <td>1.0</td>
      <td>0.450390</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>199.000000</td>
      <td>1.0</td>
      <td>0.724939</td>
      <td>4.000000</td>
      <td>1.0</td>
      <td>0.652527</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>245.000000</td>
      <td>1.0</td>
      <td>0.871358</td>
      <td>5.000000</td>
      <td>1.0</td>
      <td>0.824951</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>310.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>10.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isnull().sum()
```




    avg_monthly_hrs          0
    department             709
    filed_complaint      12191
    last_evaluation       1532
    n_projects               0
    recently_promoted    13949
    salary                   0
    satisfaction           181
    status                   0
    tenure                 181
    dtype: int64




```python
nullvalues = df.isnull().sum()
nullvalues.plot.barh()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10f227fd0>




![png](Employee_retention_files/Employee_retention_10_1.png)



```python
# Plotting histogram for numeric distributions 
df.hist(figsize=(10,10), xrot = -45)
plt.show()
```


![png](Employee_retention_files/Employee_retention_11_0.png)



```python
# # Plotting bar graphs for categorical distributions 
for feature in df.dtypes[df.dtypes == 'object'].index:
    sns.countplot(y=feature, data=df)
    plt.show()
```


![png](Employee_retention_files/Employee_retention_12_0.png)



![png](Employee_retention_files/Employee_retention_12_1.png)



![png](Employee_retention_files/Employee_retention_12_2.png)



```python
# Just getting comfortable with matplotlib
df.hist('last_evaluation',bins=50);
```


![png](Employee_retention_files/Employee_retention_13_0.png)



```python
# Just getting comfortable with matplotlib
df.plot(x='last_evaluation', y='avg_monthly_hrs',
        title='status', kind='scatter')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a11014898>




![png](Employee_retention_files/Employee_retention_14_1.png)



```python
# jointplot showing the kde distributions of satisfication vs. tenure
sns.jointplot(x='tenure',y='satisfaction',data=df,color='red',kind='kde');
```


![png](Employee_retention_files/Employee_retention_15_0.png)



```python
# Just getting comfortable with Seaborn
sns.countplot(y='n_projects', data = df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a111e9b70>




![png](Employee_retention_files/Employee_retention_16_1.png)


# CORRELATIONS 


```python
correlations = df.corr()
```


```python
plt.subplots(figsize=(7,5))
sns.heatmap(correlations)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a114eec88>




![png](Employee_retention_files/Employee_retention_19_1.png)



```python
mask = np.zeros_like(correlations)
mask[np.triu_indices_from(mask)] = True
plt.subplots(figsize=(10,5))
sns.axes_style("white")
sns.heatmap(correlations * 100, annot= True, mask=mask,
                 vmax=.3, square=True, cbar=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1711fe80>




![png](Employee_retention_files/Employee_retention_20_1.png)


# Segmentation
Cutting data to observe the relationship between categorical and numeric features


```python
# Segment satisfaction by status
sns.violinplot(y = 'status', x = 'satisfaction', data = df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a11523358>




![png](Employee_retention_files/Employee_retention_22_1.png)



```python
# Segment last_evaluation by status
sns.violinplot(y = 'status', x = 'last_evaluation', data = df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1a5265c0>




![png](Employee_retention_files/Employee_retention_23_1.png)



```python
# Segment by status and display the means within each class
df.groupby('status').mean()
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
      <th>avg_monthly_hrs</th>
      <th>filed_complaint</th>
      <th>last_evaluation</th>
      <th>n_projects</th>
      <th>recently_promoted</th>
      <th>satisfaction</th>
      <th>tenure</th>
    </tr>
    <tr>
      <th>status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Employed</th>
      <td>197.700286</td>
      <td>1.0</td>
      <td>0.714479</td>
      <td>3.755273</td>
      <td>1.0</td>
      <td>0.675979</td>
      <td>3.380245</td>
    </tr>
    <tr>
      <th>Left</th>
      <td>206.502948</td>
      <td>1.0</td>
      <td>0.730706</td>
      <td>3.833137</td>
      <td>1.0</td>
      <td>0.447500</td>
      <td>3.869023</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Since target is status (categorical) will do extra segmentation
# Scatterplot of satisfaction vs. last_evaluation
sns.lmplot(x='satisfaction', y='last_evaluation', hue='status',
           data=df, fit_reg=False)
```




    <seaborn.axisgrid.FacetGrid at 0x1a172350f0>




![png](Employee_retention_files/Employee_retention_25_1.png)



```python
# # Scatterplot of satisfaction vs. last_evaluation, only those who have left
sns.lmplot(x='satisfaction', y='last_evaluation',
           data=df[df.status == 'Left'], fit_reg=False)
```




    <seaborn.axisgrid.FacetGrid at 0x1a1a6b8470>




![png](Employee_retention_files/Employee_retention_26_1.png)


# DATA CLEANING


```python
# Dropping duplicates
# Were only a few
df = df.drop_duplicates()
df.shape
```




    (14221, 10)




```python
# Looking at classes of department
list(df.department.unique())
```




    ['engineering',
     'support',
     'sales',
     'IT',
     'product',
     'marketing',
     'temp',
     'procurement',
     'finance',
     nan,
     'management',
     'information_technology',
     'admin']




```python
# I will drop temporary workers 
df = df[df.department != 'temp']
df.shape
```




    (14068, 10)




```python
# unique values for filed_complaint
df.filed_complaint.unique()
```




    array([nan,  1.])




```python
df['filed_complaint'] = df.filed_complaint.fillna(0)
```


```python
# check results 
df.filed_complaint.unique()
```




    array([0., 1.])




```python
# unique values for recently promoted
df.recently_promoted.unique()
```




    array([nan,  1.])




```python
df['recently_promoted'] = df.recently_promoted.fillna(0)
```


```python
# check results 
df.recently_promoted.unique()
```




    array([0., 1.])




```python
# I will replace information technology with IT
df.department.replace('information_technology', 'IT',
                      inplace =True)
```


```python
# plot results
sns.countplot(y = 'department', data = df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a111ff5c0>




![png](Employee_retention_files/Employee_retention_38_1.png)



```python
df.isnull().sum()
```




    avg_monthly_hrs         0
    department            709
    filed_complaint         0
    last_evaluation      1351
    n_projects              0
    recently_promoted       0
    salary                  0
    satisfaction            0
    status                  0
    tenure                  0
    dtype: int64




```python
# I will just replace nan values with missing for department
df['department'].fillna('Missing', inplace = True)
```


```python
# create new  variable for missing last_evaluation
# astype converts 
df['last_evaluation_missing'] = df.last_evaluation.isnull().astype(int)
```


```python
# Fill missing values with 0 for last evaluation
df.last_evaluation.fillna(0 , inplace = True)
```


```python
#checking work
df.isnull().sum()
```




    avg_monthly_hrs            0
    department                 0
    filed_complaint            0
    last_evaluation            0
    n_projects                 0
    recently_promoted          0
    salary                     0
    satisfaction               0
    status                     0
    tenure                     0
    last_evaluation_missing    0
    dtype: int64




```python
df.head()
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
      <th>avg_monthly_hrs</th>
      <th>department</th>
      <th>filed_complaint</th>
      <th>last_evaluation</th>
      <th>n_projects</th>
      <th>recently_promoted</th>
      <th>salary</th>
      <th>satisfaction</th>
      <th>status</th>
      <th>tenure</th>
      <th>last_evaluation_missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221</td>
      <td>engineering</td>
      <td>0.0</td>
      <td>0.932868</td>
      <td>4</td>
      <td>0.0</td>
      <td>low</td>
      <td>0.829896</td>
      <td>Left</td>
      <td>5.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>232</td>
      <td>support</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.0</td>
      <td>low</td>
      <td>0.834544</td>
      <td>Employed</td>
      <td>2.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>184</td>
      <td>sales</td>
      <td>0.0</td>
      <td>0.788830</td>
      <td>3</td>
      <td>0.0</td>
      <td>medium</td>
      <td>0.834988</td>
      <td>Employed</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>206</td>
      <td>sales</td>
      <td>0.0</td>
      <td>0.575688</td>
      <td>4</td>
      <td>0.0</td>
      <td>low</td>
      <td>0.424764</td>
      <td>Employed</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>249</td>
      <td>sales</td>
      <td>0.0</td>
      <td>0.845217</td>
      <td>3</td>
      <td>0.0</td>
      <td>low</td>
      <td>0.779043</td>
      <td>Employed</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# FEATURE ENGINEERING


```python
# Looking Scatterplot of satisfaction vs. last_evaluation again
# for only those who have left
# To try and engineer
sns.lmplot(x='satisfaction', y='last_evaluation',
           data=df[df.status == 'Left'], fit_reg=False)
```




    <seaborn.axisgrid.FacetGrid at 0x1a111abc50>




![png](Employee_retention_files/Employee_retention_46_1.png)



```python
# I can engineer these findings 
df['underperformer'] = ((df.last_evaluation < 0.6) & 
                        (df.last_evaluation_missing == 0)).astype(int)
df['unhappy'] = (df.satisfaction < 0.2).astype(int)
df['overachiever'] = ((df.last_evaluation > 0.8) 
                     & (df.satisfaction > 0.7)).astype(int)
```


```python
# The proportion of observations belonging to each group
df[['underperformer', 'unhappy', 'overachiever']].mean()
```




    underperformer    0.285257
    unhappy           0.092195
    overachiever      0.177069
    dtype: float64




```python
# Converting status into an indicator variable
# Left = 1
# Right = 0
df['status']= pd.get_dummies(df.status).Left
```


```python
df['status'].unique()
```




    array([1, 0], dtype=uint64)




```python
df.status.head()
```




    0    1
    1    0
    2    0
    3    0
    4    0
    Name: status, dtype: uint8




```python
# Checking the proportion for who 'Left
df.status.mean()
```




    0.23933750355416547




```python
# Create new dataframe with dummy features
df = pd.get_dummies(df, columns=['department', 'salary'])

# Display first 10 rows
df.head(5)
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
      <th>avg_monthly_hrs</th>
      <th>filed_complaint</th>
      <th>last_evaluation</th>
      <th>n_projects</th>
      <th>recently_promoted</th>
      <th>satisfaction</th>
      <th>status</th>
      <th>tenure</th>
      <th>last_evaluation_missing</th>
      <th>underperformer</th>
      <th>...</th>
      <th>department_finance</th>
      <th>department_management</th>
      <th>department_marketing</th>
      <th>department_procurement</th>
      <th>department_product</th>
      <th>department_sales</th>
      <th>department_support</th>
      <th>salary_high</th>
      <th>salary_low</th>
      <th>salary_medium</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221</td>
      <td>0.0</td>
      <td>0.932868</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.829896</td>
      <td>1</td>
      <td>5.0</td>
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
      <th>1</th>
      <td>232</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.834544</td>
      <td>0</td>
      <td>2.0</td>
      <td>1</td>
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
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>184</td>
      <td>0.0</td>
      <td>0.788830</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.834988</td>
      <td>0</td>
      <td>3.0</td>
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
      <th>3</th>
      <td>206</td>
      <td>0.0</td>
      <td>0.575688</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.424764</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
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
      <th>4</th>
      <td>249</td>
      <td>0.0</td>
      <td>0.845217</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.779043</td>
      <td>0</td>
      <td>3.0</td>
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
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
list(df.columns)
```




    ['avg_monthly_hrs',
     'filed_complaint',
     'last_evaluation',
     'n_projects',
     'recently_promoted',
     'satisfaction',
     'status',
     'tenure',
     'last_evaluation_missing',
     'underperformer',
     'unhappy',
     'overachiever',
     'department_IT',
     'department_Missing',
     'department_admin',
     'department_engineering',
     'department_finance',
     'department_management',
     'department_marketing',
     'department_procurement',
     'department_product',
     'department_sales',
     'department_support',
     'salary_high',
     'salary_low',
     'salary_medium']



# SAVE PROGRESS


```python
df.to_csv('engineered_cleaned', index = None)
```


```python
df = pd.read_csv('../Employee_retention/engineered_cleaned')
unseen_data = pd.read_csv('../Employee_retention/unseen_raw_data.csv')
```


```python
df.head()
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
      <th>avg_monthly_hrs</th>
      <th>filed_complaint</th>
      <th>last_evaluation</th>
      <th>n_projects</th>
      <th>recently_promoted</th>
      <th>satisfaction</th>
      <th>status</th>
      <th>tenure</th>
      <th>last_evaluation_missing</th>
      <th>underperformer</th>
      <th>...</th>
      <th>department_finance</th>
      <th>department_management</th>
      <th>department_marketing</th>
      <th>department_procurement</th>
      <th>department_product</th>
      <th>department_sales</th>
      <th>department_support</th>
      <th>salary_high</th>
      <th>salary_low</th>
      <th>salary_medium</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221</td>
      <td>0.0</td>
      <td>0.932868</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.829896</td>
      <td>1</td>
      <td>5.0</td>
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
      <th>1</th>
      <td>232</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.834544</td>
      <td>0</td>
      <td>2.0</td>
      <td>1</td>
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
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>184</td>
      <td>0.0</td>
      <td>0.788830</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.834988</td>
      <td>0</td>
      <td>3.0</td>
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
      <th>3</th>
      <td>206</td>
      <td>0.0</td>
      <td>0.575688</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.424764</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
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
      <th>4</th>
      <td>249</td>
      <td>0.0</td>
      <td>0.845217</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.779043</td>
      <td>0</td>
      <td>3.0</td>
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
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
unseen_data.head()
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
      <th>avg_monthly_hrs</th>
      <th>department</th>
      <th>filed_complaint</th>
      <th>last_evaluation</th>
      <th>n_projects</th>
      <th>recently_promoted</th>
      <th>salary</th>
      <th>satisfaction</th>
      <th>tenure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>228</td>
      <td>management</td>
      <td>NaN</td>
      <td>0.735618</td>
      <td>2</td>
      <td>NaN</td>
      <td>high</td>
      <td>0.805661</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>229</td>
      <td>product</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>4</td>
      <td>NaN</td>
      <td>low</td>
      <td>0.719961</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>196</td>
      <td>sales</td>
      <td>1.0</td>
      <td>0.557426</td>
      <td>4</td>
      <td>NaN</td>
      <td>low</td>
      <td>0.749835</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>207</td>
      <td>IT</td>
      <td>NaN</td>
      <td>0.715171</td>
      <td>3</td>
      <td>NaN</td>
      <td>high</td>
      <td>0.987447</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>129</td>
      <td>management</td>
      <td>NaN</td>
      <td>0.484818</td>
      <td>2</td>
      <td>NaN</td>
      <td>low</td>
      <td>0.441219</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



# TRAIN TEST DATA


```python
y = df.status
X = df.drop('status', axis = 1)
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size =0.2,
                                                    random_state = 1234 )

```


```python
len(X_train), len(X_test), len(y_train), len(y_test)
```




    (11254, 2814, 11254, 2814)



# SCALE DATA


```python
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)

# X_pred = ss.transform(X_pred)
```

# LOGISTIC REGRESSION


```python
lr = LogisticRegression()
lr.fit(X_train, y_train)

print(lr.score(X_test, y_test))
```

    0.8464818763326226



```python
# Set up the parameters. Looking at C regularization strengths on a log scale.
lr_params = {
    'penalty':['l1','l2'],
    'solver':['liblinear'],
    'C':np.logspace(-5,0,100)
}

lr_gridsearch = GridSearchCV(LogisticRegression(), lr_params, cv=5, verbose=1)
```


```python
%%time
lr_gridsearch = lr_gridsearch.fit(X_train, y_train)
```

    Fitting 5 folds for each of 200 candidates, totalling 1000 fits
    CPU times: user 44.4 s, sys: 1.1 s, total: 45.5 s
    Wall time: 45.7 s


    [Parallel(n_jobs=1)]: Done 1000 out of 1000 | elapsed:   45.5s finished



```python
# best score on the training data:
lr_gridsearch.best_score_
```




    0.8516083170428292




```python
# best parameters on the training data:
# Lasso was chosen: this indicates that maybe unimportant (noise) variables
# is more of an issue in our data than multicollinearity.
lr_gridsearch.best_params_
```




    {'C': 0.7924828983539169, 'penalty': 'l1', 'solver': 'liblinear'}




```python
# assign the best estimator to a variable:
best_lr = lr_gridsearch.best_estimator_
```


```python
# Score it on the testing data:
best_lr.score(X_test, y_test)
```




    0.8468372423596304




```python
# slightly better than the default.
```


```python
coef_df = pd.DataFrame({
        'coef':best_lr.coef_[0],
        'feature':X.columns
    })
```


```python
coef_df['abs_coef'] = np.abs(coef_df.coef)
```


```python
# sort by absolute value of coefficient (magnitude)
coef_df.sort_values('abs_coef', ascending=False, inplace=True)
```


```python
# Show non-zero coefs and predictors
coef_df[coef_df.coef != 0]
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
      <th>coef</th>
      <th>feature</th>
      <th>abs_coef</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>-1.879827</td>
      <td>satisfaction</td>
      <td>1.879827</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.410796</td>
      <td>overachiever</td>
      <td>1.410796</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.373559</td>
      <td>last_evaluation_missing</td>
      <td>1.373559</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.323589</td>
      <td>underperformer</td>
      <td>1.323589</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.046259</td>
      <td>last_evaluation</td>
      <td>1.046259</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.539858</td>
      <td>filed_complaint</td>
      <td>0.539858</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.489306</td>
      <td>n_projects</td>
      <td>0.489306</td>
    </tr>
    <tr>
      <th>22</th>
      <td>-0.380817</td>
      <td>salary_high</td>
      <td>0.380817</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.373059</td>
      <td>tenure</td>
      <td>0.373059</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.298843</td>
      <td>avg_monthly_hrs</td>
      <td>0.298843</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.245474</td>
      <td>salary_low</td>
      <td>0.245474</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.227202</td>
      <td>recently_promoted</td>
      <td>0.227202</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.139250</td>
      <td>unhappy</td>
      <td>0.139250</td>
    </tr>
    <tr>
      <th>18</th>
      <td>-0.128141</td>
      <td>department_procurement</td>
      <td>0.128141</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-0.069641</td>
      <td>department_management</td>
      <td>0.069641</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.057416</td>
      <td>department_engineering</td>
      <td>0.057416</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-0.035294</td>
      <td>department_IT</td>
      <td>0.035294</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.033931</td>
      <td>department_Missing</td>
      <td>0.033931</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-0.031768</td>
      <td>department_admin</td>
      <td>0.031768</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.019949</td>
      <td>department_finance</td>
      <td>0.019949</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.013519</td>
      <td>department_sales</td>
      <td>0.013519</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-0.007938</td>
      <td>department_product</td>
      <td>0.007938</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-0.005222</td>
      <td>department_marketing</td>
      <td>0.005222</td>
    </tr>
  </tbody>
</table>
</div>



# LOGISTIC REGRESSION SCORE


```python
predictions = lr.predict(X_test)
```


```python
accuracy_score(y_true = y_test, y_pred = predictions)
```




    0.8464818763326226




```python
accuracy_score(y_true = y_test, y_pred = predictions, normalize=False)
```




    2382




```python
print(classification_report(y_test,predictions))
```

                 precision    recall  f1-score   support
    
              0       0.89      0.91      0.90      2145
              1       0.69      0.64      0.67       669
    
    avg / total       0.84      0.85      0.84      2814
    


# RANDOM FOREST


```python
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

print(lr.score(X_test, y_test))
```

    0.8464818763326226



```python
RandomForestClassifier().get_params()
```




    {'bootstrap': True,
     'class_weight': None,
     'criterion': 'gini',
     'max_depth': None,
     'max_features': 'auto',
     'max_leaf_nodes': None,
     'min_impurity_decrease': 0.0,
     'min_impurity_split': None,
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'min_weight_fraction_leaf': 0.0,
     'n_estimators': 10,
     'n_jobs': 1,
     'oob_score': False,
     'random_state': None,
     'verbose': 0,
     'warm_start': False}




```python
rf_params = {
    'min_samples_split':[5],
    'max_depth':[3, 5, 7]
}

rf_gridsearch = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, verbose=1)
```


```python
%%time
rf_gridsearch = rf_gridsearch.fit(X_train, y_train)
```

    Fitting 5 folds for each of 3 candidates, totalling 15 fits
    CPU times: user 1.01 s, sys: 19.6 ms, total: 1.03 s
    Wall time: 1.03 s


    [Parallel(n_jobs=1)]: Done  15 out of  15 | elapsed:    1.0s finished



```python
rf_gridsearch.best_score_
```




    0.9690776612759907




```python
rf_gridsearch.best_params_
```




    {'max_depth': 7, 'min_samples_split': 5}




```python
best_rf = rf_gridsearch.best_estimator_
```


```python
best_rf.score(X_test, y_test)
```




    0.9669509594882729




```python
list(best_rf.feature_importances_)
```




    [0.18520717815866733,
     0.0017834239265606083,
     0.07324057836113385,
     0.2609816806103563,
     0.0009973877944141293,
     0.18507539209847942,
     0.18776074277567353,
     0.0016795377487483693,
     0.015534653021916997,
     0.04200076195183799,
     0.031615545546549315,
     0.00030820603243023536,
     0.00037398192565454147,
     0.00020238531570987854,
     0.0008124618240543731,
     0.0003715306187323974,
     0.000299604779572024,
     9.117414532271443e-06,
     0.000869561700950858,
     8.683718767008375e-05,
     0.0008954154836238806,
     0.0004514619508305953,
     0.0028828410344619495,
     0.005094881687359401,
     0.001464831050079755]




```python
# The feature importances (the higher, the more important the feature).
feat_import = list(best_rf.feature_importances_)
```


```python
plt.plot(feat_import)
plt.ylabel('Feat_Importances')
plt.show()
```


![png](Employee_retention_files/Employee_retention_95_0.png)



```python
#The number of features when fit is performed.
best_rf.n_features_
```




    25




```python
predictions = best_rf.predict(X_test)
```


```python
print(classification_report(y_test,predictions))
```

                 precision    recall  f1-score   support
    
              0       0.97      0.99      0.98      2145
              1       0.96      0.90      0.93       669
    
    avg / total       0.97      0.97      0.97      2814
    



```python
confusion_matrix(y_true= y_test, y_pred = predictions)
```




    array([[2121,   24],
           [  69,  600]])




```python
best_rf.predict_proba(X_test)
```




    array([[0.26765861, 0.73234139],
           [0.98382641, 0.01617359],
           [0.94003303, 0.05996697],
           ...,
           [0.98239891, 0.01760109],
           [0.97468786, 0.02531214],
           [0.83351674, 0.16648326]])




```python
y_hat = best_rf.predict(X_test)

cm = confusion_matrix(y_test, y_hat)
cm = pd.DataFrame(cm)

sns.heatmap(cm, annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1a72dc88>




![png](Employee_retention_files/Employee_retention_101_1.png)



```python
# Much higher score with random forest 
```

# GRADIENT BOOSTING


```python
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

print(gb.score(X_test, y_test))
```

    0.9673063255152807



```python
GradientBoostingClassifier().get_params()
```




    {'criterion': 'friedman_mse',
     'init': None,
     'learning_rate': 0.1,
     'loss': 'deviance',
     'max_depth': 3,
     'max_features': None,
     'max_leaf_nodes': None,
     'min_impurity_decrease': 0.0,
     'min_impurity_split': None,
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'min_weight_fraction_leaf': 0.0,
     'n_estimators': 100,
     'presort': 'auto',
     'random_state': None,
     'subsample': 1.0,
     'verbose': 0,
     'warm_start': False}




```python
gb_params = {
    'learning_rate':[0.05, 0.1,0.2],
    'n_estimators':[20,100,200],
    'max_depth':[1,3,5]
}

gb_gridsearch = GridSearchCV(RandomForestClassifier(), gb_params, cv=5, verbose=1)
```


```python
%%time
gb_gridsearch = rf_gridsearch.fit(X_train, y_train)
```

    Fitting 5 folds for each of 3 candidates, totalling 15 fits
    CPU times: user 1.1 s, sys: 12.6 ms, total: 1.12 s
    Wall time: 1.12 s


    [Parallel(n_jobs=1)]: Done  15 out of  15 | elapsed:    1.0s finished



```python
# only a little better 
gb_gridsearch.best_score_
```




    0.9685445175048871




```python
gb_gridsearch.best_params_
```




    {'max_depth': 7, 'min_samples_split': 5}




```python
predictions = gb.predict(X_test)
```


```python
accuracy_score(y_true = y_test, y_pred = predictions)
```




    0.9673063255152807




```python
best_gb = lr_gridsearch.best_estimator_
```


```python
best_gb.predict_proba(X_test)
```




    array([[0.6020669 , 0.3979331 ],
           [0.97631714, 0.02368286],
           [0.82280054, 0.17719946],
           ...,
           [0.99830735, 0.00169265],
           [0.70519885, 0.29480115],
           [0.93748659, 0.06251341]])




```python
confusion_matrix(y_true= y_test, y_pred = predictions)
```




    array([[2111,   34],
           [  58,  611]])




```python
print(classification_report(y_test,predictions))
```

                 precision    recall  f1-score   support
    
              0       0.97      0.98      0.98      2145
              1       0.95      0.91      0.93       669
    
    avg / total       0.97      0.97      0.97      2814
    



```python
y_hat = best_gb.predict(X_test)

cm = confusion_matrix(y_test, y_hat)
cm = pd.DataFrame(cm)

sns.heatmap(cm, annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10f2252e8>




![png](Employee_retention_files/Employee_retention_116_1.png)

