<br>

# Lesson Guide
- [Import Data](#ab-Import)
- [Read Data](#ab-read)
- [Explore Data](#ab-explore)
- [Data Cleaning](#ab-cleaning)
- [Null values Market Category](#ab-mc)
- [Null values Engine HP](#ab-hp)
- [Null values Engine cylinders](#ab-ec)
- [Null values Engine fuel type](#ab-engineft)
- [Null values Number of doors](#ab-numdoors)
- [Segmentation](#ab-seg)
- [Examine transmission_type Unknown](#ab-unknown)
- [Visualization of clean data](#ab-visualsclean)
- [Engine HP squared column](#ab-hp^2)
- [Identifying outliers](#ab-outliers)
- [Dummies](#ab-dummies)
- [Train-Test](#ab-train/test)
- [Random forest](#ab-rf)
- [Gradient Boosting](#ab-gb)
- [K-neighbors](#ab-kn)
- [Visualize pred](#ab-visualizepred)
- [Best Model less features](#ab-bestmodel)

<a id='ab-Import'></a>

# Import Data


```python
# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_columns', 100)

# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
%matplotlib inline 

# Seaborn for easier visualization
import seaborn as sns

# For standardization
from sklearn.preprocessing import StandardScaler

# Helper for cross-validation
from sklearn.model_selection import GridSearchCV

# Function for splitting training and test set
from sklearn.model_selection import train_test_split # Scikit-Learn 0.18+

# Function for creating model pipelines
from sklearn.pipeline import make_pipeline

# Import Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model

# Import RandomForestClassifier and GradientBoostingClassifer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import PolynomialFeatures
from sklearn import model_selection

#import PipeLine, SelectKBest transformer, and RandomForest estimator classes
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_absolute_error
import scipy.stats
```

<a id='ab-read'></a>

# Read Data


```python
df = pd.read_csv('../Capstone/usedcarnew.csv')
```

<a id='ab-explore'></a>

# Explore data


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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Market Category</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BMW</td>
      <td>1 Series M</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>335.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Factory Tuner,Luxury,High-Performance</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>26</td>
      <td>19</td>
      <td>3916</td>
      <td>46135</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,Performance</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>28</td>
      <td>19</td>
      <td>3916</td>
      <td>40650</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,High-Performance</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>20</td>
      <td>3916</td>
      <td>36350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,Performance</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>29450</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>34500</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (11914, 16)




```python
# Drop duplicates
df = df.drop_duplicates()
print( df.shape )
```

    (11199, 16)



```python
# Make the figsize 7 x 6
plt.figure(figsize=(7,6))

# Plot heatmap of correlations
#sns.heatmap(correlations)
sns.heatmap(df.corr(), annot = True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10a239668>




![png](testcarprediction_files/testcarprediction_9_1.png)



```python
correlations = df.corr()
```


```python
# Generate a mask for the upper triangle
mask = np.zeros_like(correlations, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
```


```python
#Make the figsize 10 x 8
plt.figure(figsize=(10,8))

# Plot heatmap of correlations
sns.heatmap(correlations * 100, annot=True, fmt='.0f', mask=mask)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10a265208>




![png](testcarprediction_files/testcarprediction_12_1.png)



```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 11199 entries, 0 to 11913
    Data columns (total 16 columns):
    Make                 11199 non-null object
    Model                11199 non-null object
    Year                 11199 non-null int64
    Engine Fuel Type     11196 non-null object
    Engine HP            11130 non-null float64
    Engine Cylinders     11169 non-null float64
    Transmission Type    11199 non-null object
    Driven_Wheels        11199 non-null object
    Number of Doors      11193 non-null float64
    Market Category      7823 non-null object
    Vehicle Size         11199 non-null object
    Vehicle Style        11199 non-null object
    highway MPG          11199 non-null int64
    city mpg             11199 non-null int64
    Popularity           11199 non-null int64
    MSRP                 11199 non-null int64
    dtypes: float64(3), int64(5), object(8)
    memory usage: 1.5+ MB



```python
# Display summary statistics for the numerical features.
#Summarize numerical features
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
      <th>Year</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Number of Doors</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11199.000000</td>
      <td>11130.000000</td>
      <td>11169.000000</td>
      <td>11193.000000</td>
      <td>11199.000000</td>
      <td>11199.000000</td>
      <td>11199.000000</td>
      <td>1.119900e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2010.714528</td>
      <td>253.388859</td>
      <td>5.665950</td>
      <td>3.454123</td>
      <td>26.610590</td>
      <td>19.731851</td>
      <td>1558.483347</td>
      <td>4.192593e+04</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.228211</td>
      <td>110.150938</td>
      <td>1.797021</td>
      <td>0.872946</td>
      <td>8.977641</td>
      <td>9.177555</td>
      <td>1445.668872</td>
      <td>6.153505e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1990.000000</td>
      <td>55.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>12.000000</td>
      <td>7.000000</td>
      <td>2.000000</td>
      <td>2.000000e+03</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2007.000000</td>
      <td>172.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>22.000000</td>
      <td>16.000000</td>
      <td>549.000000</td>
      <td>2.159950e+04</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2015.000000</td>
      <td>239.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>25.000000</td>
      <td>18.000000</td>
      <td>1385.000000</td>
      <td>3.067500e+04</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2016.000000</td>
      <td>303.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>30.000000</td>
      <td>22.000000</td>
      <td>2009.000000</td>
      <td>4.303250e+04</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2017.000000</td>
      <td>1001.000000</td>
      <td>16.000000</td>
      <td>4.000000</td>
      <td>354.000000</td>
      <td>137.000000</td>
      <td>5657.000000</td>
      <td>2.065902e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
# xrot= argument that rotates x-axis labels counter-clockwise.

# Plot histogram grid
df.hist(figsize=(14,14), xrot=-45)

# Clear the text "residue"
plt.show()
```


![png](testcarprediction_files/testcarprediction_15_0.png)



```python
# Summarize categorical features
df.describe(include=['object'])
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
      <th>Make</th>
      <th>Model</th>
      <th>Engine Fuel Type</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Market Category</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11199</td>
      <td>11199</td>
      <td>11196</td>
      <td>11199</td>
      <td>11199</td>
      <td>7823</td>
      <td>11199</td>
      <td>11199</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>48</td>
      <td>915</td>
      <td>10</td>
      <td>5</td>
      <td>4</td>
      <td>71</td>
      <td>3</td>
      <td>16</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Chevrolet</td>
      <td>Silverado 1500</td>
      <td>regular unleaded</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>Crossover</td>
      <td>Compact</td>
      <td>Sedan</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1083</td>
      <td>156</td>
      <td>6658</td>
      <td>7932</td>
      <td>4354</td>
      <td>1075</td>
      <td>4395</td>
      <td>2843</td>
    </tr>
  </tbody>
</table>
</div>




```python
# barchart for missing values in each column
nullvalues = df.isnull().sum()
nullvalues.plot.barh()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a116aeeb8>




![png](testcarprediction_files/testcarprediction_17_1.png)



```python
df.head()
#df.Year.unique()
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Market Category</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BMW</td>
      <td>1 Series M</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>335.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Factory Tuner,Luxury,High-Performance</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>26</td>
      <td>19</td>
      <td>3916</td>
      <td>46135</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,Performance</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>28</td>
      <td>19</td>
      <td>3916</td>
      <td>40650</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,High-Performance</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>20</td>
      <td>3916</td>
      <td>36350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,Performance</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>29450</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>34500</td>
    </tr>
  </tbody>
</table>
</div>



<a id='ab-seg'></a>

# SEGMENTATION 

---

Cutting the data to observe the relationship between categorical features and numeric features


```python
# Bar plot for Vehicle Style
plt.subplots(figsize=(10,10))
sns.countplot(y='Vehicle Style', data=df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a116ceb00>




![png](testcarprediction_files/testcarprediction_20_1.png)



```python
# Plot bar plot for each categorical feature
# Show count of observations
#  for loop to plot bar plots of each of the categorical features.
# Some plots there is too many columns to visualize with this 

plt.subplots(figsize=(20,15))
             
             
for feature in df.dtypes[df.dtypes == 'object'].index:
    sns.countplot(y=feature, data=df)
    plt.show()
```


![png](testcarprediction_files/testcarprediction_21_0.png)



![png](testcarprediction_files/testcarprediction_21_1.png)



![png](testcarprediction_files/testcarprediction_21_2.png)



![png](testcarprediction_files/testcarprediction_21_3.png)



![png](testcarprediction_files/testcarprediction_21_4.png)



![png](testcarprediction_files/testcarprediction_21_5.png)



![png](testcarprediction_files/testcarprediction_21_6.png)



![png](testcarprediction_files/testcarprediction_21_7.png)



```python
# Segment tx_price by property_type and plot distributions
plt.subplots(figsize=(20,15))
sns.boxplot(y='Engine Fuel Type', x='MSRP', data=df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a140cf780>




![png](testcarprediction_files/testcarprediction_22_1.png)



```python
# Segment by Engine fuel Type and display the means within each class
# Maybe sue city/ Highway mpg to compare fuel type
df.groupby('Engine Fuel Type').mean()
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
      <th>Year</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Number of Doors</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
    <tr>
      <th>Engine Fuel Type</th>
      <th></th>
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
      <th>diesel</th>
      <td>2013.360000</td>
      <td>184.100671</td>
      <td>4.806667</td>
      <td>3.720000</td>
      <td>36.473333</td>
      <td>26.346667</td>
      <td>1649.240000</td>
      <td>40449.680000</td>
    </tr>
    <tr>
      <th>electric</th>
      <td>2015.318182</td>
      <td>145.318182</td>
      <td>0.000000</td>
      <td>3.901639</td>
      <td>99.590909</td>
      <td>112.696970</td>
      <td>1773.454545</td>
      <td>47943.030303</td>
    </tr>
    <tr>
      <th>flex-fuel (premium unleaded recommended/E85)</th>
      <td>2012.961538</td>
      <td>283.346154</td>
      <td>5.384615</td>
      <td>3.307692</td>
      <td>25.346154</td>
      <td>16.923077</td>
      <td>1332.807692</td>
      <td>48641.923077</td>
    </tr>
    <tr>
      <th>flex-fuel (premium unleaded required/E85)</th>
      <td>2013.849057</td>
      <td>514.716981</td>
      <td>9.396226</td>
      <td>3.358491</td>
      <td>19.943396</td>
      <td>13.283019</td>
      <td>376.641509</td>
      <td>160692.264151</td>
    </tr>
    <tr>
      <th>flex-fuel (unleaded/E85)</th>
      <td>2013.714769</td>
      <td>286.213078</td>
      <td>6.626832</td>
      <td>3.523112</td>
      <td>22.624577</td>
      <td>16.160090</td>
      <td>2278.855693</td>
      <td>36279.217587</td>
    </tr>
    <tr>
      <th>flex-fuel (unleaded/natural gas)</th>
      <td>2016.000000</td>
      <td>NaN</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>25.000000</td>
      <td>17.000000</td>
      <td>1385.000000</td>
      <td>39194.166667</td>
    </tr>
    <tr>
      <th>natural gas</th>
      <td>2015.000000</td>
      <td>110.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>38.000000</td>
      <td>27.000000</td>
      <td>2202.000000</td>
      <td>28065.000000</td>
    </tr>
    <tr>
      <th>premium unleaded (recommended)</th>
      <td>2014.686782</td>
      <td>276.525937</td>
      <td>5.173851</td>
      <td>3.377155</td>
      <td>28.407328</td>
      <td>20.190374</td>
      <td>1227.055316</td>
      <td>41812.512213</td>
    </tr>
    <tr>
      <th>premium unleaded (required)</th>
      <td>2012.688650</td>
      <td>375.906953</td>
      <td>7.005157</td>
      <td>3.062916</td>
      <td>23.856851</td>
      <td>16.649796</td>
      <td>1449.656442</td>
      <td>102814.088957</td>
    </tr>
    <tr>
      <th>regular unleaded</th>
      <td>2008.762391</td>
      <td>207.901114</td>
      <td>5.289106</td>
      <td>3.566236</td>
      <td>26.686092</td>
      <td>20.010514</td>
      <td>1570.338690</td>
      <td>23833.156053</td>
    </tr>
  </tbody>
</table>
</div>



<a id='ab-cleaning'></a>

# Data cleaning 


```python
df.isnull().sum()
```




    Make                    0
    Model                   0
    Year                    0
    Engine Fuel Type        3
    Engine HP              69
    Engine Cylinders       30
    Transmission Type       0
    Driven_Wheels           0
    Number of Doors         6
    Market Category      3376
    Vehicle Size            0
    Vehicle Style           0
    highway MPG             0
    city mpg                0
    Popularity              0
    MSRP                    0
    dtype: int64



## Examine how to deal with null values


```python
# shows all null rows information 
df.loc[(df['Market Category'].isnull()) |
              (df['Engine HP'].isnull()) |
              (df['Engine Cylinders'].isnull()) |
              (df['Number of Doors'].isnull()) |
              (df['Engine Fuel Type'].isnull())]
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Market Category</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>87</th>
      <td>Nissan</td>
      <td>200SX</td>
      <td>1996</td>
      <td>regular unleaded</td>
      <td>115.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>36</td>
      <td>26</td>
      <td>2009</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>91</th>
      <td>Nissan</td>
      <td>200SX</td>
      <td>1997</td>
      <td>regular unleaded</td>
      <td>115.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>35</td>
      <td>25</td>
      <td>2009</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>93</th>
      <td>Nissan</td>
      <td>200SX</td>
      <td>1998</td>
      <td>regular unleaded</td>
      <td>115.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>35</td>
      <td>25</td>
      <td>2009</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>203</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>37570</td>
    </tr>
    <tr>
      <th>204</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>31</td>
      <td>19</td>
      <td>1013</td>
      <td>31695</td>
    </tr>
    <tr>
      <th>205</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>31</td>
      <td>19</td>
      <td>1013</td>
      <td>38070</td>
    </tr>
    <tr>
      <th>206</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>44895</td>
    </tr>
    <tr>
      <th>209</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>34195</td>
    </tr>
    <tr>
      <th>210</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>40570</td>
    </tr>
    <tr>
      <th>211</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>38095</td>
    </tr>
    <tr>
      <th>213</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>45190</td>
    </tr>
    <tr>
      <th>214</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>31</td>
      <td>19</td>
      <td>1013</td>
      <td>32260</td>
    </tr>
    <tr>
      <th>215</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>37755</td>
    </tr>
    <tr>
      <th>216</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>41055</td>
    </tr>
    <tr>
      <th>219</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>31</td>
      <td>19</td>
      <td>1013</td>
      <td>38555</td>
    </tr>
    <tr>
      <th>220</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>31</td>
      <td>19</td>
      <td>1013</td>
      <td>35255</td>
    </tr>
    <tr>
      <th>221</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>38590</td>
    </tr>
    <tr>
      <th>222</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>34760</td>
    </tr>
    <tr>
      <th>223</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>41135</td>
    </tr>
    <tr>
      <th>224</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>45270</td>
    </tr>
    <tr>
      <th>225</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>38670</td>
    </tr>
    <tr>
      <th>228</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>38175</td>
    </tr>
    <tr>
      <th>229</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>30</td>
      <td>19</td>
      <td>1013</td>
      <td>32340</td>
    </tr>
    <tr>
      <th>231</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>34840</td>
    </tr>
    <tr>
      <th>360</th>
      <td>Mazda</td>
      <td>3</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>155.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>41</td>
      <td>30</td>
      <td>586</td>
      <td>23795</td>
    </tr>
    <tr>
      <th>361</th>
      <td>Mazda</td>
      <td>3</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>155.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>41</td>
      <td>29</td>
      <td>586</td>
      <td>19595</td>
    </tr>
    <tr>
      <th>362</th>
      <td>Mazda</td>
      <td>3</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>155.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>41</td>
      <td>29</td>
      <td>586</td>
      <td>18445</td>
    </tr>
    <tr>
      <th>368</th>
      <td>Mazda</td>
      <td>3</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>155.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>41</td>
      <td>30</td>
      <td>586</td>
      <td>19495</td>
    </tr>
    <tr>
      <th>373</th>
      <td>Mazda</td>
      <td>3</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>155.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>41</td>
      <td>29</td>
      <td>586</td>
      <td>16945</td>
    </tr>
    <tr>
      <th>375</th>
      <td>Mazda</td>
      <td>3</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>155.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>41</td>
      <td>30</td>
      <td>586</td>
      <td>20645</td>
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
    </tr>
    <tr>
      <th>11686</th>
      <td>Suzuki</td>
      <td>XL-7</td>
      <td>2006</td>
      <td>regular unleaded</td>
      <td>185.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>21</td>
      <td>16</td>
      <td>481</td>
      <td>25499</td>
    </tr>
    <tr>
      <th>11687</th>
      <td>Suzuki</td>
      <td>XL-7</td>
      <td>2006</td>
      <td>regular unleaded</td>
      <td>185.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>21</td>
      <td>16</td>
      <td>481</td>
      <td>21999</td>
    </tr>
    <tr>
      <th>11744</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2013</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>26900</td>
    </tr>
    <tr>
      <th>11745</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2013</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>16</td>
      <td>2009</td>
      <td>29440</td>
    </tr>
    <tr>
      <th>11746</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2013</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>16</td>
      <td>2009</td>
      <td>25850</td>
    </tr>
    <tr>
      <th>11747</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2013</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>30490</td>
    </tr>
    <tr>
      <th>11748</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2013</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>22</td>
      <td>16</td>
      <td>2009</td>
      <td>24850</td>
    </tr>
    <tr>
      <th>11749</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2013</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>24990</td>
    </tr>
    <tr>
      <th>11750</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2013</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>22</td>
      <td>16</td>
      <td>2009</td>
      <td>22940</td>
    </tr>
    <tr>
      <th>11751</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2014</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>31370</td>
    </tr>
    <tr>
      <th>11752</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2014</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>22</td>
      <td>16</td>
      <td>2009</td>
      <td>25300</td>
    </tr>
    <tr>
      <th>11753</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2014</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>27350</td>
    </tr>
    <tr>
      <th>11754</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2014</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>25440</td>
    </tr>
    <tr>
      <th>11755</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2014</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>16</td>
      <td>2009</td>
      <td>26300</td>
    </tr>
    <tr>
      <th>11756</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2014</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>22</td>
      <td>16</td>
      <td>2009</td>
      <td>23390</td>
    </tr>
    <tr>
      <th>11757</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2014</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>16</td>
      <td>2009</td>
      <td>30320</td>
    </tr>
    <tr>
      <th>11758</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>26670</td>
    </tr>
    <tr>
      <th>11759</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>27720</td>
    </tr>
    <tr>
      <th>11760</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>25710</td>
    </tr>
    <tr>
      <th>11761</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>30590</td>
    </tr>
    <tr>
      <th>11762</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>22</td>
      <td>16</td>
      <td>2009</td>
      <td>23660</td>
    </tr>
    <tr>
      <th>11763</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>31640</td>
    </tr>
    <tr>
      <th>11764</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>22</td>
      <td>16</td>
      <td>2009</td>
      <td>25670</td>
    </tr>
    <tr>
      <th>11792</th>
      <td>Subaru</td>
      <td>XT</td>
      <td>1991</td>
      <td>regular unleaded</td>
      <td>97.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>29</td>
      <td>22</td>
      <td>640</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>11793</th>
      <td>Subaru</td>
      <td>XT</td>
      <td>1991</td>
      <td>regular unleaded</td>
      <td>145.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>26</td>
      <td>18</td>
      <td>640</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>11794</th>
      <td>Subaru</td>
      <td>XT</td>
      <td>1991</td>
      <td>regular unleaded</td>
      <td>145.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>all wheel drive</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>640</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>11809</th>
      <td>Toyota</td>
      <td>Yaris iA</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>106.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>39</td>
      <td>30</td>
      <td>2031</td>
      <td>15950</td>
    </tr>
    <tr>
      <th>11810</th>
      <td>Toyota</td>
      <td>Yaris iA</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>106.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>40</td>
      <td>32</td>
      <td>2031</td>
      <td>17050</td>
    </tr>
    <tr>
      <th>11867</th>
      <td>GMC</td>
      <td>Yukon</td>
      <td>2015</td>
      <td>premium unleaded (recommended)</td>
      <td>420.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>4dr SUV</td>
      <td>21</td>
      <td>15</td>
      <td>549</td>
      <td>64520</td>
    </tr>
    <tr>
      <th>11868</th>
      <td>GMC</td>
      <td>Yukon</td>
      <td>2015</td>
      <td>premium unleaded (recommended)</td>
      <td>420.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>4dr SUV</td>
      <td>21</td>
      <td>14</td>
      <td>549</td>
      <td>67520</td>
    </tr>
  </tbody>
</table>
<p>3464 rows × 16 columns</p>
</div>



<a id='ab-mc'></a>

## Null values in Market Category 


```python
# Examine the null values of Market Category 
df[df['Market Category'].isnull()]
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Market Category</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>87</th>
      <td>Nissan</td>
      <td>200SX</td>
      <td>1996</td>
      <td>regular unleaded</td>
      <td>115.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>36</td>
      <td>26</td>
      <td>2009</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>91</th>
      <td>Nissan</td>
      <td>200SX</td>
      <td>1997</td>
      <td>regular unleaded</td>
      <td>115.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>35</td>
      <td>25</td>
      <td>2009</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>93</th>
      <td>Nissan</td>
      <td>200SX</td>
      <td>1998</td>
      <td>regular unleaded</td>
      <td>115.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>35</td>
      <td>25</td>
      <td>2009</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>203</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>37570</td>
    </tr>
    <tr>
      <th>204</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>31</td>
      <td>19</td>
      <td>1013</td>
      <td>31695</td>
    </tr>
    <tr>
      <th>205</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>31</td>
      <td>19</td>
      <td>1013</td>
      <td>38070</td>
    </tr>
    <tr>
      <th>206</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>44895</td>
    </tr>
    <tr>
      <th>209</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>34195</td>
    </tr>
    <tr>
      <th>210</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>40570</td>
    </tr>
    <tr>
      <th>211</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>38095</td>
    </tr>
    <tr>
      <th>213</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>45190</td>
    </tr>
    <tr>
      <th>214</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>31</td>
      <td>19</td>
      <td>1013</td>
      <td>32260</td>
    </tr>
    <tr>
      <th>215</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>37755</td>
    </tr>
    <tr>
      <th>216</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>41055</td>
    </tr>
    <tr>
      <th>219</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>31</td>
      <td>19</td>
      <td>1013</td>
      <td>38555</td>
    </tr>
    <tr>
      <th>220</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>31</td>
      <td>19</td>
      <td>1013</td>
      <td>35255</td>
    </tr>
    <tr>
      <th>221</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>38590</td>
    </tr>
    <tr>
      <th>222</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>34760</td>
    </tr>
    <tr>
      <th>223</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>41135</td>
    </tr>
    <tr>
      <th>224</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>45270</td>
    </tr>
    <tr>
      <th>225</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>38670</td>
    </tr>
    <tr>
      <th>228</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>38175</td>
    </tr>
    <tr>
      <th>229</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>30</td>
      <td>19</td>
      <td>1013</td>
      <td>32340</td>
    </tr>
    <tr>
      <th>231</th>
      <td>Chrysler</td>
      <td>300</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>292.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>1013</td>
      <td>34840</td>
    </tr>
    <tr>
      <th>360</th>
      <td>Mazda</td>
      <td>3</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>155.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>41</td>
      <td>30</td>
      <td>586</td>
      <td>23795</td>
    </tr>
    <tr>
      <th>361</th>
      <td>Mazda</td>
      <td>3</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>155.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>41</td>
      <td>29</td>
      <td>586</td>
      <td>19595</td>
    </tr>
    <tr>
      <th>362</th>
      <td>Mazda</td>
      <td>3</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>155.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>41</td>
      <td>29</td>
      <td>586</td>
      <td>18445</td>
    </tr>
    <tr>
      <th>368</th>
      <td>Mazda</td>
      <td>3</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>155.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>41</td>
      <td>30</td>
      <td>586</td>
      <td>19495</td>
    </tr>
    <tr>
      <th>373</th>
      <td>Mazda</td>
      <td>3</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>155.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>41</td>
      <td>29</td>
      <td>586</td>
      <td>16945</td>
    </tr>
    <tr>
      <th>375</th>
      <td>Mazda</td>
      <td>3</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>155.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>41</td>
      <td>30</td>
      <td>586</td>
      <td>20645</td>
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
    </tr>
    <tr>
      <th>11686</th>
      <td>Suzuki</td>
      <td>XL-7</td>
      <td>2006</td>
      <td>regular unleaded</td>
      <td>185.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>21</td>
      <td>16</td>
      <td>481</td>
      <td>25499</td>
    </tr>
    <tr>
      <th>11687</th>
      <td>Suzuki</td>
      <td>XL-7</td>
      <td>2006</td>
      <td>regular unleaded</td>
      <td>185.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>21</td>
      <td>16</td>
      <td>481</td>
      <td>21999</td>
    </tr>
    <tr>
      <th>11744</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2013</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>26900</td>
    </tr>
    <tr>
      <th>11745</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2013</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>16</td>
      <td>2009</td>
      <td>29440</td>
    </tr>
    <tr>
      <th>11746</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2013</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>16</td>
      <td>2009</td>
      <td>25850</td>
    </tr>
    <tr>
      <th>11747</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2013</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>30490</td>
    </tr>
    <tr>
      <th>11748</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2013</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>22</td>
      <td>16</td>
      <td>2009</td>
      <td>24850</td>
    </tr>
    <tr>
      <th>11749</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2013</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>24990</td>
    </tr>
    <tr>
      <th>11750</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2013</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>22</td>
      <td>16</td>
      <td>2009</td>
      <td>22940</td>
    </tr>
    <tr>
      <th>11751</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2014</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>31370</td>
    </tr>
    <tr>
      <th>11752</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2014</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>22</td>
      <td>16</td>
      <td>2009</td>
      <td>25300</td>
    </tr>
    <tr>
      <th>11753</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2014</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>27350</td>
    </tr>
    <tr>
      <th>11754</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2014</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>25440</td>
    </tr>
    <tr>
      <th>11755</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2014</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>16</td>
      <td>2009</td>
      <td>26300</td>
    </tr>
    <tr>
      <th>11756</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2014</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>22</td>
      <td>16</td>
      <td>2009</td>
      <td>23390</td>
    </tr>
    <tr>
      <th>11757</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2014</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>16</td>
      <td>2009</td>
      <td>30320</td>
    </tr>
    <tr>
      <th>11758</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>26670</td>
    </tr>
    <tr>
      <th>11759</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>27720</td>
    </tr>
    <tr>
      <th>11760</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>25710</td>
    </tr>
    <tr>
      <th>11761</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>30590</td>
    </tr>
    <tr>
      <th>11762</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>22</td>
      <td>16</td>
      <td>2009</td>
      <td>23660</td>
    </tr>
    <tr>
      <th>11763</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>2009</td>
      <td>31640</td>
    </tr>
    <tr>
      <th>11764</th>
      <td>Nissan</td>
      <td>Xterra</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>261.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>22</td>
      <td>16</td>
      <td>2009</td>
      <td>25670</td>
    </tr>
    <tr>
      <th>11792</th>
      <td>Subaru</td>
      <td>XT</td>
      <td>1991</td>
      <td>regular unleaded</td>
      <td>97.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>29</td>
      <td>22</td>
      <td>640</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>11793</th>
      <td>Subaru</td>
      <td>XT</td>
      <td>1991</td>
      <td>regular unleaded</td>
      <td>145.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>26</td>
      <td>18</td>
      <td>640</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>11794</th>
      <td>Subaru</td>
      <td>XT</td>
      <td>1991</td>
      <td>regular unleaded</td>
      <td>145.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>all wheel drive</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>640</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>11809</th>
      <td>Toyota</td>
      <td>Yaris iA</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>106.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>39</td>
      <td>30</td>
      <td>2031</td>
      <td>15950</td>
    </tr>
    <tr>
      <th>11810</th>
      <td>Toyota</td>
      <td>Yaris iA</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>106.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>40</td>
      <td>32</td>
      <td>2031</td>
      <td>17050</td>
    </tr>
    <tr>
      <th>11867</th>
      <td>GMC</td>
      <td>Yukon</td>
      <td>2015</td>
      <td>premium unleaded (recommended)</td>
      <td>420.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>4dr SUV</td>
      <td>21</td>
      <td>15</td>
      <td>549</td>
      <td>64520</td>
    </tr>
    <tr>
      <th>11868</th>
      <td>GMC</td>
      <td>Yukon</td>
      <td>2015</td>
      <td>premium unleaded (recommended)</td>
      <td>420.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>Large</td>
      <td>4dr SUV</td>
      <td>21</td>
      <td>14</td>
      <td>549</td>
      <td>67520</td>
    </tr>
  </tbody>
</table>
<p>3376 rows × 16 columns</p>
</div>




```python
list(df['Market Category'].unique())
```




    ['Factory Tuner,Luxury,High-Performance',
     'Luxury,Performance',
     'Luxury,High-Performance',
     'Luxury',
     'Performance',
     'Flex Fuel',
     'Flex Fuel,Performance',
     nan,
     'Hatchback',
     'Hatchback,Luxury,Performance',
     'Hatchback,Luxury',
     'Luxury,High-Performance,Hybrid',
     'Diesel,Luxury',
     'Hatchback,Performance',
     'Hatchback,Factory Tuner,Performance',
     'High-Performance',
     'Factory Tuner,High-Performance',
     'Exotic,High-Performance',
     'Exotic,Factory Tuner,High-Performance',
     'Factory Tuner,Performance',
     'Crossover',
     'Exotic,Luxury',
     'Exotic,Luxury,High-Performance',
     'Exotic,Luxury,Performance',
     'Factory Tuner,Luxury,Performance',
     'Flex Fuel,Luxury',
     'Crossover,Luxury',
     'Hatchback,Factory Tuner,Luxury,Performance',
     'Crossover,Hatchback',
     'Hybrid',
     'Luxury,Performance,Hybrid',
     'Crossover,Luxury,Performance,Hybrid',
     'Crossover,Luxury,Performance',
     'Exotic,Factory Tuner,Luxury,High-Performance',
     'Flex Fuel,Luxury,High-Performance',
     'Crossover,Flex Fuel',
     'Diesel',
     'Hatchback,Diesel',
     'Crossover,Luxury,Diesel',
     'Crossover,Luxury,High-Performance',
     'Exotic,Flex Fuel,Factory Tuner,Luxury,High-Performance',
     'Exotic,Flex Fuel,Luxury,High-Performance',
     'Exotic,Factory Tuner,Luxury,Performance',
     'Hatchback,Hybrid',
     'Crossover,Hybrid',
     'Hatchback,Luxury,Hybrid',
     'Flex Fuel,Luxury,Performance',
     'Crossover,Performance',
     'Luxury,Hybrid',
     'Crossover,Flex Fuel,Luxury,Performance',
     'Crossover,Flex Fuel,Luxury',
     'Crossover,Flex Fuel,Performance',
     'Hatchback,Factory Tuner,High-Performance',
     'Hatchback,Flex Fuel',
     'Factory Tuner,Luxury',
     'Crossover,Factory Tuner,Luxury,High-Performance',
     'Crossover,Factory Tuner,Luxury,Performance',
     'Crossover,Hatchback,Factory Tuner,Performance',
     'Crossover,Hatchback,Performance',
     'Flex Fuel,Hybrid',
     'Flex Fuel,Performance,Hybrid',
     'Crossover,Exotic,Luxury,High-Performance',
     'Crossover,Exotic,Luxury,Performance',
     'Exotic,Performance',
     'Exotic,Luxury,High-Performance,Hybrid',
     'Crossover,Luxury,Hybrid',
     'Flex Fuel,Factory Tuner,Luxury,High-Performance',
     'Performance,Hybrid',
     'Crossover,Factory Tuner,Performance',
     'Crossover,Diesel',
     'Flex Fuel,Diesel',
     'Crossover,Hatchback,Luxury']




```python
# Decided to drop column becuse there are too many rows missing 
# Plus other features can still describe the market Category 
df.drop('Market Category', axis = 1, inplace=True) 
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BMW</td>
      <td>1 Series M</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>335.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>26</td>
      <td>19</td>
      <td>3916</td>
      <td>46135</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>28</td>
      <td>19</td>
      <td>3916</td>
      <td>40650</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>20</td>
      <td>3916</td>
      <td>36350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>29450</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>34500</td>
    </tr>
  </tbody>
</table>
</div>



<a id='ab-hp'></a>

## All Null values in Engine HP


```python
# Examine the null values of Engine HP
df[df['Engine HP'].isnull()]
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>539</th>
      <td>FIAT</td>
      <td>500e</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr Hatchback</td>
      <td>108</td>
      <td>122</td>
      <td>819</td>
      <td>31800</td>
    </tr>
    <tr>
      <th>540</th>
      <td>FIAT</td>
      <td>500e</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr Hatchback</td>
      <td>103</td>
      <td>121</td>
      <td>819</td>
      <td>31800</td>
    </tr>
    <tr>
      <th>541</th>
      <td>FIAT</td>
      <td>500e</td>
      <td>2017</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr Hatchback</td>
      <td>103</td>
      <td>121</td>
      <td>819</td>
      <td>31800</td>
    </tr>
    <tr>
      <th>2905</th>
      <td>Lincoln</td>
      <td>Continental</td>
      <td>2017</td>
      <td>premium unleaded (recommended)</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>61</td>
      <td>55915</td>
    </tr>
    <tr>
      <th>2906</th>
      <td>Lincoln</td>
      <td>Continental</td>
      <td>2017</td>
      <td>premium unleaded (recommended)</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>61</td>
      <td>62915</td>
    </tr>
    <tr>
      <th>2907</th>
      <td>Lincoln</td>
      <td>Continental</td>
      <td>2017</td>
      <td>premium unleaded (recommended)</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>61</td>
      <td>53915</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>Lincoln</td>
      <td>Continental</td>
      <td>2017</td>
      <td>premium unleaded (recommended)</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>61</td>
      <td>64915</td>
    </tr>
    <tr>
      <th>4203</th>
      <td>Ford</td>
      <td>Escape</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr SUV</td>
      <td>30</td>
      <td>23</td>
      <td>5657</td>
      <td>29100</td>
    </tr>
    <tr>
      <th>4204</th>
      <td>Ford</td>
      <td>Escape</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr SUV</td>
      <td>28</td>
      <td>22</td>
      <td>5657</td>
      <td>30850</td>
    </tr>
    <tr>
      <th>4205</th>
      <td>Ford</td>
      <td>Escape</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr SUV</td>
      <td>28</td>
      <td>22</td>
      <td>5657</td>
      <td>26850</td>
    </tr>
    <tr>
      <th>4206</th>
      <td>Ford</td>
      <td>Escape</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr SUV</td>
      <td>30</td>
      <td>23</td>
      <td>5657</td>
      <td>25100</td>
    </tr>
    <tr>
      <th>4705</th>
      <td>Honda</td>
      <td>Fit EV</td>
      <td>2013</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>105</td>
      <td>132</td>
      <td>2202</td>
      <td>36625</td>
    </tr>
    <tr>
      <th>4706</th>
      <td>Honda</td>
      <td>Fit EV</td>
      <td>2014</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>105</td>
      <td>132</td>
      <td>2202</td>
      <td>36625</td>
    </tr>
    <tr>
      <th>4785</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>99</td>
      <td>110</td>
      <td>5657</td>
      <td>29170</td>
    </tr>
    <tr>
      <th>4789</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>99</td>
      <td>110</td>
      <td>5657</td>
      <td>29170</td>
    </tr>
    <tr>
      <th>4798</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2017</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>99</td>
      <td>110</td>
      <td>5657</td>
      <td>29120</td>
    </tr>
    <tr>
      <th>4914</th>
      <td>Ford</td>
      <td>Freestar</td>
      <td>2005</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Passenger Minivan</td>
      <td>22</td>
      <td>16</td>
      <td>5657</td>
      <td>28030</td>
    </tr>
    <tr>
      <th>4915</th>
      <td>Ford</td>
      <td>Freestar</td>
      <td>2005</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Passenger Minivan</td>
      <td>22</td>
      <td>16</td>
      <td>5657</td>
      <td>23930</td>
    </tr>
    <tr>
      <th>4916</th>
      <td>Ford</td>
      <td>Freestar</td>
      <td>2005</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Cargo Minivan</td>
      <td>22</td>
      <td>16</td>
      <td>5657</td>
      <td>21630</td>
    </tr>
    <tr>
      <th>4917</th>
      <td>Ford</td>
      <td>Freestar</td>
      <td>2005</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Passenger Minivan</td>
      <td>22</td>
      <td>16</td>
      <td>5657</td>
      <td>26530</td>
    </tr>
    <tr>
      <th>4918</th>
      <td>Ford</td>
      <td>Freestar</td>
      <td>2005</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Passenger Minivan</td>
      <td>21</td>
      <td>16</td>
      <td>5657</td>
      <td>29030</td>
    </tr>
    <tr>
      <th>4919</th>
      <td>Ford</td>
      <td>Freestar</td>
      <td>2005</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Passenger Minivan</td>
      <td>21</td>
      <td>16</td>
      <td>5657</td>
      <td>32755</td>
    </tr>
    <tr>
      <th>5778</th>
      <td>Mitsubishi</td>
      <td>i-MiEV</td>
      <td>2014</td>
      <td>electric</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>99</td>
      <td>126</td>
      <td>436</td>
      <td>22995</td>
    </tr>
    <tr>
      <th>5825</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2015</td>
      <td>flex-fuel (unleaded/natural gas)</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>1385</td>
      <td>40660</td>
    </tr>
    <tr>
      <th>5830</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2015</td>
      <td>flex-fuel (unleaded/natural gas)</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>1385</td>
      <td>37535</td>
    </tr>
    <tr>
      <th>5831</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2016</td>
      <td>flex-fuel (unleaded/natural gas)</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>1385</td>
      <td>40810</td>
    </tr>
    <tr>
      <th>5833</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2016</td>
      <td>flex-fuel (unleaded/natural gas)</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>1385</td>
      <td>37570</td>
    </tr>
    <tr>
      <th>5839</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2017</td>
      <td>flex-fuel (unleaded/natural gas)</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>1385</td>
      <td>37675</td>
    </tr>
    <tr>
      <th>5840</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2017</td>
      <td>flex-fuel (unleaded/natural gas)</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>1385</td>
      <td>40915</td>
    </tr>
    <tr>
      <th>6385</th>
      <td>Nissan</td>
      <td>Leaf</td>
      <td>2014</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>101</td>
      <td>126</td>
      <td>2009</td>
      <td>35020</td>
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
      <th>6578</th>
      <td>Mercedes-Benz</td>
      <td>M-Class</td>
      <td>2015</td>
      <td>diesel</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>29</td>
      <td>22</td>
      <td>617</td>
      <td>49800</td>
    </tr>
    <tr>
      <th>6908</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>38</td>
      <td>41</td>
      <td>61</td>
      <td>35010</td>
    </tr>
    <tr>
      <th>6910</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>38</td>
      <td>41</td>
      <td>61</td>
      <td>39510</td>
    </tr>
    <tr>
      <th>6916</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>38</td>
      <td>41</td>
      <td>61</td>
      <td>36760</td>
    </tr>
    <tr>
      <th>6918</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>38</td>
      <td>41</td>
      <td>61</td>
      <td>47670</td>
    </tr>
    <tr>
      <th>6921</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2014</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>90</td>
      <td>88</td>
      <td>1391</td>
      <td>79900</td>
    </tr>
    <tr>
      <th>6922</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2014</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>97</td>
      <td>94</td>
      <td>1391</td>
      <td>69900</td>
    </tr>
    <tr>
      <th>6923</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2014</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>94</td>
      <td>86</td>
      <td>1391</td>
      <td>104500</td>
    </tr>
    <tr>
      <th>6924</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2014</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>90</td>
      <td>88</td>
      <td>1391</td>
      <td>93400</td>
    </tr>
    <tr>
      <th>6925</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>97</td>
      <td>94</td>
      <td>1391</td>
      <td>69900</td>
    </tr>
    <tr>
      <th>6926</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>102</td>
      <td>101</td>
      <td>1391</td>
      <td>75000</td>
    </tr>
    <tr>
      <th>6927</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>106</td>
      <td>95</td>
      <td>1391</td>
      <td>85000</td>
    </tr>
    <tr>
      <th>6928</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>98</td>
      <td>89</td>
      <td>1391</td>
      <td>105000</td>
    </tr>
    <tr>
      <th>6929</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>90</td>
      <td>88</td>
      <td>1391</td>
      <td>80000</td>
    </tr>
    <tr>
      <th>6930</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>105</td>
      <td>102</td>
      <td>1391</td>
      <td>79500</td>
    </tr>
    <tr>
      <th>6931</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>101</td>
      <td>98</td>
      <td>1391</td>
      <td>66000</td>
    </tr>
    <tr>
      <th>6932</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>105</td>
      <td>92</td>
      <td>1391</td>
      <td>134500</td>
    </tr>
    <tr>
      <th>6933</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>100</td>
      <td>97</td>
      <td>1391</td>
      <td>74500</td>
    </tr>
    <tr>
      <th>6934</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>107</td>
      <td>101</td>
      <td>1391</td>
      <td>71000</td>
    </tr>
    <tr>
      <th>6935</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>102</td>
      <td>101</td>
      <td>1391</td>
      <td>75000</td>
    </tr>
    <tr>
      <th>6936</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>107</td>
      <td>101</td>
      <td>1391</td>
      <td>89500</td>
    </tr>
    <tr>
      <th>6937</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>100</td>
      <td>91</td>
      <td>1391</td>
      <td>112000</td>
    </tr>
    <tr>
      <th>6938</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>90</td>
      <td>88</td>
      <td>1391</td>
      <td>70000</td>
    </tr>
    <tr>
      <th>8374</th>
      <td>Toyota</td>
      <td>RAV4 EV</td>
      <td>2013</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>74</td>
      <td>78</td>
      <td>2031</td>
      <td>49800</td>
    </tr>
    <tr>
      <th>8375</th>
      <td>Toyota</td>
      <td>RAV4 EV</td>
      <td>2014</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>74</td>
      <td>78</td>
      <td>2031</td>
      <td>49800</td>
    </tr>
    <tr>
      <th>9850</th>
      <td>Kia</td>
      <td>Soul EV</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Wagon</td>
      <td>92</td>
      <td>120</td>
      <td>1720</td>
      <td>35700</td>
    </tr>
    <tr>
      <th>9851</th>
      <td>Kia</td>
      <td>Soul EV</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Wagon</td>
      <td>92</td>
      <td>120</td>
      <td>1720</td>
      <td>33700</td>
    </tr>
    <tr>
      <th>9852</th>
      <td>Kia</td>
      <td>Soul EV</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Wagon</td>
      <td>92</td>
      <td>120</td>
      <td>1720</td>
      <td>33950</td>
    </tr>
    <tr>
      <th>9853</th>
      <td>Kia</td>
      <td>Soul EV</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Wagon</td>
      <td>92</td>
      <td>120</td>
      <td>1720</td>
      <td>31950</td>
    </tr>
    <tr>
      <th>9854</th>
      <td>Kia</td>
      <td>Soul EV</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Wagon</td>
      <td>92</td>
      <td>120</td>
      <td>1720</td>
      <td>35950</td>
    </tr>
  </tbody>
</table>
<p>69 rows × 15 columns</p>
</div>



### Null values for model Fiat model 500e 
Fill in Engine HP


```python
# Couldn't compare to anything else provided in data
df[df['Model'] == '500e']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>539</th>
      <td>FIAT</td>
      <td>500e</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr Hatchback</td>
      <td>108</td>
      <td>122</td>
      <td>819</td>
      <td>31800</td>
    </tr>
    <tr>
      <th>540</th>
      <td>FIAT</td>
      <td>500e</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr Hatchback</td>
      <td>103</td>
      <td>121</td>
      <td>819</td>
      <td>31800</td>
    </tr>
    <tr>
      <th>541</th>
      <td>FIAT</td>
      <td>500e</td>
      <td>2017</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr Hatchback</td>
      <td>103</td>
      <td>121</td>
      <td>819</td>
      <td>31800</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Searched Web for Engine HP and all years are the same 

for i, j in df['Engine HP'].iteritems():
    if np.isnan(df['Engine HP'][i]):
#         print (i,j)
        if df['Make'][i] == 'FIAT':
            df['Engine HP'][i] = 111
```

    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
df[df['Model'] == '500e']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>539</th>
      <td>FIAT</td>
      <td>500e</td>
      <td>2015</td>
      <td>electric</td>
      <td>111.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr Hatchback</td>
      <td>108</td>
      <td>122</td>
      <td>819</td>
      <td>31800</td>
    </tr>
    <tr>
      <th>540</th>
      <td>FIAT</td>
      <td>500e</td>
      <td>2016</td>
      <td>electric</td>
      <td>111.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr Hatchback</td>
      <td>103</td>
      <td>121</td>
      <td>819</td>
      <td>31800</td>
    </tr>
    <tr>
      <th>541</th>
      <td>FIAT</td>
      <td>500e</td>
      <td>2017</td>
      <td>electric</td>
      <td>111.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr Hatchback</td>
      <td>103</td>
      <td>121</td>
      <td>819</td>
      <td>31800</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Engine HP'].isnull().sum()
```




    66



### model Continental null values 
Fill in Engine HP


```python
df[df['Model'] == 'Continental']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2891</th>
      <td>Bentley</td>
      <td>Continental</td>
      <td>2001</td>
      <td>premium unleaded (required)</td>
      <td>420.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Coupe</td>
      <td>15</td>
      <td>10</td>
      <td>520</td>
      <td>299900</td>
    </tr>
    <tr>
      <th>2892</th>
      <td>Bentley</td>
      <td>Continental</td>
      <td>2001</td>
      <td>premium unleaded (required)</td>
      <td>400.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Coupe</td>
      <td>15</td>
      <td>10</td>
      <td>520</td>
      <td>279900</td>
    </tr>
    <tr>
      <th>2893</th>
      <td>Bentley</td>
      <td>Continental</td>
      <td>2001</td>
      <td>premium unleaded (required)</td>
      <td>420.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Coupe</td>
      <td>15</td>
      <td>10</td>
      <td>520</td>
      <td>309900</td>
    </tr>
    <tr>
      <th>2894</th>
      <td>Bentley</td>
      <td>Continental</td>
      <td>2001</td>
      <td>premium unleaded (required)</td>
      <td>420.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Coupe</td>
      <td>15</td>
      <td>10</td>
      <td>520</td>
      <td>319900</td>
    </tr>
    <tr>
      <th>2895</th>
      <td>Bentley</td>
      <td>Continental</td>
      <td>2003</td>
      <td>premium unleaded (required)</td>
      <td>420.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Coupe</td>
      <td>15</td>
      <td>10</td>
      <td>520</td>
      <td>328990</td>
    </tr>
    <tr>
      <th>2896</th>
      <td>Bentley</td>
      <td>Continental</td>
      <td>2003</td>
      <td>premium unleaded (required)</td>
      <td>420.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Coupe</td>
      <td>15</td>
      <td>10</td>
      <td>520</td>
      <td>318990</td>
    </tr>
    <tr>
      <th>2897</th>
      <td>Lincoln</td>
      <td>Continental</td>
      <td>2001</td>
      <td>regular unleaded</td>
      <td>275.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>23</td>
      <td>15</td>
      <td>61</td>
      <td>39660</td>
    </tr>
    <tr>
      <th>2898</th>
      <td>Lincoln</td>
      <td>Continental</td>
      <td>2002</td>
      <td>premium unleaded (required)</td>
      <td>275.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>23</td>
      <td>15</td>
      <td>61</td>
      <td>39895</td>
    </tr>
    <tr>
      <th>2899</th>
      <td>Lincoln</td>
      <td>Continental</td>
      <td>2002</td>
      <td>premium unleaded (required)</td>
      <td>275.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>23</td>
      <td>15</td>
      <td>61</td>
      <td>38185</td>
    </tr>
    <tr>
      <th>2900</th>
      <td>Lincoln</td>
      <td>Continental</td>
      <td>2002</td>
      <td>premium unleaded (required)</td>
      <td>275.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>23</td>
      <td>15</td>
      <td>61</td>
      <td>38790</td>
    </tr>
    <tr>
      <th>2901</th>
      <td>Lincoln</td>
      <td>Continental</td>
      <td>2002</td>
      <td>premium unleaded (required)</td>
      <td>275.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>23</td>
      <td>15</td>
      <td>61</td>
      <td>39775</td>
    </tr>
    <tr>
      <th>2902</th>
      <td>Lincoln</td>
      <td>Continental</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>305.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>26</td>
      <td>17</td>
      <td>61</td>
      <td>44560</td>
    </tr>
    <tr>
      <th>2903</th>
      <td>Lincoln</td>
      <td>Continental</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>305.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>24</td>
      <td>16</td>
      <td>61</td>
      <td>46560</td>
    </tr>
    <tr>
      <th>2904</th>
      <td>Lincoln</td>
      <td>Continental</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>305.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>24</td>
      <td>16</td>
      <td>61</td>
      <td>49515</td>
    </tr>
    <tr>
      <th>2905</th>
      <td>Lincoln</td>
      <td>Continental</td>
      <td>2017</td>
      <td>premium unleaded (recommended)</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>61</td>
      <td>55915</td>
    </tr>
    <tr>
      <th>2906</th>
      <td>Lincoln</td>
      <td>Continental</td>
      <td>2017</td>
      <td>premium unleaded (recommended)</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>61</td>
      <td>62915</td>
    </tr>
    <tr>
      <th>2907</th>
      <td>Lincoln</td>
      <td>Continental</td>
      <td>2017</td>
      <td>premium unleaded (recommended)</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>61</td>
      <td>53915</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>Lincoln</td>
      <td>Continental</td>
      <td>2017</td>
      <td>premium unleaded (recommended)</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>61</td>
      <td>64915</td>
    </tr>
    <tr>
      <th>2909</th>
      <td>Lincoln</td>
      <td>Continental</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>305.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>26</td>
      <td>17</td>
      <td>61</td>
      <td>47515</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[2905,'Model'] = 'Continental R'
df.loc[2906,'Model'] = 'Continental SR'
df.loc[2907,'Model'] = 'Continental SR'
df.loc[2908,'Model'] = 'Continental R'
```


```python
df.loc[2905]
```




    Make                                        Lincoln
    Model                                 Continental R
    Year                                           2017
    Engine Fuel Type     premium unleaded (recommended)
    Engine HP                                       NaN
    Engine Cylinders                                  6
    Transmission Type                         AUTOMATIC
    Driven_Wheels                       all wheel drive
    Number of Doors                                   4
    Vehicle Size                                  Large
    Vehicle Style                                 Sedan
    highway MPG                                      25
    city mpg                                         17
    Popularity                                       61
    MSRP                                          55915
    Name: 2905, dtype: object




```python
for i, j in df['Engine HP'].iteritems():
    if np.isnan(df['Engine HP'][i]):
#         print (i,j)
        if df['Model'][i] == 'Continental R':
            df['Engine HP'][i] = 400
        elif df['Model'][i] == 'Continental SR':
            df['Engine HP'][i] = 335
```

    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
df[df['Model'] == 'Continental R']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2905</th>
      <td>Lincoln</td>
      <td>Continental R</td>
      <td>2017</td>
      <td>premium unleaded (recommended)</td>
      <td>400.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>61</td>
      <td>55915</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>Lincoln</td>
      <td>Continental R</td>
      <td>2017</td>
      <td>premium unleaded (recommended)</td>
      <td>400.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>61</td>
      <td>64915</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Number is still going down
df['Engine HP'].isnull().sum()
```




    62



### model Escape null values 
Fill in Engine HP


```python
df[df['Model'] == 'Escape']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4192</th>
      <td>Ford</td>
      <td>Escape</td>
      <td>2015</td>
      <td>premium unleaded (recommended)</td>
      <td>178.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr SUV</td>
      <td>32</td>
      <td>23</td>
      <td>5657</td>
      <td>29735</td>
    </tr>
    <tr>
      <th>4193</th>
      <td>Ford</td>
      <td>Escape</td>
      <td>2015</td>
      <td>premium unleaded (recommended)</td>
      <td>178.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr SUV</td>
      <td>32</td>
      <td>23</td>
      <td>5657</td>
      <td>25650</td>
    </tr>
    <tr>
      <th>4194</th>
      <td>Ford</td>
      <td>Escape</td>
      <td>2015</td>
      <td>premium unleaded (recommended)</td>
      <td>178.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr SUV</td>
      <td>30</td>
      <td>22</td>
      <td>5657</td>
      <td>27400</td>
    </tr>
    <tr>
      <th>4195</th>
      <td>Ford</td>
      <td>Escape</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>168.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr SUV</td>
      <td>31</td>
      <td>22</td>
      <td>5657</td>
      <td>23450</td>
    </tr>
    <tr>
      <th>4196</th>
      <td>Ford</td>
      <td>Escape</td>
      <td>2015</td>
      <td>premium unleaded (recommended)</td>
      <td>178.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr SUV</td>
      <td>30</td>
      <td>22</td>
      <td>5657</td>
      <td>31485</td>
    </tr>
    <tr>
      <th>4197</th>
      <td>Ford</td>
      <td>Escape</td>
      <td>2016</td>
      <td>premium unleaded (recommended)</td>
      <td>178.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr SUV</td>
      <td>29</td>
      <td>22</td>
      <td>5657</td>
      <td>27540</td>
    </tr>
    <tr>
      <th>4198</th>
      <td>Ford</td>
      <td>Escape</td>
      <td>2016</td>
      <td>premium unleaded (recommended)</td>
      <td>178.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr SUV</td>
      <td>32</td>
      <td>23</td>
      <td>5657</td>
      <td>25790</td>
    </tr>
    <tr>
      <th>4199</th>
      <td>Ford</td>
      <td>Escape</td>
      <td>2016</td>
      <td>premium unleaded (recommended)</td>
      <td>178.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr SUV</td>
      <td>32</td>
      <td>23</td>
      <td>5657</td>
      <td>29995</td>
    </tr>
    <tr>
      <th>4200</th>
      <td>Ford</td>
      <td>Escape</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>168.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr SUV</td>
      <td>31</td>
      <td>22</td>
      <td>5657</td>
      <td>23590</td>
    </tr>
    <tr>
      <th>4201</th>
      <td>Ford</td>
      <td>Escape</td>
      <td>2016</td>
      <td>premium unleaded (recommended)</td>
      <td>178.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr SUV</td>
      <td>29</td>
      <td>22</td>
      <td>5657</td>
      <td>31745</td>
    </tr>
    <tr>
      <th>4202</th>
      <td>Ford</td>
      <td>Escape</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>168.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr SUV</td>
      <td>29</td>
      <td>21</td>
      <td>5657</td>
      <td>23600</td>
    </tr>
    <tr>
      <th>4203</th>
      <td>Ford</td>
      <td>Escape</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr SUV</td>
      <td>30</td>
      <td>23</td>
      <td>5657</td>
      <td>29100</td>
    </tr>
    <tr>
      <th>4204</th>
      <td>Ford</td>
      <td>Escape</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr SUV</td>
      <td>28</td>
      <td>22</td>
      <td>5657</td>
      <td>30850</td>
    </tr>
    <tr>
      <th>4205</th>
      <td>Ford</td>
      <td>Escape</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr SUV</td>
      <td>28</td>
      <td>22</td>
      <td>5657</td>
      <td>26850</td>
    </tr>
    <tr>
      <th>4206</th>
      <td>Ford</td>
      <td>Escape</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr SUV</td>
      <td>30</td>
      <td>23</td>
      <td>5657</td>
      <td>25100</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[4203,'Model'] = 'Escape S'
df.loc[4204,'Model'] = 'Escape SE'
df.loc[4205,'Model'] = 'Escape SE'
df.loc[4206,'Model'] = 'Escape S'
```


```python
df.loc[4205]
```




    Make                             Ford
    Model                       Escape SE
    Year                             2017
    Engine Fuel Type     regular unleaded
    Engine HP                         NaN
    Engine Cylinders                    4
    Transmission Type           AUTOMATIC
    Driven_Wheels         all wheel drive
    Number of Doors                     4
    Vehicle Size                  Compact
    Vehicle Style                 4dr SUV
    highway MPG                        28
    city mpg                           22
    Popularity                       5657
    MSRP                            26850
    Name: 4205, dtype: object




```python
for i, j in df['Engine HP'].iteritems():
    if np.isnan(df['Engine HP'][i]):
#         print (i,j)
        if df['Model'][i] == 'Escape S':
            df['Engine HP'][i] = 168
        elif df['Model'][i] == 'Escape SE':
            df['Engine HP'][i] = 179
```

    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
# Number is still going down
df['Engine HP'].isnull().sum()
```




    58



### model Fit EV null values 
Fill in Engine HP


```python
df[df['Model'] == 'Fit EV']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4705</th>
      <td>Honda</td>
      <td>Fit EV</td>
      <td>2013</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>105</td>
      <td>132</td>
      <td>2202</td>
      <td>36625</td>
    </tr>
    <tr>
      <th>4706</th>
      <td>Honda</td>
      <td>Fit EV</td>
      <td>2014</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>105</td>
      <td>132</td>
      <td>2202</td>
      <td>36625</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i, j in df['Engine HP'].iteritems():
    if np.isnan(df['Engine HP'][i]):
#         print (i,j)
        if df['Model'][i] == 'Fit EV':
            df['Engine HP'][i] = 189
```

    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
df[df['Model'] == 'Fit EV']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4705</th>
      <td>Honda</td>
      <td>Fit EV</td>
      <td>2013</td>
      <td>electric</td>
      <td>189.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>105</td>
      <td>132</td>
      <td>2202</td>
      <td>36625</td>
    </tr>
    <tr>
      <th>4706</th>
      <td>Honda</td>
      <td>Fit EV</td>
      <td>2014</td>
      <td>electric</td>
      <td>189.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>105</td>
      <td>132</td>
      <td>2202</td>
      <td>36625</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Number is still going down
df['Engine HP'].isnull().sum()
```




    56



### model Focus null values 
Fill in Engine HP


```python
df[df['Model'] == 'Focus']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4780</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2015</td>
      <td>flex-fuel (unleaded/E85)</td>
      <td>160.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>36</td>
      <td>26</td>
      <td>5657</td>
      <td>18460</td>
    </tr>
    <tr>
      <th>4781</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2015</td>
      <td>flex-fuel (unleaded/E85)</td>
      <td>160.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>36</td>
      <td>26</td>
      <td>5657</td>
      <td>18960</td>
    </tr>
    <tr>
      <th>4782</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2015</td>
      <td>flex-fuel (unleaded/E85)</td>
      <td>160.0</td>
      <td>4.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>40</td>
      <td>27</td>
      <td>5657</td>
      <td>23170</td>
    </tr>
    <tr>
      <th>4783</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2015</td>
      <td>flex-fuel (unleaded/E85)</td>
      <td>160.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>36</td>
      <td>26</td>
      <td>5657</td>
      <td>17170</td>
    </tr>
    <tr>
      <th>4784</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2015</td>
      <td>flex-fuel (unleaded/E85)</td>
      <td>160.0</td>
      <td>4.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>40</td>
      <td>27</td>
      <td>5657</td>
      <td>23670</td>
    </tr>
    <tr>
      <th>4785</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>99</td>
      <td>110</td>
      <td>5657</td>
      <td>29170</td>
    </tr>
    <tr>
      <th>4786</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2016</td>
      <td>flex-fuel (unleaded/E85)</td>
      <td>160.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>36</td>
      <td>26</td>
      <td>5657</td>
      <td>18515</td>
    </tr>
    <tr>
      <th>4787</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2016</td>
      <td>flex-fuel (unleaded/E85)</td>
      <td>160.0</td>
      <td>4.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>40</td>
      <td>27</td>
      <td>5657</td>
      <td>23725</td>
    </tr>
    <tr>
      <th>4788</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2016</td>
      <td>flex-fuel (unleaded/E85)</td>
      <td>160.0</td>
      <td>4.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>40</td>
      <td>27</td>
      <td>5657</td>
      <td>23225</td>
    </tr>
    <tr>
      <th>4789</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>99</td>
      <td>110</td>
      <td>5657</td>
      <td>29170</td>
    </tr>
    <tr>
      <th>4790</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2016</td>
      <td>flex-fuel (unleaded/E85)</td>
      <td>160.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>36</td>
      <td>26</td>
      <td>5657</td>
      <td>19015</td>
    </tr>
    <tr>
      <th>4791</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2016</td>
      <td>flex-fuel (unleaded/E85)</td>
      <td>160.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>36</td>
      <td>26</td>
      <td>5657</td>
      <td>17225</td>
    </tr>
    <tr>
      <th>4792</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2017</td>
      <td>flex-fuel (unleaded/E85)</td>
      <td>160.0</td>
      <td>4.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>40</td>
      <td>27</td>
      <td>5657</td>
      <td>21675</td>
    </tr>
    <tr>
      <th>4793</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>123.0</td>
      <td>3.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>42</td>
      <td>30</td>
      <td>5657</td>
      <td>18175</td>
    </tr>
    <tr>
      <th>4794</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2017</td>
      <td>flex-fuel (unleaded/E85)</td>
      <td>160.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>36</td>
      <td>26</td>
      <td>5657</td>
      <td>16775</td>
    </tr>
    <tr>
      <th>4795</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2017</td>
      <td>flex-fuel (unleaded/E85)</td>
      <td>160.0</td>
      <td>4.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>40</td>
      <td>27</td>
      <td>5657</td>
      <td>24075</td>
    </tr>
    <tr>
      <th>4796</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2017</td>
      <td>flex-fuel (unleaded/E85)</td>
      <td>160.0</td>
      <td>4.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>40</td>
      <td>27</td>
      <td>5657</td>
      <td>23575</td>
    </tr>
    <tr>
      <th>4797</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2017</td>
      <td>flex-fuel (unleaded/E85)</td>
      <td>160.0</td>
      <td>4.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>40</td>
      <td>27</td>
      <td>5657</td>
      <td>21175</td>
    </tr>
    <tr>
      <th>4798</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2017</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>99</td>
      <td>110</td>
      <td>5657</td>
      <td>29120</td>
    </tr>
    <tr>
      <th>4799</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2017</td>
      <td>flex-fuel (unleaded/E85)</td>
      <td>160.0</td>
      <td>4.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>40</td>
      <td>27</td>
      <td>5657</td>
      <td>19765</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i, j in df['Engine HP'].iteritems():
    if np.isnan(df['Engine HP'][i]):
#         print (i,j)
        if df['Model'][i] == 'Focus':
            df['Engine HP'][i] = 143
```

    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
# Number is still going down
df['Engine HP'].isnull().sum()
```




    53



### model Freestar null values 
Fill in Engine HP


```python
df[df['Model'] == 'Freestar']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4914</th>
      <td>Ford</td>
      <td>Freestar</td>
      <td>2005</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Passenger Minivan</td>
      <td>22</td>
      <td>16</td>
      <td>5657</td>
      <td>28030</td>
    </tr>
    <tr>
      <th>4915</th>
      <td>Ford</td>
      <td>Freestar</td>
      <td>2005</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Passenger Minivan</td>
      <td>22</td>
      <td>16</td>
      <td>5657</td>
      <td>23930</td>
    </tr>
    <tr>
      <th>4916</th>
      <td>Ford</td>
      <td>Freestar</td>
      <td>2005</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Cargo Minivan</td>
      <td>22</td>
      <td>16</td>
      <td>5657</td>
      <td>21630</td>
    </tr>
    <tr>
      <th>4917</th>
      <td>Ford</td>
      <td>Freestar</td>
      <td>2005</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Passenger Minivan</td>
      <td>22</td>
      <td>16</td>
      <td>5657</td>
      <td>26530</td>
    </tr>
    <tr>
      <th>4918</th>
      <td>Ford</td>
      <td>Freestar</td>
      <td>2005</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Passenger Minivan</td>
      <td>21</td>
      <td>16</td>
      <td>5657</td>
      <td>29030</td>
    </tr>
    <tr>
      <th>4919</th>
      <td>Ford</td>
      <td>Freestar</td>
      <td>2005</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Passenger Minivan</td>
      <td>21</td>
      <td>16</td>
      <td>5657</td>
      <td>32755</td>
    </tr>
    <tr>
      <th>4920</th>
      <td>Ford</td>
      <td>Freestar</td>
      <td>2006</td>
      <td>regular unleaded</td>
      <td>193.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Cargo Minivan</td>
      <td>22</td>
      <td>16</td>
      <td>5657</td>
      <td>19650</td>
    </tr>
    <tr>
      <th>4921</th>
      <td>Ford</td>
      <td>Freestar</td>
      <td>2006</td>
      <td>regular unleaded</td>
      <td>201.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Passenger Minivan</td>
      <td>21</td>
      <td>15</td>
      <td>5657</td>
      <td>29575</td>
    </tr>
    <tr>
      <th>4922</th>
      <td>Ford</td>
      <td>Freestar</td>
      <td>2006</td>
      <td>regular unleaded</td>
      <td>193.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Passenger Minivan</td>
      <td>22</td>
      <td>16</td>
      <td>5657</td>
      <td>23655</td>
    </tr>
    <tr>
      <th>4923</th>
      <td>Ford</td>
      <td>Freestar</td>
      <td>2006</td>
      <td>regular unleaded</td>
      <td>201.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Passenger Minivan</td>
      <td>21</td>
      <td>15</td>
      <td>5657</td>
      <td>26615</td>
    </tr>
    <tr>
      <th>4924</th>
      <td>Ford</td>
      <td>Freestar</td>
      <td>2007</td>
      <td>regular unleaded</td>
      <td>201.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Passenger Minivan</td>
      <td>21</td>
      <td>15</td>
      <td>5657</td>
      <td>26665</td>
    </tr>
    <tr>
      <th>4925</th>
      <td>Ford</td>
      <td>Freestar</td>
      <td>2007</td>
      <td>regular unleaded</td>
      <td>201.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Passenger Minivan</td>
      <td>21</td>
      <td>15</td>
      <td>5657</td>
      <td>23705</td>
    </tr>
    <tr>
      <th>4926</th>
      <td>Ford</td>
      <td>Freestar</td>
      <td>2007</td>
      <td>regular unleaded</td>
      <td>201.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Passenger Minivan</td>
      <td>21</td>
      <td>15</td>
      <td>5657</td>
      <td>29575</td>
    </tr>
    <tr>
      <th>4927</th>
      <td>Ford</td>
      <td>Freestar</td>
      <td>2007</td>
      <td>regular unleaded</td>
      <td>201.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Cargo Minivan</td>
      <td>21</td>
      <td>15</td>
      <td>5657</td>
      <td>19700</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i, j in df['Engine HP'].iteritems():
    if np.isnan(df['Engine HP'][i]):
#         print (i,j)
        if df['Model'][i] == 'Freestar':
            df['Engine HP'][i] = 201
```

    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
# Number is still going down
df['Engine HP'].isnull().sum()
```




    47



### model I-MIEV null values 
Fill in Engine HP


```python
df[df['Model'] == 'i-MiEV']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5778</th>
      <td>Mitsubishi</td>
      <td>i-MiEV</td>
      <td>2014</td>
      <td>electric</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>99</td>
      <td>126</td>
      <td>436</td>
      <td>22995</td>
    </tr>
    <tr>
      <th>5779</th>
      <td>Mitsubishi</td>
      <td>i-MiEV</td>
      <td>2016</td>
      <td>electric</td>
      <td>66.0</td>
      <td>NaN</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>99</td>
      <td>126</td>
      <td>436</td>
      <td>22995</td>
    </tr>
    <tr>
      <th>5780</th>
      <td>Mitsubishi</td>
      <td>i-MiEV</td>
      <td>2017</td>
      <td>electric</td>
      <td>66.0</td>
      <td>NaN</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>102</td>
      <td>121</td>
      <td>436</td>
      <td>22995</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i, j in df['Engine HP'].iteritems():
    if np.isnan(df['Engine HP'][i]):
#         print (i,j)
        if df['Model'][i] == 'i-MiEV':
            df['Engine HP'][i] = 201
```

    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
# Number is still going down
df['Engine HP'].isnull().sum()
```




    46



### model Impala null values 
Fill in Engine HP


```python
# All years have the same HP for flex fuel v6
df[df['Model'] == 'Impala']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5823</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>195.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>31</td>
      <td>21</td>
      <td>1385</td>
      <td>34465</td>
    </tr>
    <tr>
      <th>5824</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>195.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>31</td>
      <td>21</td>
      <td>1385</td>
      <td>27060</td>
    </tr>
    <tr>
      <th>5825</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2015</td>
      <td>flex-fuel (unleaded/natural gas)</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>1385</td>
      <td>40660</td>
    </tr>
    <tr>
      <th>5826</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2015</td>
      <td>flex-fuel (unleaded/E85)</td>
      <td>305.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>29</td>
      <td>19</td>
      <td>1385</td>
      <td>35440</td>
    </tr>
    <tr>
      <th>5827</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2015</td>
      <td>flex-fuel (unleaded/E85)</td>
      <td>305.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>29</td>
      <td>19</td>
      <td>1385</td>
      <td>30285</td>
    </tr>
    <tr>
      <th>5828</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>195.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>31</td>
      <td>21</td>
      <td>1385</td>
      <td>29310</td>
    </tr>
    <tr>
      <th>5830</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2015</td>
      <td>flex-fuel (unleaded/natural gas)</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>1385</td>
      <td>37535</td>
    </tr>
    <tr>
      <th>5831</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2016</td>
      <td>flex-fuel (unleaded/natural gas)</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>1385</td>
      <td>40810</td>
    </tr>
    <tr>
      <th>5832</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2016</td>
      <td>flex-fuel (unleaded/E85)</td>
      <td>305.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>29</td>
      <td>19</td>
      <td>1385</td>
      <td>35540</td>
    </tr>
    <tr>
      <th>5833</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2016</td>
      <td>flex-fuel (unleaded/natural gas)</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>1385</td>
      <td>37570</td>
    </tr>
    <tr>
      <th>5834</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2016</td>
      <td>flex-fuel (unleaded/E85)</td>
      <td>305.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>29</td>
      <td>19</td>
      <td>1385</td>
      <td>30435</td>
    </tr>
    <tr>
      <th>5835</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>195.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>31</td>
      <td>22</td>
      <td>1385</td>
      <td>27095</td>
    </tr>
    <tr>
      <th>5836</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>195.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>31</td>
      <td>22</td>
      <td>1385</td>
      <td>29460</td>
    </tr>
    <tr>
      <th>5838</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2017</td>
      <td>flex-fuel (unleaded/E85)</td>
      <td>305.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>28</td>
      <td>19</td>
      <td>1385</td>
      <td>35645</td>
    </tr>
    <tr>
      <th>5839</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2017</td>
      <td>flex-fuel (unleaded/natural gas)</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>1385</td>
      <td>37675</td>
    </tr>
    <tr>
      <th>5840</th>
      <td>Chevrolet</td>
      <td>Impala</td>
      <td>2017</td>
      <td>flex-fuel (unleaded/natural gas)</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>1385</td>
      <td>40915</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i, j in df['Engine HP'].iteritems():
    if np.isnan(df['Engine HP'][i]):
        if df['Model'][i] == 'Impala':
            df['Engine HP'][i] = 305
```

    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
# Number is still going down
df['Engine HP'].isnull().sum()
```




    40



### model Leaf null values 
Fill in Engine HP


```python
df[df['Model'] == 'Leaf']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6385</th>
      <td>Nissan</td>
      <td>Leaf</td>
      <td>2014</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>101</td>
      <td>126</td>
      <td>2009</td>
      <td>35020</td>
    </tr>
    <tr>
      <th>6386</th>
      <td>Nissan</td>
      <td>Leaf</td>
      <td>2014</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>101</td>
      <td>126</td>
      <td>2009</td>
      <td>32000</td>
    </tr>
    <tr>
      <th>6387</th>
      <td>Nissan</td>
      <td>Leaf</td>
      <td>2014</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>101</td>
      <td>126</td>
      <td>2009</td>
      <td>28980</td>
    </tr>
    <tr>
      <th>6388</th>
      <td>Nissan</td>
      <td>Leaf</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>101</td>
      <td>126</td>
      <td>2009</td>
      <td>32100</td>
    </tr>
    <tr>
      <th>6389</th>
      <td>Nissan</td>
      <td>Leaf</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>101</td>
      <td>126</td>
      <td>2009</td>
      <td>35120</td>
    </tr>
    <tr>
      <th>6390</th>
      <td>Nissan</td>
      <td>Leaf</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>101</td>
      <td>126</td>
      <td>2009</td>
      <td>29010</td>
    </tr>
    <tr>
      <th>6391</th>
      <td>Nissan</td>
      <td>Leaf</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>101</td>
      <td>126</td>
      <td>2009</td>
      <td>32000</td>
    </tr>
    <tr>
      <th>6392</th>
      <td>Nissan</td>
      <td>Leaf</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>101</td>
      <td>126</td>
      <td>2009</td>
      <td>29010</td>
    </tr>
    <tr>
      <th>6393</th>
      <td>Nissan</td>
      <td>Leaf</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>101</td>
      <td>124</td>
      <td>2009</td>
      <td>34200</td>
    </tr>
    <tr>
      <th>6394</th>
      <td>Nissan</td>
      <td>Leaf</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>101</td>
      <td>124</td>
      <td>2009</td>
      <td>36790</td>
    </tr>
  </tbody>
</table>
</div>




```python
# All leaf models are 110 HP
for i, j in df['Engine HP'].iteritems():
    if np.isnan(df['Engine HP'][i]):
        if df['Model'][i] == 'Leaf':
            df['Engine HP'][i] = 110
```

    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
# Number is still going down
df['Engine HP'].isnull().sum()
```




    30



### Null value for M-Class


```python
df[df['Model'] == 'M-Class']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6566</th>
      <td>Mercedes-Benz</td>
      <td>M-Class</td>
      <td>2013</td>
      <td>diesel</td>
      <td>240.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>28</td>
      <td>20</td>
      <td>617</td>
      <td>51270</td>
    </tr>
    <tr>
      <th>6567</th>
      <td>Mercedes-Benz</td>
      <td>M-Class</td>
      <td>2013</td>
      <td>premium unleaded (required)</td>
      <td>518.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>17</td>
      <td>13</td>
      <td>617</td>
      <td>96100</td>
    </tr>
    <tr>
      <th>6568</th>
      <td>Mercedes-Benz</td>
      <td>M-Class</td>
      <td>2013</td>
      <td>premium unleaded (required)</td>
      <td>402.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>14</td>
      <td>617</td>
      <td>58800</td>
    </tr>
    <tr>
      <th>6569</th>
      <td>Mercedes-Benz</td>
      <td>M-Class</td>
      <td>2013</td>
      <td>premium unleaded (required)</td>
      <td>302.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>23</td>
      <td>18</td>
      <td>617</td>
      <td>49770</td>
    </tr>
    <tr>
      <th>6570</th>
      <td>Mercedes-Benz</td>
      <td>M-Class</td>
      <td>2013</td>
      <td>premium unleaded (required)</td>
      <td>302.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>23</td>
      <td>18</td>
      <td>617</td>
      <td>47270</td>
    </tr>
    <tr>
      <th>6571</th>
      <td>Mercedes-Benz</td>
      <td>M-Class</td>
      <td>2014</td>
      <td>premium unleaded (required)</td>
      <td>518.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>17</td>
      <td>13</td>
      <td>617</td>
      <td>97250</td>
    </tr>
    <tr>
      <th>6572</th>
      <td>Mercedes-Benz</td>
      <td>M-Class</td>
      <td>2014</td>
      <td>diesel</td>
      <td>240.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>28</td>
      <td>20</td>
      <td>617</td>
      <td>51790</td>
    </tr>
    <tr>
      <th>6573</th>
      <td>Mercedes-Benz</td>
      <td>M-Class</td>
      <td>2014</td>
      <td>premium unleaded (required)</td>
      <td>302.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>22</td>
      <td>17</td>
      <td>617</td>
      <td>50290</td>
    </tr>
    <tr>
      <th>6574</th>
      <td>Mercedes-Benz</td>
      <td>M-Class</td>
      <td>2014</td>
      <td>premium unleaded (required)</td>
      <td>402.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>19</td>
      <td>14</td>
      <td>617</td>
      <td>59450</td>
    </tr>
    <tr>
      <th>6575</th>
      <td>Mercedes-Benz</td>
      <td>M-Class</td>
      <td>2014</td>
      <td>premium unleaded (required)</td>
      <td>302.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>24</td>
      <td>18</td>
      <td>617</td>
      <td>47790</td>
    </tr>
    <tr>
      <th>6576</th>
      <td>Mercedes-Benz</td>
      <td>M-Class</td>
      <td>2015</td>
      <td>premium unleaded (required)</td>
      <td>329.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>22</td>
      <td>18</td>
      <td>617</td>
      <td>62900</td>
    </tr>
    <tr>
      <th>6577</th>
      <td>Mercedes-Benz</td>
      <td>M-Class</td>
      <td>2015</td>
      <td>premium unleaded (required)</td>
      <td>302.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>24</td>
      <td>18</td>
      <td>617</td>
      <td>48300</td>
    </tr>
    <tr>
      <th>6578</th>
      <td>Mercedes-Benz</td>
      <td>M-Class</td>
      <td>2015</td>
      <td>diesel</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>29</td>
      <td>22</td>
      <td>617</td>
      <td>49800</td>
    </tr>
    <tr>
      <th>6579</th>
      <td>Mercedes-Benz</td>
      <td>M-Class</td>
      <td>2015</td>
      <td>premium unleaded (required)</td>
      <td>518.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>17</td>
      <td>13</td>
      <td>617</td>
      <td>98400</td>
    </tr>
    <tr>
      <th>6580</th>
      <td>Mercedes-Benz</td>
      <td>M-Class</td>
      <td>2015</td>
      <td>premium unleaded (required)</td>
      <td>302.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>22</td>
      <td>17</td>
      <td>617</td>
      <td>50800</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i, j in df['Engine HP'].iteritems():
    if np.isnan(df['Engine HP'][i]):
        if df['Model'][i] == 'M-Class':
            df['Engine HP'][i] = 240
```

    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
# Number is still going down
df['Engine HP'].isnull().sum()
```




    29



### Model MKZ
Fill in null Engine HP


```python
# Only difference are the fuel type and Drivn_wheels
# 2.0L version can be front or all wheel drive 
# Would still use 245 HP for year 2017
df[df['Model'] == 'MKZ']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6896</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>231.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>33</td>
      <td>22</td>
      <td>61</td>
      <td>45555</td>
    </tr>
    <tr>
      <th>6897</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>231.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>33</td>
      <td>22</td>
      <td>61</td>
      <td>35190</td>
    </tr>
    <tr>
      <th>6898</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>188.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>39</td>
      <td>41</td>
      <td>61</td>
      <td>35190</td>
    </tr>
    <tr>
      <th>6899</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>231.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>31</td>
      <td>22</td>
      <td>61</td>
      <td>37080</td>
    </tr>
    <tr>
      <th>6900</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>188.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>39</td>
      <td>41</td>
      <td>61</td>
      <td>45555</td>
    </tr>
    <tr>
      <th>6901</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2015</td>
      <td>regular unleaded</td>
      <td>231.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>31</td>
      <td>22</td>
      <td>61</td>
      <td>47445</td>
    </tr>
    <tr>
      <th>6902</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>231.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>31</td>
      <td>22</td>
      <td>61</td>
      <td>37080</td>
    </tr>
    <tr>
      <th>6903</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>188.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>39</td>
      <td>41</td>
      <td>61</td>
      <td>45605</td>
    </tr>
    <tr>
      <th>6904</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>231.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>33</td>
      <td>22</td>
      <td>61</td>
      <td>45605</td>
    </tr>
    <tr>
      <th>6905</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>231.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>31</td>
      <td>22</td>
      <td>61</td>
      <td>47495</td>
    </tr>
    <tr>
      <th>6906</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>188.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>39</td>
      <td>41</td>
      <td>61</td>
      <td>35190</td>
    </tr>
    <tr>
      <th>6907</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2016</td>
      <td>regular unleaded</td>
      <td>231.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>33</td>
      <td>22</td>
      <td>61</td>
      <td>35190</td>
    </tr>
    <tr>
      <th>6908</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>38</td>
      <td>41</td>
      <td>61</td>
      <td>35010</td>
    </tr>
    <tr>
      <th>6909</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2017</td>
      <td>premium unleaded (recommended)</td>
      <td>245.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>28</td>
      <td>20</td>
      <td>61</td>
      <td>36900</td>
    </tr>
    <tr>
      <th>6910</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>38</td>
      <td>41</td>
      <td>61</td>
      <td>39510</td>
    </tr>
    <tr>
      <th>6911</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2017</td>
      <td>premium unleaded (recommended)</td>
      <td>245.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>31</td>
      <td>21</td>
      <td>61</td>
      <td>39510</td>
    </tr>
    <tr>
      <th>6912</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2017</td>
      <td>premium unleaded (recommended)</td>
      <td>245.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>28</td>
      <td>20</td>
      <td>61</td>
      <td>49560</td>
    </tr>
    <tr>
      <th>6913</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2017</td>
      <td>premium unleaded (recommended)</td>
      <td>245.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>31</td>
      <td>21</td>
      <td>61</td>
      <td>35010</td>
    </tr>
    <tr>
      <th>6914</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2017</td>
      <td>premium unleaded (recommended)</td>
      <td>245.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>28</td>
      <td>20</td>
      <td>61</td>
      <td>41400</td>
    </tr>
    <tr>
      <th>6915</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2017</td>
      <td>premium unleaded (recommended)</td>
      <td>245.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>31</td>
      <td>21</td>
      <td>61</td>
      <td>47670</td>
    </tr>
    <tr>
      <th>6916</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>38</td>
      <td>41</td>
      <td>61</td>
      <td>36760</td>
    </tr>
    <tr>
      <th>6917</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2017</td>
      <td>premium unleaded (recommended)</td>
      <td>245.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>31</td>
      <td>21</td>
      <td>61</td>
      <td>36760</td>
    </tr>
    <tr>
      <th>6918</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2017</td>
      <td>regular unleaded</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>38</td>
      <td>41</td>
      <td>61</td>
      <td>47670</td>
    </tr>
    <tr>
      <th>6919</th>
      <td>Lincoln</td>
      <td>MKZ</td>
      <td>2017</td>
      <td>premium unleaded (recommended)</td>
      <td>245.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>28</td>
      <td>20</td>
      <td>61</td>
      <td>38650</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i, j in df['Engine HP'].iteritems():
    if np.isnan(df['Engine HP'][i]):
#         print (i,j)
        if df['Model'][i] == 'MKZ':
            df['Engine HP'][i] = 245
```

    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
# Number is still going down
df['Engine HP'].isnull().sum()
```




    25



### Model RAV4 EV
Fill in null Engine HP


```python
# All model years are the same HP
df[df['Model'] == 'RAV4 EV']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8373</th>
      <td>Toyota</td>
      <td>RAV4 EV</td>
      <td>2012</td>
      <td>electric</td>
      <td>154.0</td>
      <td>NaN</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>74</td>
      <td>78</td>
      <td>2031</td>
      <td>49800</td>
    </tr>
    <tr>
      <th>8374</th>
      <td>Toyota</td>
      <td>RAV4 EV</td>
      <td>2013</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>74</td>
      <td>78</td>
      <td>2031</td>
      <td>49800</td>
    </tr>
    <tr>
      <th>8375</th>
      <td>Toyota</td>
      <td>RAV4 EV</td>
      <td>2014</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>74</td>
      <td>78</td>
      <td>2031</td>
      <td>49800</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i, j in df['Engine HP'].iteritems():
    if np.isnan(df['Engine HP'][i]):
        if df['Model'][i] == 'RAV4 EV':
            df['Engine HP'][i] = 154
```

    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
# Number is still going down
df['Engine HP'].isnull().sum()
```




    23



### Model Soul S
Fill in null Engine HP


```python
df[df['Model'] == 'Soul EV']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9850</th>
      <td>Kia</td>
      <td>Soul EV</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Wagon</td>
      <td>92</td>
      <td>120</td>
      <td>1720</td>
      <td>35700</td>
    </tr>
    <tr>
      <th>9851</th>
      <td>Kia</td>
      <td>Soul EV</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Wagon</td>
      <td>92</td>
      <td>120</td>
      <td>1720</td>
      <td>33700</td>
    </tr>
    <tr>
      <th>9852</th>
      <td>Kia</td>
      <td>Soul EV</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Wagon</td>
      <td>92</td>
      <td>120</td>
      <td>1720</td>
      <td>33950</td>
    </tr>
    <tr>
      <th>9853</th>
      <td>Kia</td>
      <td>Soul EV</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Wagon</td>
      <td>92</td>
      <td>120</td>
      <td>1720</td>
      <td>31950</td>
    </tr>
    <tr>
      <th>9854</th>
      <td>Kia</td>
      <td>Soul EV</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Wagon</td>
      <td>92</td>
      <td>120</td>
      <td>1720</td>
      <td>35950</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i, j in df['Engine HP'].iteritems():
    if np.isnan(df['Engine HP'][i]):
        if df['Model'][i] == 'Soul EV':
            df['Engine HP'][i] = 109
```

    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
# Number is still going down
df['Engine HP'].isnull().sum()
```




    18



### Model S
Fill in null Engine HP


```python
df[df['Model'] == 'Model S']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6921</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2014</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>90</td>
      <td>88</td>
      <td>1391</td>
      <td>79900</td>
    </tr>
    <tr>
      <th>6922</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2014</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>97</td>
      <td>94</td>
      <td>1391</td>
      <td>69900</td>
    </tr>
    <tr>
      <th>6923</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2014</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>94</td>
      <td>86</td>
      <td>1391</td>
      <td>104500</td>
    </tr>
    <tr>
      <th>6924</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2014</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>90</td>
      <td>88</td>
      <td>1391</td>
      <td>93400</td>
    </tr>
    <tr>
      <th>6925</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>97</td>
      <td>94</td>
      <td>1391</td>
      <td>69900</td>
    </tr>
    <tr>
      <th>6926</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>102</td>
      <td>101</td>
      <td>1391</td>
      <td>75000</td>
    </tr>
    <tr>
      <th>6927</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>106</td>
      <td>95</td>
      <td>1391</td>
      <td>85000</td>
    </tr>
    <tr>
      <th>6928</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>98</td>
      <td>89</td>
      <td>1391</td>
      <td>105000</td>
    </tr>
    <tr>
      <th>6929</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>90</td>
      <td>88</td>
      <td>1391</td>
      <td>80000</td>
    </tr>
    <tr>
      <th>6930</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>105</td>
      <td>102</td>
      <td>1391</td>
      <td>79500</td>
    </tr>
    <tr>
      <th>6931</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>101</td>
      <td>98</td>
      <td>1391</td>
      <td>66000</td>
    </tr>
    <tr>
      <th>6932</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>105</td>
      <td>92</td>
      <td>1391</td>
      <td>134500</td>
    </tr>
    <tr>
      <th>6933</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>100</td>
      <td>97</td>
      <td>1391</td>
      <td>74500</td>
    </tr>
    <tr>
      <th>6934</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>107</td>
      <td>101</td>
      <td>1391</td>
      <td>71000</td>
    </tr>
    <tr>
      <th>6935</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>102</td>
      <td>101</td>
      <td>1391</td>
      <td>75000</td>
    </tr>
    <tr>
      <th>6936</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>107</td>
      <td>101</td>
      <td>1391</td>
      <td>89500</td>
    </tr>
    <tr>
      <th>6937</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>100</td>
      <td>91</td>
      <td>1391</td>
      <td>112000</td>
    </tr>
    <tr>
      <th>6938</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>90</td>
      <td>88</td>
      <td>1391</td>
      <td>70000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Change model name by the driven wheels

for i, j in df['Model'].iteritems():
    if np.isnan(df['Engine HP'][i]):
#         print (i,j)
        if df['Driven_Wheels'][i] == 'all wheel drive':
            df['Model'][i] = 'Model D'
        elif df['Driven_Wheels'][i] == 'rear wheel drive':
            df['Model'][i] = 'Model S'
```

    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
df[df['Model'] == 'Model D']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6923</th>
      <td>Tesla</td>
      <td>Model D</td>
      <td>2014</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>94</td>
      <td>86</td>
      <td>1391</td>
      <td>104500</td>
    </tr>
    <tr>
      <th>6926</th>
      <td>Tesla</td>
      <td>Model D</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>102</td>
      <td>101</td>
      <td>1391</td>
      <td>75000</td>
    </tr>
    <tr>
      <th>6927</th>
      <td>Tesla</td>
      <td>Model D</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>106</td>
      <td>95</td>
      <td>1391</td>
      <td>85000</td>
    </tr>
    <tr>
      <th>6928</th>
      <td>Tesla</td>
      <td>Model D</td>
      <td>2015</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>98</td>
      <td>89</td>
      <td>1391</td>
      <td>105000</td>
    </tr>
    <tr>
      <th>6930</th>
      <td>Tesla</td>
      <td>Model D</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>105</td>
      <td>102</td>
      <td>1391</td>
      <td>79500</td>
    </tr>
    <tr>
      <th>6931</th>
      <td>Tesla</td>
      <td>Model D</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>101</td>
      <td>98</td>
      <td>1391</td>
      <td>66000</td>
    </tr>
    <tr>
      <th>6932</th>
      <td>Tesla</td>
      <td>Model D</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>105</td>
      <td>92</td>
      <td>1391</td>
      <td>134500</td>
    </tr>
    <tr>
      <th>6934</th>
      <td>Tesla</td>
      <td>Model D</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>107</td>
      <td>101</td>
      <td>1391</td>
      <td>71000</td>
    </tr>
    <tr>
      <th>6935</th>
      <td>Tesla</td>
      <td>Model D</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>102</td>
      <td>101</td>
      <td>1391</td>
      <td>75000</td>
    </tr>
    <tr>
      <th>6936</th>
      <td>Tesla</td>
      <td>Model D</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>107</td>
      <td>101</td>
      <td>1391</td>
      <td>89500</td>
    </tr>
    <tr>
      <th>6937</th>
      <td>Tesla</td>
      <td>Model D</td>
      <td>2016</td>
      <td>electric</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>4.0</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>100</td>
      <td>91</td>
      <td>1391</td>
      <td>112000</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i, j in df['Engine HP'].iteritems():
    if np.isnan(df['Engine HP'][i]):
        if df['Model'][i] == 'Model S':
            df['Engine HP'][i] = 362
        elif df['Model'][i] == 'Model D':
            df['Engine HP'][i] = 259
```

    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
# Down to Zero
df['Engine HP'].isnull().sum()
```




    0



<a id='ab-ec'></a>

## NUll values in Engine Cylinders 


```python
# Examine the null values of Engine Cylinder 
df[df['Engine Cylinders'].isnull()]
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1983</th>
      <td>Chevrolet</td>
      <td>Bolt EV</td>
      <td>2017</td>
      <td>electric</td>
      <td>200.0</td>
      <td>NaN</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>110</td>
      <td>128</td>
      <td>1385</td>
      <td>40905</td>
    </tr>
    <tr>
      <th>1984</th>
      <td>Chevrolet</td>
      <td>Bolt EV</td>
      <td>2017</td>
      <td>electric</td>
      <td>200.0</td>
      <td>NaN</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>110</td>
      <td>128</td>
      <td>1385</td>
      <td>36620</td>
    </tr>
    <tr>
      <th>3716</th>
      <td>Volkswagen</td>
      <td>e-Golf</td>
      <td>2015</td>
      <td>electric</td>
      <td>115.0</td>
      <td>NaN</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>105</td>
      <td>126</td>
      <td>873</td>
      <td>33450</td>
    </tr>
    <tr>
      <th>3717</th>
      <td>Volkswagen</td>
      <td>e-Golf</td>
      <td>2015</td>
      <td>electric</td>
      <td>115.0</td>
      <td>NaN</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>105</td>
      <td>126</td>
      <td>873</td>
      <td>35445</td>
    </tr>
    <tr>
      <th>3718</th>
      <td>Volkswagen</td>
      <td>e-Golf</td>
      <td>2016</td>
      <td>electric</td>
      <td>115.0</td>
      <td>NaN</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>105</td>
      <td>126</td>
      <td>873</td>
      <td>28995</td>
    </tr>
    <tr>
      <th>3719</th>
      <td>Volkswagen</td>
      <td>e-Golf</td>
      <td>2016</td>
      <td>electric</td>
      <td>115.0</td>
      <td>NaN</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>105</td>
      <td>126</td>
      <td>873</td>
      <td>35595</td>
    </tr>
    <tr>
      <th>5778</th>
      <td>Mitsubishi</td>
      <td>i-MiEV</td>
      <td>2014</td>
      <td>electric</td>
      <td>201.0</td>
      <td>NaN</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>99</td>
      <td>126</td>
      <td>436</td>
      <td>22995</td>
    </tr>
    <tr>
      <th>5779</th>
      <td>Mitsubishi</td>
      <td>i-MiEV</td>
      <td>2016</td>
      <td>electric</td>
      <td>66.0</td>
      <td>NaN</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>99</td>
      <td>126</td>
      <td>436</td>
      <td>22995</td>
    </tr>
    <tr>
      <th>5780</th>
      <td>Mitsubishi</td>
      <td>i-MiEV</td>
      <td>2017</td>
      <td>electric</td>
      <td>66.0</td>
      <td>NaN</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>102</td>
      <td>121</td>
      <td>436</td>
      <td>22995</td>
    </tr>
    <tr>
      <th>8373</th>
      <td>Toyota</td>
      <td>RAV4 EV</td>
      <td>2012</td>
      <td>electric</td>
      <td>154.0</td>
      <td>NaN</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>74</td>
      <td>78</td>
      <td>2031</td>
      <td>49800</td>
    </tr>
    <tr>
      <th>8695</th>
      <td>Mazda</td>
      <td>RX-7</td>
      <td>1993</td>
      <td>regular unleaded</td>
      <td>255.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>15</td>
      <td>586</td>
      <td>7523</td>
    </tr>
    <tr>
      <th>8696</th>
      <td>Mazda</td>
      <td>RX-7</td>
      <td>1994</td>
      <td>regular unleaded</td>
      <td>255.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>15</td>
      <td>586</td>
      <td>8147</td>
    </tr>
    <tr>
      <th>8697</th>
      <td>Mazda</td>
      <td>RX-7</td>
      <td>1995</td>
      <td>regular unleaded</td>
      <td>255.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>15</td>
      <td>586</td>
      <td>8839</td>
    </tr>
    <tr>
      <th>8698</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>31930</td>
    </tr>
    <tr>
      <th>8699</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>212.0</td>
      <td>NaN</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>586</td>
      <td>26435</td>
    </tr>
    <tr>
      <th>8700</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>27860</td>
    </tr>
    <tr>
      <th>8701</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>31000</td>
    </tr>
    <tr>
      <th>8702</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>26435</td>
    </tr>
    <tr>
      <th>8703</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>212.0</td>
      <td>NaN</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>586</td>
      <td>31700</td>
    </tr>
    <tr>
      <th>8704</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>212.0</td>
      <td>NaN</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>586</td>
      <td>28560</td>
    </tr>
    <tr>
      <th>8705</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2010</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>32140</td>
    </tr>
    <tr>
      <th>8706</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2010</td>
      <td>premium unleaded (required)</td>
      <td>212.0</td>
      <td>NaN</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>586</td>
      <td>26645</td>
    </tr>
    <tr>
      <th>8707</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2010</td>
      <td>premium unleaded (required)</td>
      <td>212.0</td>
      <td>NaN</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>586</td>
      <td>32810</td>
    </tr>
    <tr>
      <th>8708</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2010</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>26645</td>
    </tr>
    <tr>
      <th>8709</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2010</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>32110</td>
    </tr>
    <tr>
      <th>8710</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>212.0</td>
      <td>NaN</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>586</td>
      <td>32960</td>
    </tr>
    <tr>
      <th>8711</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>32260</td>
    </tr>
    <tr>
      <th>8712</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>32290</td>
    </tr>
    <tr>
      <th>8713</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>26795</td>
    </tr>
    <tr>
      <th>8714</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>212.0</td>
      <td>NaN</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>586</td>
      <td>26795</td>
    </tr>
  </tbody>
</table>
</div>




```python
# from what I can see this model is a rotary engine which means it have no cylinders 
df[df['Model'] == 'RX-7']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8695</th>
      <td>Mazda</td>
      <td>RX-7</td>
      <td>1993</td>
      <td>regular unleaded</td>
      <td>255.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>15</td>
      <td>586</td>
      <td>7523</td>
    </tr>
    <tr>
      <th>8696</th>
      <td>Mazda</td>
      <td>RX-7</td>
      <td>1994</td>
      <td>regular unleaded</td>
      <td>255.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>15</td>
      <td>586</td>
      <td>8147</td>
    </tr>
    <tr>
      <th>8697</th>
      <td>Mazda</td>
      <td>RX-7</td>
      <td>1995</td>
      <td>regular unleaded</td>
      <td>255.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>15</td>
      <td>586</td>
      <td>8839</td>
    </tr>
  </tbody>
</table>
</div>




```python
# from what I can see this model is a rotary engine which means it have no cylinders 
df[df['Model'] == 'RX-8']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8698</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>31930</td>
    </tr>
    <tr>
      <th>8699</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>212.0</td>
      <td>NaN</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>586</td>
      <td>26435</td>
    </tr>
    <tr>
      <th>8700</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>27860</td>
    </tr>
    <tr>
      <th>8701</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>31000</td>
    </tr>
    <tr>
      <th>8702</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>26435</td>
    </tr>
    <tr>
      <th>8703</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>212.0</td>
      <td>NaN</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>586</td>
      <td>31700</td>
    </tr>
    <tr>
      <th>8704</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>212.0</td>
      <td>NaN</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>586</td>
      <td>28560</td>
    </tr>
    <tr>
      <th>8705</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2010</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>32140</td>
    </tr>
    <tr>
      <th>8706</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2010</td>
      <td>premium unleaded (required)</td>
      <td>212.0</td>
      <td>NaN</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>586</td>
      <td>26645</td>
    </tr>
    <tr>
      <th>8707</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2010</td>
      <td>premium unleaded (required)</td>
      <td>212.0</td>
      <td>NaN</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>586</td>
      <td>32810</td>
    </tr>
    <tr>
      <th>8708</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2010</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>26645</td>
    </tr>
    <tr>
      <th>8709</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2010</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>32110</td>
    </tr>
    <tr>
      <th>8710</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>212.0</td>
      <td>NaN</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>586</td>
      <td>32960</td>
    </tr>
    <tr>
      <th>8711</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>32260</td>
    </tr>
    <tr>
      <th>8712</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>32290</td>
    </tr>
    <tr>
      <th>8713</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>NaN</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>26795</td>
    </tr>
    <tr>
      <th>8714</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>212.0</td>
      <td>NaN</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>586</td>
      <td>26795</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Also no electric cars have cylinders
# was able to repalce all nan values with 0
for i, j in df['Engine Cylinders'].iteritems():
    if np.isnan(df['Engine Cylinders'][i]):
            df['Engine Cylinders'][i] = 0
```

    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
df[df['Engine Cylinders'] <=0]
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>539</th>
      <td>FIAT</td>
      <td>500e</td>
      <td>2015</td>
      <td>electric</td>
      <td>111.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr Hatchback</td>
      <td>108</td>
      <td>122</td>
      <td>819</td>
      <td>31800</td>
    </tr>
    <tr>
      <th>540</th>
      <td>FIAT</td>
      <td>500e</td>
      <td>2016</td>
      <td>electric</td>
      <td>111.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr Hatchback</td>
      <td>103</td>
      <td>121</td>
      <td>819</td>
      <td>31800</td>
    </tr>
    <tr>
      <th>541</th>
      <td>FIAT</td>
      <td>500e</td>
      <td>2017</td>
      <td>electric</td>
      <td>111.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr Hatchback</td>
      <td>103</td>
      <td>121</td>
      <td>819</td>
      <td>31800</td>
    </tr>
    <tr>
      <th>1680</th>
      <td>Mercedes-Benz</td>
      <td>B-Class Electric Drive</td>
      <td>2015</td>
      <td>electric</td>
      <td>177.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>82</td>
      <td>85</td>
      <td>617</td>
      <td>41450</td>
    </tr>
    <tr>
      <th>1681</th>
      <td>Mercedes-Benz</td>
      <td>B-Class Electric Drive</td>
      <td>2016</td>
      <td>electric</td>
      <td>177.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>82</td>
      <td>85</td>
      <td>617</td>
      <td>41450</td>
    </tr>
    <tr>
      <th>1682</th>
      <td>Mercedes-Benz</td>
      <td>B-Class Electric Drive</td>
      <td>2017</td>
      <td>electric</td>
      <td>177.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>82</td>
      <td>85</td>
      <td>617</td>
      <td>39900</td>
    </tr>
    <tr>
      <th>1983</th>
      <td>Chevrolet</td>
      <td>Bolt EV</td>
      <td>2017</td>
      <td>electric</td>
      <td>200.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>110</td>
      <td>128</td>
      <td>1385</td>
      <td>40905</td>
    </tr>
    <tr>
      <th>1984</th>
      <td>Chevrolet</td>
      <td>Bolt EV</td>
      <td>2017</td>
      <td>electric</td>
      <td>200.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>110</td>
      <td>128</td>
      <td>1385</td>
      <td>36620</td>
    </tr>
    <tr>
      <th>3716</th>
      <td>Volkswagen</td>
      <td>e-Golf</td>
      <td>2015</td>
      <td>electric</td>
      <td>115.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>105</td>
      <td>126</td>
      <td>873</td>
      <td>33450</td>
    </tr>
    <tr>
      <th>3717</th>
      <td>Volkswagen</td>
      <td>e-Golf</td>
      <td>2015</td>
      <td>electric</td>
      <td>115.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>105</td>
      <td>126</td>
      <td>873</td>
      <td>35445</td>
    </tr>
    <tr>
      <th>3718</th>
      <td>Volkswagen</td>
      <td>e-Golf</td>
      <td>2016</td>
      <td>electric</td>
      <td>115.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>105</td>
      <td>126</td>
      <td>873</td>
      <td>28995</td>
    </tr>
    <tr>
      <th>3719</th>
      <td>Volkswagen</td>
      <td>e-Golf</td>
      <td>2016</td>
      <td>electric</td>
      <td>115.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>105</td>
      <td>126</td>
      <td>873</td>
      <td>35595</td>
    </tr>
    <tr>
      <th>4705</th>
      <td>Honda</td>
      <td>Fit EV</td>
      <td>2013</td>
      <td>electric</td>
      <td>189.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>105</td>
      <td>132</td>
      <td>2202</td>
      <td>36625</td>
    </tr>
    <tr>
      <th>4706</th>
      <td>Honda</td>
      <td>Fit EV</td>
      <td>2014</td>
      <td>electric</td>
      <td>189.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>105</td>
      <td>132</td>
      <td>2202</td>
      <td>36625</td>
    </tr>
    <tr>
      <th>4785</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2015</td>
      <td>electric</td>
      <td>143.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>99</td>
      <td>110</td>
      <td>5657</td>
      <td>29170</td>
    </tr>
    <tr>
      <th>4789</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2016</td>
      <td>electric</td>
      <td>143.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>99</td>
      <td>110</td>
      <td>5657</td>
      <td>29170</td>
    </tr>
    <tr>
      <th>4798</th>
      <td>Ford</td>
      <td>Focus</td>
      <td>2017</td>
      <td>electric</td>
      <td>143.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>99</td>
      <td>110</td>
      <td>5657</td>
      <td>29120</td>
    </tr>
    <tr>
      <th>5778</th>
      <td>Mitsubishi</td>
      <td>i-MiEV</td>
      <td>2014</td>
      <td>electric</td>
      <td>201.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>99</td>
      <td>126</td>
      <td>436</td>
      <td>22995</td>
    </tr>
    <tr>
      <th>5779</th>
      <td>Mitsubishi</td>
      <td>i-MiEV</td>
      <td>2016</td>
      <td>electric</td>
      <td>66.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>99</td>
      <td>126</td>
      <td>436</td>
      <td>22995</td>
    </tr>
    <tr>
      <th>5780</th>
      <td>Mitsubishi</td>
      <td>i-MiEV</td>
      <td>2017</td>
      <td>electric</td>
      <td>66.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>102</td>
      <td>121</td>
      <td>436</td>
      <td>22995</td>
    </tr>
    <tr>
      <th>5790</th>
      <td>BMW</td>
      <td>i3</td>
      <td>2015</td>
      <td>electric</td>
      <td>170.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>111</td>
      <td>137</td>
      <td>3916</td>
      <td>42400</td>
    </tr>
    <tr>
      <th>5791</th>
      <td>BMW</td>
      <td>i3</td>
      <td>2016</td>
      <td>electric</td>
      <td>170.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>111</td>
      <td>137</td>
      <td>3916</td>
      <td>42400</td>
    </tr>
    <tr>
      <th>5792</th>
      <td>BMW</td>
      <td>i3</td>
      <td>2017</td>
      <td>electric</td>
      <td>170.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>111</td>
      <td>137</td>
      <td>3916</td>
      <td>42400</td>
    </tr>
    <tr>
      <th>5793</th>
      <td>BMW</td>
      <td>i3</td>
      <td>2017</td>
      <td>electric</td>
      <td>170.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>106</td>
      <td>129</td>
      <td>3916</td>
      <td>43600</td>
    </tr>
    <tr>
      <th>6385</th>
      <td>Nissan</td>
      <td>Leaf</td>
      <td>2014</td>
      <td>electric</td>
      <td>110.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>101</td>
      <td>126</td>
      <td>2009</td>
      <td>35020</td>
    </tr>
    <tr>
      <th>6386</th>
      <td>Nissan</td>
      <td>Leaf</td>
      <td>2014</td>
      <td>electric</td>
      <td>110.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>101</td>
      <td>126</td>
      <td>2009</td>
      <td>32000</td>
    </tr>
    <tr>
      <th>6387</th>
      <td>Nissan</td>
      <td>Leaf</td>
      <td>2014</td>
      <td>electric</td>
      <td>110.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>101</td>
      <td>126</td>
      <td>2009</td>
      <td>28980</td>
    </tr>
    <tr>
      <th>6388</th>
      <td>Nissan</td>
      <td>Leaf</td>
      <td>2015</td>
      <td>electric</td>
      <td>110.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>101</td>
      <td>126</td>
      <td>2009</td>
      <td>32100</td>
    </tr>
    <tr>
      <th>6389</th>
      <td>Nissan</td>
      <td>Leaf</td>
      <td>2015</td>
      <td>electric</td>
      <td>110.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>101</td>
      <td>126</td>
      <td>2009</td>
      <td>35120</td>
    </tr>
    <tr>
      <th>6390</th>
      <td>Nissan</td>
      <td>Leaf</td>
      <td>2015</td>
      <td>electric</td>
      <td>110.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>101</td>
      <td>126</td>
      <td>2009</td>
      <td>29010</td>
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
      <th>8696</th>
      <td>Mazda</td>
      <td>RX-7</td>
      <td>1994</td>
      <td>regular unleaded</td>
      <td>255.0</td>
      <td>0.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>15</td>
      <td>586</td>
      <td>8147</td>
    </tr>
    <tr>
      <th>8697</th>
      <td>Mazda</td>
      <td>RX-7</td>
      <td>1995</td>
      <td>regular unleaded</td>
      <td>255.0</td>
      <td>0.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>15</td>
      <td>586</td>
      <td>8839</td>
    </tr>
    <tr>
      <th>8698</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>0.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>31930</td>
    </tr>
    <tr>
      <th>8699</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>212.0</td>
      <td>0.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>586</td>
      <td>26435</td>
    </tr>
    <tr>
      <th>8700</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>0.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>27860</td>
    </tr>
    <tr>
      <th>8701</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>0.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>31000</td>
    </tr>
    <tr>
      <th>8702</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>0.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>26435</td>
    </tr>
    <tr>
      <th>8703</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>212.0</td>
      <td>0.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>586</td>
      <td>31700</td>
    </tr>
    <tr>
      <th>8704</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>212.0</td>
      <td>0.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>586</td>
      <td>28560</td>
    </tr>
    <tr>
      <th>8705</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2010</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>0.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>32140</td>
    </tr>
    <tr>
      <th>8706</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2010</td>
      <td>premium unleaded (required)</td>
      <td>212.0</td>
      <td>0.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>586</td>
      <td>26645</td>
    </tr>
    <tr>
      <th>8707</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2010</td>
      <td>premium unleaded (required)</td>
      <td>212.0</td>
      <td>0.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>586</td>
      <td>32810</td>
    </tr>
    <tr>
      <th>8708</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2010</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>0.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>26645</td>
    </tr>
    <tr>
      <th>8709</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2010</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>0.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>32110</td>
    </tr>
    <tr>
      <th>8710</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>212.0</td>
      <td>0.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>586</td>
      <td>32960</td>
    </tr>
    <tr>
      <th>8711</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>0.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>32260</td>
    </tr>
    <tr>
      <th>8712</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>0.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>32290</td>
    </tr>
    <tr>
      <th>8713</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>232.0</td>
      <td>0.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>22</td>
      <td>16</td>
      <td>586</td>
      <td>26795</td>
    </tr>
    <tr>
      <th>8714</th>
      <td>Mazda</td>
      <td>RX-8</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>212.0</td>
      <td>0.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>586</td>
      <td>26795</td>
    </tr>
    <tr>
      <th>9850</th>
      <td>Kia</td>
      <td>Soul EV</td>
      <td>2015</td>
      <td>electric</td>
      <td>109.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Wagon</td>
      <td>92</td>
      <td>120</td>
      <td>1720</td>
      <td>35700</td>
    </tr>
    <tr>
      <th>9851</th>
      <td>Kia</td>
      <td>Soul EV</td>
      <td>2015</td>
      <td>electric</td>
      <td>109.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Wagon</td>
      <td>92</td>
      <td>120</td>
      <td>1720</td>
      <td>33700</td>
    </tr>
    <tr>
      <th>9852</th>
      <td>Kia</td>
      <td>Soul EV</td>
      <td>2016</td>
      <td>electric</td>
      <td>109.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Wagon</td>
      <td>92</td>
      <td>120</td>
      <td>1720</td>
      <td>33950</td>
    </tr>
    <tr>
      <th>9853</th>
      <td>Kia</td>
      <td>Soul EV</td>
      <td>2016</td>
      <td>electric</td>
      <td>109.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Wagon</td>
      <td>92</td>
      <td>120</td>
      <td>1720</td>
      <td>31950</td>
    </tr>
    <tr>
      <th>9854</th>
      <td>Kia</td>
      <td>Soul EV</td>
      <td>2016</td>
      <td>electric</td>
      <td>109.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Wagon</td>
      <td>92</td>
      <td>120</td>
      <td>1720</td>
      <td>35950</td>
    </tr>
    <tr>
      <th>9867</th>
      <td>Chevrolet</td>
      <td>Spark EV</td>
      <td>2014</td>
      <td>electric</td>
      <td>140.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>109</td>
      <td>128</td>
      <td>1385</td>
      <td>26685</td>
    </tr>
    <tr>
      <th>9868</th>
      <td>Chevrolet</td>
      <td>Spark EV</td>
      <td>2014</td>
      <td>electric</td>
      <td>140.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>109</td>
      <td>128</td>
      <td>1385</td>
      <td>27010</td>
    </tr>
    <tr>
      <th>9869</th>
      <td>Chevrolet</td>
      <td>Spark EV</td>
      <td>2015</td>
      <td>electric</td>
      <td>140.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>109</td>
      <td>128</td>
      <td>1385</td>
      <td>25170</td>
    </tr>
    <tr>
      <th>9870</th>
      <td>Chevrolet</td>
      <td>Spark EV</td>
      <td>2015</td>
      <td>electric</td>
      <td>140.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>109</td>
      <td>128</td>
      <td>1385</td>
      <td>25560</td>
    </tr>
    <tr>
      <th>9871</th>
      <td>Chevrolet</td>
      <td>Spark EV</td>
      <td>2016</td>
      <td>electric</td>
      <td>140.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>109</td>
      <td>128</td>
      <td>1385</td>
      <td>25510</td>
    </tr>
    <tr>
      <th>9872</th>
      <td>Chevrolet</td>
      <td>Spark EV</td>
      <td>2016</td>
      <td>electric</td>
      <td>140.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>4dr Hatchback</td>
      <td>109</td>
      <td>128</td>
      <td>1385</td>
      <td>25120</td>
    </tr>
  </tbody>
</table>
<p>86 rows × 15 columns</p>
</div>




```python
df.groupby(['Make','Model', 'Engine Cylinders']).count()
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
      <th></th>
      <th></th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
    <tr>
      <th>Make</th>
      <th>Model</th>
      <th>Engine Cylinders</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th rowspan="20" valign="top">Acura</th>
      <th>CL</th>
      <th>6.0</th>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>ILX</th>
      <th>4.0</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>ILX Hybrid</th>
      <th>4.0</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Integra</th>
      <th>4.0</th>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
    </tr>
    <tr>
      <th>Legend</th>
      <th>6.0</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>MDX</th>
      <th>6.0</th>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
    </tr>
    <tr>
      <th>NSX</th>
      <th>6.0</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>RDX</th>
      <th>6.0</th>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
    </tr>
    <tr>
      <th>RL</th>
      <th>6.0</th>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>RLX</th>
      <th>6.0</th>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>RSX</th>
      <th>4.0</th>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th>SLX</th>
      <th>6.0</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>TL</th>
      <th>6.0</th>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">TLX</th>
      <th>4.0</th>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">TSX</th>
      <th>4.0</th>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>TSX Sport Wagon</th>
      <th>4.0</th>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Vigor</th>
      <th>5.0</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>ZDX</th>
      <th>6.0</th>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Alfa Romeo</th>
      <th>4C</th>
      <th>4.0</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">Aston Martin</th>
      <th>DB7</th>
      <th>12.0</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>DB9</th>
      <th>12.0</th>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>DB9 GT</th>
      <th>12.0</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>DBS</th>
      <th>12.0</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>Rapide</th>
      <th>12.0</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Rapide S</th>
      <th>12.0</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>V12 Vanquish</th>
      <th>12.0</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>V12 Vantage</th>
      <th>12.0</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>V12 Vantage S</th>
      <th>12.0</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
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
    </tr>
    <tr>
      <th rowspan="30" valign="top">Volvo</th>
      <th>C70</th>
      <th>5.0</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Coupe</th>
      <th>4.0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>S40</th>
      <th>5.0</th>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">S60</th>
      <th>4.0</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>S60 Cross Country</th>
      <th>4.0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>S70</th>
      <th>5.0</th>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">S80</th>
      <th>4.0</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">S90</th>
      <th>4.0</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>V40</th>
      <th>4.0</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>V50</th>
      <th>5.0</th>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">V60</th>
      <th>4.0</th>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">V60 Cross Country</th>
      <th>4.0</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>V70</th>
      <th>6.0</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>V90</th>
      <th>6.0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>XC</th>
      <th>5.0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">XC60</th>
      <th>4.0</th>
      <td>26</td>
      <td>26</td>
      <td>26</td>
      <td>26</td>
      <td>26</td>
      <td>26</td>
      <td>26</td>
      <td>26</td>
      <td>26</td>
      <td>26</td>
      <td>26</td>
      <td>26</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">XC70</th>
      <th>4.0</th>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">XC90</th>
      <th>4.0</th>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>1160 rows × 12 columns</p>
</div>




```python
grouped = df.groupby('Year')
print(grouped.get_group(2014))
```

                  Make                  Model  Year             Engine Fuel Type  \
    120          Mazda                      2  2014             regular unleaded   
    121          Mazda                      2  2014             regular unleaded   
    122          Mazda                      2  2014             regular unleaded   
    123          Mazda                      2  2014             regular unleaded   
    468        Ferrari             458 Italia  2014  premium unleaded (required)   
    469        Ferrari             458 Italia  2014  premium unleaded (required)   
    470        Ferrari             458 Italia  2014  premium unleaded (required)   
    479         Toyota                4Runner  2014             regular unleaded   
    480         Toyota                4Runner  2014             regular unleaded   
    481         Toyota                4Runner  2014             regular unleaded   
    482         Toyota                4Runner  2014             regular unleaded   
    483         Toyota                4Runner  2014             regular unleaded   
    484         Toyota                4Runner  2014             regular unleaded   
    485         Toyota                4Runner  2014             regular unleaded   
    486         Toyota                4Runner  2014             regular unleaded   
    627          Mazda                      5  2014             regular unleaded   
    628          Mazda                      5  2014             regular unleaded   
    629          Mazda                      5  2014             regular unleaded   
    630          Mazda                      5  2014             regular unleaded   
    1171       Hyundai                 Accent  2014             regular unleaded   
    1172       Hyundai                 Accent  2014             regular unleaded   
    1173       Hyundai                 Accent  2014             regular unleaded   
    1174       Hyundai                 Accent  2014             regular unleaded   
    1175       Hyundai                 Accent  2014             regular unleaded   
    1176       Hyundai                 Accent  2014             regular unleaded   
    1202         Honda          Accord Hybrid  2014             regular unleaded   
    1203         Honda          Accord Hybrid  2014             regular unleaded   
    1204         Honda          Accord Hybrid  2014             regular unleaded   
    1211         Honda  Accord Plug-In Hybrid  2014             regular unleaded   
    1292           BMW         ActiveHybrid 5  2014  premium unleaded (required)   
    ...            ...                    ...   ...                          ...   
    11283       Toyota                  Venza  2014             regular unleaded   
    11284       Toyota                  Venza  2014             regular unleaded   
    11285       Toyota                  Venza  2014             regular unleaded   
    11286       Toyota                  Venza  2014             regular unleaded   
    11287       Toyota                  Venza  2014             regular unleaded   
    11448  Rolls-Royce                 Wraith  2014  premium unleaded (required)   
    11545        Scion                     xB  2014             regular unleaded   
    11546        Scion                     xB  2014             regular unleaded   
    11547        Scion                     xB  2014             regular unleaded   
    11605        Volvo                   XC70  2014             regular unleaded   
    11606        Volvo                   XC70  2014             regular unleaded   
    11623        Volvo                   XC90  2014             regular unleaded   
    11624        Volvo                   XC90  2014             regular unleaded   
    11649        Scion                     xD  2014             regular unleaded   
    11650        Scion                     xD  2014             regular unleaded   
    11751       Nissan                 Xterra  2014             regular unleaded   
    11752       Nissan                 Xterra  2014             regular unleaded   
    11753       Nissan                 Xterra  2014             regular unleaded   
    11754       Nissan                 Xterra  2014             regular unleaded   
    11755       Nissan                 Xterra  2014             regular unleaded   
    11756       Nissan                 Xterra  2014             regular unleaded   
    11757       Nissan                 Xterra  2014             regular unleaded   
    11798       Subaru           XV Crosstrek  2014             regular unleaded   
    11799       Subaru           XV Crosstrek  2014             regular unleaded   
    11800       Subaru           XV Crosstrek  2014             regular unleaded   
    11801       Subaru           XV Crosstrek  2014             regular unleaded   
    11802       Subaru           XV Crosstrek  2014             regular unleaded   
    11894          BMW                     Z4  2014  premium unleaded (required)   
    11895          BMW                     Z4  2014  premium unleaded (required)   
    11896          BMW                     Z4  2014  premium unleaded (required)   
    
           Engine HP  Engine Cylinders Transmission Type      Driven_Wheels  \
    120        100.0               4.0         AUTOMATIC  front wheel drive   
    121        100.0               4.0         AUTOMATIC  front wheel drive   
    122        100.0               4.0            MANUAL  front wheel drive   
    123        100.0               4.0            MANUAL  front wheel drive   
    468        562.0               8.0  AUTOMATED_MANUAL   rear wheel drive   
    469        597.0               8.0  AUTOMATED_MANUAL   rear wheel drive   
    470        562.0               8.0  AUTOMATED_MANUAL   rear wheel drive   
    479        270.0               6.0         AUTOMATIC   rear wheel drive   
    480        270.0               6.0         AUTOMATIC   rear wheel drive   
    481        270.0               6.0         AUTOMATIC   four wheel drive   
    482        270.0               6.0         AUTOMATIC   four wheel drive   
    483        270.0               6.0         AUTOMATIC   four wheel drive   
    484        270.0               6.0         AUTOMATIC   four wheel drive   
    485        270.0               6.0         AUTOMATIC   rear wheel drive   
    486        270.0               6.0         AUTOMATIC   four wheel drive   
    627        157.0               4.0            MANUAL  front wheel drive   
    628        157.0               4.0         AUTOMATIC  front wheel drive   
    629        157.0               4.0         AUTOMATIC  front wheel drive   
    630        157.0               4.0         AUTOMATIC  front wheel drive   
    1171       138.0               4.0         AUTOMATIC  front wheel drive   
    1172       138.0               4.0            MANUAL  front wheel drive   
    1173       138.0               4.0         AUTOMATIC  front wheel drive   
    1174       138.0               4.0         AUTOMATIC  front wheel drive   
    1175       138.0               4.0            MANUAL  front wheel drive   
    1176       138.0               4.0            MANUAL  front wheel drive   
    1202       195.0               4.0         AUTOMATIC  front wheel drive   
    1203       195.0               4.0         AUTOMATIC  front wheel drive   
    1204       195.0               4.0         AUTOMATIC  front wheel drive   
    1211       196.0               4.0         AUTOMATIC  front wheel drive   
    1292       335.0               6.0         AUTOMATIC   rear wheel drive   
    ...          ...               ...               ...                ...   
    11283      268.0               6.0         AUTOMATIC    all wheel drive   
    11284      181.0               4.0         AUTOMATIC  front wheel drive   
    11285      268.0               6.0         AUTOMATIC  front wheel drive   
    11286      181.0               4.0         AUTOMATIC  front wheel drive   
    11287      181.0               4.0         AUTOMATIC    all wheel drive   
    11448      624.0              12.0         AUTOMATIC   rear wheel drive   
    11545      158.0               4.0            MANUAL  front wheel drive   
    11546      158.0               4.0         AUTOMATIC  front wheel drive   
    11547      158.0               4.0         AUTOMATIC  front wheel drive   
    11605      240.0               6.0         AUTOMATIC  front wheel drive   
    11606      300.0               6.0         AUTOMATIC    all wheel drive   
    11623      240.0               6.0         AUTOMATIC  front wheel drive   
    11624      240.0               6.0         AUTOMATIC  front wheel drive   
    11649      128.0               4.0            MANUAL  front wheel drive   
    11650      128.0               4.0         AUTOMATIC  front wheel drive   
    11751      261.0               6.0         AUTOMATIC   four wheel drive   
    11752      261.0               6.0         AUTOMATIC   rear wheel drive   
    11753      261.0               6.0         AUTOMATIC   four wheel drive   
    11754      261.0               6.0         AUTOMATIC   four wheel drive   
    11755      261.0               6.0            MANUAL   four wheel drive   
    11756      261.0               6.0         AUTOMATIC   rear wheel drive   
    11757      261.0               6.0            MANUAL   four wheel drive   
    11798      160.0               4.0         AUTOMATIC    all wheel drive   
    11799      160.0               4.0         AUTOMATIC    all wheel drive   
    11800      148.0               4.0         AUTOMATIC    all wheel drive   
    11801      148.0               4.0         AUTOMATIC    all wheel drive   
    11802      148.0               4.0            MANUAL    all wheel drive   
    11894      240.0               4.0            MANUAL   rear wheel drive   
    11895      300.0               6.0            MANUAL   rear wheel drive   
    11896      335.0               6.0  AUTOMATED_MANUAL   rear wheel drive   
    
           Number of Doors Vehicle Size      Vehicle Style  highway MPG  city mpg  \
    120                4.0      Compact      4dr Hatchback           34        28   
    121                4.0      Compact      4dr Hatchback           34        28   
    122                4.0      Compact      4dr Hatchback           35        29   
    123                4.0      Compact      4dr Hatchback           35        29   
    468                2.0      Compact              Coupe           17        13   
    469                2.0      Compact              Coupe           17        13   
    470                2.0      Compact        Convertible           17        13   
    479                4.0      Midsize            4dr SUV           23        17   
    480                4.0      Midsize            4dr SUV           23        17   
    481                4.0      Midsize            4dr SUV           22        17   
    482                4.0      Midsize            4dr SUV           22        17   
    483                4.0      Midsize            4dr SUV           22        17   
    484                4.0      Midsize            4dr SUV           22        17   
    485                4.0      Midsize            4dr SUV           23        17   
    486                4.0      Midsize            4dr SUV           22        17   
    627                4.0      Compact  Passenger Minivan           28        21   
    628                4.0      Compact  Passenger Minivan           28        22   
    629                4.0      Compact  Passenger Minivan           28        22   
    630                4.0      Compact  Passenger Minivan           28        22   
    1171               4.0      Compact      4dr Hatchback           37        27   
    1172               4.0      Compact      4dr Hatchback           38        27   
    1173               4.0      Compact              Sedan           37        27   
    1174               4.0      Compact      4dr Hatchback           37        27   
    1175               4.0      Compact              Sedan           38        27   
    1176               4.0      Compact      4dr Hatchback           38        27   
    1202               4.0      Midsize              Sedan           45        50   
    1203               4.0      Midsize              Sedan           45        50   
    1204               4.0      Midsize              Sedan           45        50   
    1211               4.0      Midsize              Sedan           46        47   
    1292               4.0        Large              Sedan           30        23   
    ...                ...          ...                ...          ...       ...   
    11283              4.0      Midsize              Wagon           25        18   
    11284              4.0      Midsize              Wagon           26        20   
    11285              4.0      Midsize              Wagon           26        19   
    11286              4.0      Midsize              Wagon           26        20   
    11287              4.0      Midsize              Wagon           26        20   
    11448              2.0        Large              Coupe           21        13   
    11545              4.0      Compact              Wagon           28        22   
    11546              4.0      Compact              Wagon           28        22   
    11547              4.0      Compact              Wagon           28        22   
    11605              4.0      Midsize              Wagon           26        18   
    11606              4.0      Midsize              Wagon           24        17   
    11623              4.0      Midsize            4dr SUV           25        16   
    11624              4.0      Midsize            4dr SUV           25        16   
    11649              4.0      Compact      4dr Hatchback           33        27   
    11650              4.0      Compact      4dr Hatchback           33        27   
    11751              4.0      Midsize            4dr SUV           20        15   
    11752              4.0      Midsize            4dr SUV           22        16   
    11753              4.0      Midsize            4dr SUV           20        15   
    11754              4.0      Midsize            4dr SUV           20        15   
    11755              4.0      Midsize            4dr SUV           20        16   
    11756              4.0      Midsize            4dr SUV           22        16   
    11757              4.0      Midsize            4dr SUV           20        16   
    11798              4.0      Compact            4dr SUV           33        29   
    11799              4.0      Compact            4dr SUV           33        29   
    11800              4.0      Compact            4dr SUV           33        25   
    11801              4.0      Compact            4dr SUV           33        25   
    11802              4.0      Compact            4dr SUV           30        23   
    11894              2.0      Compact        Convertible           34        22   
    11895              2.0      Compact        Convertible           26        19   
    11896              2.0      Compact        Convertible           24        17   
    
           Popularity    MSRP  
    120           586   17050  
    121           586   15560  
    122           586   16210  
    123           586   14720  
    468          2774  233509  
    469          2774  288000  
    470          2774  257412  
    479          2031   41365  
    480          2031   35740  
    481          2031   37615  
    482          2031   34695  
    483          2031   35725  
    484          2031   43400  
    485          2031   32820  
    486          2031   38645  
    627           586   20140  
    628           586   24670  
    629           586   21140  
    630           586   22270  
    1171         1439   17395  
    1172         1439   14895  
    1173         1439   15645  
    1174         1439   16095  
    1175         1439   14645  
    1176         1439   16395  
    1202         2202   31905  
    1203         2202   29155  
    1204         2202   34905  
    1211         2202   39780  
    1292         3916   61400  
    ...           ...     ...  
    11283        2031   31220  
    11284        2031   27950  
    11285        2031   38120  
    11286        2031   31810  
    11287        2031   33260  
    11448          86  284900  
    11545         105   16970  
    11546         105   20420  
    11547         105   17920  
    11605         870   34500  
    11606         870   40950  
    11623         870   39700  
    11624         870   42700  
    11649         105   15920  
    11650         105   16720  
    11751        2009   31370  
    11752        2009   25300  
    11753        2009   27350  
    11754        2009   25440  
    11755        2009   26300  
    11756        2009   23390  
    11757        2009   30320  
    11798         640   25995  
    11799         640   29295  
    11800         640   22995  
    11801         640   24495  
    11802         640   21995  
    11894        3916   48950  
    11895        3916   56950  
    11896        3916   65800  
    
    [554 rows x 15 columns]


<a id='ab-engineft'></a>

### Null values in Engine Fuel Type 


```python
# Examine the null values of Engine Fuel Type
df[df['Engine Fuel Type'].isnull()]
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11321</th>
      <td>Suzuki</td>
      <td>Verona</td>
      <td>2004</td>
      <td>NaN</td>
      <td>155.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>481</td>
      <td>17199</td>
    </tr>
    <tr>
      <th>11322</th>
      <td>Suzuki</td>
      <td>Verona</td>
      <td>2004</td>
      <td>NaN</td>
      <td>155.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>481</td>
      <td>20199</td>
    </tr>
    <tr>
      <th>11323</th>
      <td>Suzuki</td>
      <td>Verona</td>
      <td>2004</td>
      <td>NaN</td>
      <td>155.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>481</td>
      <td>18499</td>
    </tr>
  </tbody>
</table>
</div>




```python
# look at other models with the same model
# Ended up searching the web to get answer. Since the years didn't match
df[df['Model'] == 'Verona']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11321</th>
      <td>Suzuki</td>
      <td>Verona</td>
      <td>2004</td>
      <td>NaN</td>
      <td>155.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>481</td>
      <td>17199</td>
    </tr>
    <tr>
      <th>11322</th>
      <td>Suzuki</td>
      <td>Verona</td>
      <td>2004</td>
      <td>NaN</td>
      <td>155.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>481</td>
      <td>20199</td>
    </tr>
    <tr>
      <th>11323</th>
      <td>Suzuki</td>
      <td>Verona</td>
      <td>2004</td>
      <td>NaN</td>
      <td>155.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>481</td>
      <td>18499</td>
    </tr>
    <tr>
      <th>11324</th>
      <td>Suzuki</td>
      <td>Verona</td>
      <td>2005</td>
      <td>regular unleaded</td>
      <td>155.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>25</td>
      <td>18</td>
      <td>481</td>
      <td>19349</td>
    </tr>
    <tr>
      <th>11325</th>
      <td>Suzuki</td>
      <td>Verona</td>
      <td>2005</td>
      <td>regular unleaded</td>
      <td>155.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>25</td>
      <td>18</td>
      <td>481</td>
      <td>21049</td>
    </tr>
    <tr>
      <th>11326</th>
      <td>Suzuki</td>
      <td>Verona</td>
      <td>2005</td>
      <td>regular unleaded</td>
      <td>155.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>25</td>
      <td>18</td>
      <td>481</td>
      <td>17549</td>
    </tr>
    <tr>
      <th>11327</th>
      <td>Suzuki</td>
      <td>Verona</td>
      <td>2005</td>
      <td>regular unleaded</td>
      <td>155.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>25</td>
      <td>18</td>
      <td>481</td>
      <td>20549</td>
    </tr>
    <tr>
      <th>11328</th>
      <td>Suzuki</td>
      <td>Verona</td>
      <td>2006</td>
      <td>regular unleaded</td>
      <td>155.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>481</td>
      <td>20299</td>
    </tr>
    <tr>
      <th>11329</th>
      <td>Suzuki</td>
      <td>Verona</td>
      <td>2006</td>
      <td>regular unleaded</td>
      <td>155.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>25</td>
      <td>17</td>
      <td>481</td>
      <td>18299</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Couldn't get the for loop too work. 
# Took the long way to solve instead of using so much time trying to figure it out
# 
df.loc[11321,'Engine Fuel Type'] = 'regular unleaded'
df.loc[11322,'Engine Fuel Type'] = 'regular unleaded'
df.loc[11323,'Engine Fuel Type'] = 'regular unleaded'
```


```python
df['Engine Fuel Type'].isnull().sum()
```




    0



<a id='ab-numdoors'></a>

## Null values in Number of Doors


```python
# Examine the null values of number of Doors
df[df['Number of Doors'].isnull()]
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4666</th>
      <td>Ferrari</td>
      <td>FF</td>
      <td>2013</td>
      <td>premium unleaded (required)</td>
      <td>651.0</td>
      <td>12.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>all wheel drive</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Coupe</td>
      <td>16</td>
      <td>11</td>
      <td>2774</td>
      <td>295000</td>
    </tr>
    <tr>
      <th>6930</th>
      <td>Tesla</td>
      <td>Model D</td>
      <td>2016</td>
      <td>electric</td>
      <td>259.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>105</td>
      <td>102</td>
      <td>1391</td>
      <td>79500</td>
    </tr>
    <tr>
      <th>6931</th>
      <td>Tesla</td>
      <td>Model D</td>
      <td>2016</td>
      <td>electric</td>
      <td>259.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>101</td>
      <td>98</td>
      <td>1391</td>
      <td>66000</td>
    </tr>
    <tr>
      <th>6932</th>
      <td>Tesla</td>
      <td>Model D</td>
      <td>2016</td>
      <td>electric</td>
      <td>259.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>105</td>
      <td>92</td>
      <td>1391</td>
      <td>134500</td>
    </tr>
    <tr>
      <th>6933</th>
      <td>Tesla</td>
      <td>Model S</td>
      <td>2016</td>
      <td>electric</td>
      <td>362.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>rear wheel drive</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>100</td>
      <td>97</td>
      <td>1391</td>
      <td>74500</td>
    </tr>
    <tr>
      <th>6934</th>
      <td>Tesla</td>
      <td>Model D</td>
      <td>2016</td>
      <td>electric</td>
      <td>259.0</td>
      <td>0.0</td>
      <td>DIRECT_DRIVE</td>
      <td>all wheel drive</td>
      <td>NaN</td>
      <td>Large</td>
      <td>Sedan</td>
      <td>107</td>
      <td>101</td>
      <td>1391</td>
      <td>71000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# np.isnan(df['Number of Doors'][6930])
```


```python
# i will iterate through index 
# j will show value of column 'Number of Doors'
# if statement will replace number of doors 

for i, j in df['Number of Doors'].iteritems():
    if np.isnan(df['Number of Doors'][i]):
#         print (i,j)
        if df['Vehicle Style'][i] == 'Coupe':
            df['Number of Doors'][i] = 2
        elif df['Vehicle Style'][i] == 'Sedan':
            df['Number of Doors'][i] = 4
            
            

        
#     print (i, j)
# if df['Number of Doors'].isnull():
#     df['Vehicle Style'] == 'Coupe'
    
#     else:
#     df['Vehicle Style'] == 'Sedan'
    
```

    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
df['Number of Doors'].isnull().sum()
```




    0




```python
df.isnull().sum()
```




    Make                 0
    Model                0
    Year                 0
    Engine Fuel Type     0
    Engine HP            0
    Engine Cylinders     0
    Transmission Type    0
    Driven_Wheels        0
    Number of Doors      0
    Vehicle Size         0
    Vehicle Style        0
    highway MPG          0
    city mpg             0
    Popularity           0
    MSRP                 0
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BMW</td>
      <td>1 Series M</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>335.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>26</td>
      <td>19</td>
      <td>3916</td>
      <td>46135</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>28</td>
      <td>19</td>
      <td>3916</td>
      <td>40650</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>20</td>
      <td>3916</td>
      <td>36350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>29450</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>34500</td>
    </tr>
  </tbody>
</table>
</div>



<a id='ab-unknown'></a>

# Examine transmission type Unknown


```python
df[df['Transmission Type'] == 'UNKNOWN']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1289</th>
      <td>Oldsmobile</td>
      <td>Achieva</td>
      <td>1997</td>
      <td>regular unleaded</td>
      <td>150.0</td>
      <td>4.0</td>
      <td>UNKNOWN</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>Coupe</td>
      <td>29</td>
      <td>19</td>
      <td>26</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>1290</th>
      <td>Oldsmobile</td>
      <td>Achieva</td>
      <td>1997</td>
      <td>regular unleaded</td>
      <td>150.0</td>
      <td>4.0</td>
      <td>UNKNOWN</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>29</td>
      <td>19</td>
      <td>26</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>4691</th>
      <td>Pontiac</td>
      <td>Firebird</td>
      <td>2000</td>
      <td>regular unleaded</td>
      <td>305.0</td>
      <td>8.0</td>
      <td>UNKNOWN</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>2dr Hatchback</td>
      <td>23</td>
      <td>15</td>
      <td>210</td>
      <td>6175</td>
    </tr>
    <tr>
      <th>4692</th>
      <td>Pontiac</td>
      <td>Firebird</td>
      <td>2000</td>
      <td>regular unleaded</td>
      <td>305.0</td>
      <td>8.0</td>
      <td>UNKNOWN</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>2dr Hatchback</td>
      <td>23</td>
      <td>15</td>
      <td>210</td>
      <td>8548</td>
    </tr>
    <tr>
      <th>4693</th>
      <td>Pontiac</td>
      <td>Firebird</td>
      <td>2000</td>
      <td>regular unleaded</td>
      <td>305.0</td>
      <td>8.0</td>
      <td>UNKNOWN</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>Convertible</td>
      <td>23</td>
      <td>15</td>
      <td>210</td>
      <td>9567</td>
    </tr>
    <tr>
      <th>6158</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>1999</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>UNKNOWN</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr SUV</td>
      <td>19</td>
      <td>14</td>
      <td>549</td>
      <td>2182</td>
    </tr>
    <tr>
      <th>6160</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>1999</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>UNKNOWN</td>
      <td>four wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr SUV</td>
      <td>19</td>
      <td>14</td>
      <td>549</td>
      <td>2317</td>
    </tr>
    <tr>
      <th>6165</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2000</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>UNKNOWN</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>549</td>
      <td>2407</td>
    </tr>
    <tr>
      <th>6174</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2000</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>UNKNOWN</td>
      <td>four wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr SUV</td>
      <td>18</td>
      <td>14</td>
      <td>549</td>
      <td>2578</td>
    </tr>
    <tr>
      <th>6366</th>
      <td>Chrysler</td>
      <td>Le Baron</td>
      <td>1993</td>
      <td>regular unleaded</td>
      <td>100.0</td>
      <td>4.0</td>
      <td>UNKNOWN</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>26</td>
      <td>21</td>
      <td>1013</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>6368</th>
      <td>Chrysler</td>
      <td>Le Baron</td>
      <td>1993</td>
      <td>regular unleaded</td>
      <td>100.0</td>
      <td>4.0</td>
      <td>UNKNOWN</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>24</td>
      <td>18</td>
      <td>1013</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>8042</th>
      <td>Dodge</td>
      <td>RAM 150</td>
      <td>1991</td>
      <td>regular unleaded</td>
      <td>125.0</td>
      <td>6.0</td>
      <td>UNKNOWN</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Regular Cab Pickup</td>
      <td>17</td>
      <td>12</td>
      <td>1851</td>
      <td>2000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Look through model firebird to see if I can use data to fill transmission type
df[df['Model'] == 'Firebird']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4690</th>
      <td>Pontiac</td>
      <td>Firebird</td>
      <td>2000</td>
      <td>regular unleaded</td>
      <td>200.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>2dr Hatchback</td>
      <td>28</td>
      <td>17</td>
      <td>210</td>
      <td>4677</td>
    </tr>
    <tr>
      <th>4691</th>
      <td>Pontiac</td>
      <td>Firebird</td>
      <td>2000</td>
      <td>regular unleaded</td>
      <td>305.0</td>
      <td>8.0</td>
      <td>UNKNOWN</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>2dr Hatchback</td>
      <td>23</td>
      <td>15</td>
      <td>210</td>
      <td>6175</td>
    </tr>
    <tr>
      <th>4692</th>
      <td>Pontiac</td>
      <td>Firebird</td>
      <td>2000</td>
      <td>regular unleaded</td>
      <td>305.0</td>
      <td>8.0</td>
      <td>UNKNOWN</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>2dr Hatchback</td>
      <td>23</td>
      <td>15</td>
      <td>210</td>
      <td>8548</td>
    </tr>
    <tr>
      <th>4693</th>
      <td>Pontiac</td>
      <td>Firebird</td>
      <td>2000</td>
      <td>regular unleaded</td>
      <td>305.0</td>
      <td>8.0</td>
      <td>UNKNOWN</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>Convertible</td>
      <td>23</td>
      <td>15</td>
      <td>210</td>
      <td>9567</td>
    </tr>
    <tr>
      <th>4694</th>
      <td>Pontiac</td>
      <td>Firebird</td>
      <td>2000</td>
      <td>regular unleaded</td>
      <td>200.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>Convertible</td>
      <td>28</td>
      <td>17</td>
      <td>210</td>
      <td>5844</td>
    </tr>
    <tr>
      <th>4695</th>
      <td>Pontiac</td>
      <td>Firebird</td>
      <td>2001</td>
      <td>premium unleaded (required)</td>
      <td>310.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>2dr Hatchback</td>
      <td>23</td>
      <td>16</td>
      <td>210</td>
      <td>24035</td>
    </tr>
    <tr>
      <th>4696</th>
      <td>Pontiac</td>
      <td>Firebird</td>
      <td>2001</td>
      <td>regular unleaded</td>
      <td>200.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>Convertible</td>
      <td>28</td>
      <td>17</td>
      <td>210</td>
      <td>25475</td>
    </tr>
    <tr>
      <th>4697</th>
      <td>Pontiac</td>
      <td>Firebird</td>
      <td>2001</td>
      <td>premium unleaded (required)</td>
      <td>310.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>Convertible</td>
      <td>23</td>
      <td>16</td>
      <td>210</td>
      <td>31215</td>
    </tr>
    <tr>
      <th>4698</th>
      <td>Pontiac</td>
      <td>Firebird</td>
      <td>2001</td>
      <td>regular unleaded</td>
      <td>200.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>2dr Hatchback</td>
      <td>28</td>
      <td>17</td>
      <td>210</td>
      <td>18855</td>
    </tr>
    <tr>
      <th>4699</th>
      <td>Pontiac</td>
      <td>Firebird</td>
      <td>2001</td>
      <td>premium unleaded (required)</td>
      <td>310.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>2dr Hatchback</td>
      <td>23</td>
      <td>16</td>
      <td>210</td>
      <td>27145</td>
    </tr>
    <tr>
      <th>4700</th>
      <td>Pontiac</td>
      <td>Firebird</td>
      <td>2002</td>
      <td>premium unleaded (required)</td>
      <td>310.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>Convertible</td>
      <td>23</td>
      <td>16</td>
      <td>210</td>
      <td>32095</td>
    </tr>
    <tr>
      <th>4701</th>
      <td>Pontiac</td>
      <td>Firebird</td>
      <td>2002</td>
      <td>premium unleaded (required)</td>
      <td>310.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>2dr Hatchback</td>
      <td>23</td>
      <td>16</td>
      <td>210</td>
      <td>25995</td>
    </tr>
    <tr>
      <th>4702</th>
      <td>Pontiac</td>
      <td>Firebird</td>
      <td>2002</td>
      <td>premium unleaded (required)</td>
      <td>310.0</td>
      <td>8.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>2dr Hatchback</td>
      <td>23</td>
      <td>16</td>
      <td>210</td>
      <td>28025</td>
    </tr>
    <tr>
      <th>4703</th>
      <td>Pontiac</td>
      <td>Firebird</td>
      <td>2002</td>
      <td>regular unleaded</td>
      <td>200.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>2dr Hatchback</td>
      <td>29</td>
      <td>17</td>
      <td>210</td>
      <td>20050</td>
    </tr>
    <tr>
      <th>4704</th>
      <td>Pontiac</td>
      <td>Firebird</td>
      <td>2002</td>
      <td>regular unleaded</td>
      <td>200.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>Convertible</td>
      <td>28</td>
      <td>17</td>
      <td>210</td>
      <td>26965</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['Model'] == 'Achieva']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1287</th>
      <td>Oldsmobile</td>
      <td>Achieva</td>
      <td>1996</td>
      <td>regular unleaded</td>
      <td>150.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>30</td>
      <td>20</td>
      <td>26</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>1288</th>
      <td>Oldsmobile</td>
      <td>Achieva</td>
      <td>1996</td>
      <td>regular unleaded</td>
      <td>150.0</td>
      <td>4.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>Coupe</td>
      <td>30</td>
      <td>20</td>
      <td>26</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>1289</th>
      <td>Oldsmobile</td>
      <td>Achieva</td>
      <td>1997</td>
      <td>regular unleaded</td>
      <td>150.0</td>
      <td>4.0</td>
      <td>UNKNOWN</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>Coupe</td>
      <td>29</td>
      <td>19</td>
      <td>26</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>1290</th>
      <td>Oldsmobile</td>
      <td>Achieva</td>
      <td>1997</td>
      <td>regular unleaded</td>
      <td>150.0</td>
      <td>4.0</td>
      <td>UNKNOWN</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>29</td>
      <td>19</td>
      <td>26</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>1291</th>
      <td>Oldsmobile</td>
      <td>Achieva</td>
      <td>1998</td>
      <td>regular unleaded</td>
      <td>150.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>Sedan</td>
      <td>27</td>
      <td>18</td>
      <td>26</td>
      <td>2000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['Model'] == 'Jimmy']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6155</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>1999</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>four wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr SUV</td>
      <td>16</td>
      <td>13</td>
      <td>549</td>
      <td>2347</td>
    </tr>
    <tr>
      <th>6156</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>1999</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>19</td>
      <td>14</td>
      <td>549</td>
      <td>2554</td>
    </tr>
    <tr>
      <th>6157</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>1999</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>19</td>
      <td>14</td>
      <td>549</td>
      <td>2590</td>
    </tr>
    <tr>
      <th>6158</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>1999</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>UNKNOWN</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr SUV</td>
      <td>19</td>
      <td>14</td>
      <td>549</td>
      <td>2182</td>
    </tr>
    <tr>
      <th>6159</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>1999</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>19</td>
      <td>14</td>
      <td>549</td>
      <td>2691</td>
    </tr>
    <tr>
      <th>6160</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>1999</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>UNKNOWN</td>
      <td>four wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr SUV</td>
      <td>19</td>
      <td>14</td>
      <td>549</td>
      <td>2317</td>
    </tr>
    <tr>
      <th>6161</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>1999</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>19</td>
      <td>14</td>
      <td>549</td>
      <td>2368</td>
    </tr>
    <tr>
      <th>6162</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>1999</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>19</td>
      <td>14</td>
      <td>549</td>
      <td>2377</td>
    </tr>
    <tr>
      <th>6163</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>1999</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>19</td>
      <td>14</td>
      <td>549</td>
      <td>2251</td>
    </tr>
    <tr>
      <th>6164</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>1999</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr SUV</td>
      <td>21</td>
      <td>15</td>
      <td>549</td>
      <td>2038</td>
    </tr>
    <tr>
      <th>6165</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2000</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>UNKNOWN</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>549</td>
      <td>2407</td>
    </tr>
    <tr>
      <th>6166</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2000</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>four wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr SUV</td>
      <td>16</td>
      <td>13</td>
      <td>549</td>
      <td>2463</td>
    </tr>
    <tr>
      <th>6167</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2000</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>18</td>
      <td>14</td>
      <td>549</td>
      <td>2773</td>
    </tr>
    <tr>
      <th>6168</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2000</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>549</td>
      <td>2756</td>
    </tr>
    <tr>
      <th>6169</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2000</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>549</td>
      <td>2590</td>
    </tr>
    <tr>
      <th>6170</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2000</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>18</td>
      <td>14</td>
      <td>549</td>
      <td>2916</td>
    </tr>
    <tr>
      <th>6171</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2000</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr SUV</td>
      <td>21</td>
      <td>15</td>
      <td>549</td>
      <td>2322</td>
    </tr>
    <tr>
      <th>6172</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2000</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>549</td>
      <td>2623</td>
    </tr>
    <tr>
      <th>6173</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2000</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>18</td>
      <td>14</td>
      <td>549</td>
      <td>2655</td>
    </tr>
    <tr>
      <th>6174</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2000</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>UNKNOWN</td>
      <td>four wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr SUV</td>
      <td>18</td>
      <td>14</td>
      <td>549</td>
      <td>2578</td>
    </tr>
    <tr>
      <th>6175</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2001</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>549</td>
      <td>26770</td>
    </tr>
    <tr>
      <th>6176</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2001</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>four wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr SUV</td>
      <td>16</td>
      <td>13</td>
      <td>549</td>
      <td>22270</td>
    </tr>
    <tr>
      <th>6177</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2001</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr SUV</td>
      <td>18</td>
      <td>14</td>
      <td>549</td>
      <td>25170</td>
    </tr>
    <tr>
      <th>6178</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2001</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>18</td>
      <td>14</td>
      <td>549</td>
      <td>31925</td>
    </tr>
    <tr>
      <th>6179</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2001</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr SUV</td>
      <td>20</td>
      <td>14</td>
      <td>549</td>
      <td>19270</td>
    </tr>
    <tr>
      <th>6180</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2001</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>2dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>549</td>
      <td>22170</td>
    </tr>
    <tr>
      <th>6181</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2001</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>18</td>
      <td>14</td>
      <td>549</td>
      <td>30225</td>
    </tr>
    <tr>
      <th>6182</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2001</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>549</td>
      <td>28225</td>
    </tr>
    <tr>
      <th>6183</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2001</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>18</td>
      <td>14</td>
      <td>549</td>
      <td>33920</td>
    </tr>
    <tr>
      <th>6184</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2001</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>four wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>18</td>
      <td>14</td>
      <td>549</td>
      <td>28770</td>
    </tr>
    <tr>
      <th>6185</th>
      <td>GMC</td>
      <td>Jimmy</td>
      <td>2001</td>
      <td>regular unleaded</td>
      <td>190.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>4.0</td>
      <td>Midsize</td>
      <td>4dr SUV</td>
      <td>20</td>
      <td>15</td>
      <td>549</td>
      <td>29925</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['Model'] == 'Le Baron']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6364</th>
      <td>Chrysler</td>
      <td>Le Baron</td>
      <td>1993</td>
      <td>regular unleaded</td>
      <td>141.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>26</td>
      <td>17</td>
      <td>1013</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>6365</th>
      <td>Chrysler</td>
      <td>Le Baron</td>
      <td>1993</td>
      <td>regular unleaded</td>
      <td>141.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>26</td>
      <td>18</td>
      <td>1013</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>6366</th>
      <td>Chrysler</td>
      <td>Le Baron</td>
      <td>1993</td>
      <td>regular unleaded</td>
      <td>100.0</td>
      <td>4.0</td>
      <td>UNKNOWN</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>26</td>
      <td>21</td>
      <td>1013</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>6367</th>
      <td>Chrysler</td>
      <td>Le Baron</td>
      <td>1993</td>
      <td>regular unleaded</td>
      <td>141.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>24</td>
      <td>17</td>
      <td>1013</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>6368</th>
      <td>Chrysler</td>
      <td>Le Baron</td>
      <td>1993</td>
      <td>regular unleaded</td>
      <td>100.0</td>
      <td>4.0</td>
      <td>UNKNOWN</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>24</td>
      <td>18</td>
      <td>1013</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>6369</th>
      <td>Chrysler</td>
      <td>Le Baron</td>
      <td>1993</td>
      <td>regular unleaded</td>
      <td>141.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>26</td>
      <td>18</td>
      <td>1013</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>6370</th>
      <td>Chrysler</td>
      <td>Le Baron</td>
      <td>1993</td>
      <td>regular unleaded</td>
      <td>100.0</td>
      <td>4.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>26</td>
      <td>21</td>
      <td>1013</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>6371</th>
      <td>Chrysler</td>
      <td>Le Baron</td>
      <td>1993</td>
      <td>regular unleaded</td>
      <td>141.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>26</td>
      <td>18</td>
      <td>1013</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>6372</th>
      <td>Chrysler</td>
      <td>Le Baron</td>
      <td>1994</td>
      <td>regular unleaded</td>
      <td>141.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>26</td>
      <td>18</td>
      <td>1013</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>6373</th>
      <td>Chrysler</td>
      <td>Le Baron</td>
      <td>1994</td>
      <td>regular unleaded</td>
      <td>142.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>24</td>
      <td>18</td>
      <td>1013</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>6374</th>
      <td>Chrysler</td>
      <td>Le Baron</td>
      <td>1994</td>
      <td>regular unleaded</td>
      <td>142.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>4.0</td>
      <td>Compact</td>
      <td>Sedan</td>
      <td>26</td>
      <td>18</td>
      <td>1013</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>6375</th>
      <td>Chrysler</td>
      <td>Le Baron</td>
      <td>1995</td>
      <td>regular unleaded</td>
      <td>141.0</td>
      <td>6.0</td>
      <td>AUTOMATIC</td>
      <td>front wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>26</td>
      <td>18</td>
      <td>1013</td>
      <td>2000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['Model'] == 'RAM 150']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8042</th>
      <td>Dodge</td>
      <td>RAM 150</td>
      <td>1991</td>
      <td>regular unleaded</td>
      <td>125.0</td>
      <td>6.0</td>
      <td>UNKNOWN</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Regular Cab Pickup</td>
      <td>17</td>
      <td>12</td>
      <td>1851</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>8044</th>
      <td>Dodge</td>
      <td>RAM 150</td>
      <td>1991</td>
      <td>regular unleaded</td>
      <td>170.0</td>
      <td>8.0</td>
      <td>MANUAL</td>
      <td>four wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Extended Cab Pickup</td>
      <td>13</td>
      <td>10</td>
      <td>1851</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>8045</th>
      <td>Dodge</td>
      <td>RAM 150</td>
      <td>1991</td>
      <td>regular unleaded</td>
      <td>170.0</td>
      <td>8.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Extended Cab Pickup</td>
      <td>14</td>
      <td>11</td>
      <td>1851</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>8054</th>
      <td>Dodge</td>
      <td>RAM 150</td>
      <td>1992</td>
      <td>regular unleaded</td>
      <td>180.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>four wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Regular Cab Pickup</td>
      <td>16</td>
      <td>11</td>
      <td>1851</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>8055</th>
      <td>Dodge</td>
      <td>RAM 150</td>
      <td>1992</td>
      <td>regular unleaded</td>
      <td>230.0</td>
      <td>8.0</td>
      <td>MANUAL</td>
      <td>four wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Extended Cab Pickup</td>
      <td>15</td>
      <td>12</td>
      <td>1851</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>8056</th>
      <td>Dodge</td>
      <td>RAM 150</td>
      <td>1992</td>
      <td>regular unleaded</td>
      <td>180.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Regular Cab Pickup</td>
      <td>17</td>
      <td>14</td>
      <td>1851</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>8058</th>
      <td>Dodge</td>
      <td>RAM 150</td>
      <td>1992</td>
      <td>regular unleaded</td>
      <td>230.0</td>
      <td>8.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Extended Cab Pickup</td>
      <td>16</td>
      <td>11</td>
      <td>1851</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>8068</th>
      <td>Dodge</td>
      <td>RAM 150</td>
      <td>1993</td>
      <td>regular unleaded</td>
      <td>180.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>four wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Regular Cab Pickup</td>
      <td>16</td>
      <td>11</td>
      <td>1851</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>8071</th>
      <td>Dodge</td>
      <td>RAM 150</td>
      <td>1993</td>
      <td>regular unleaded</td>
      <td>230.0</td>
      <td>8.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Extended Cab Pickup</td>
      <td>16</td>
      <td>12</td>
      <td>1851</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>8072</th>
      <td>Dodge</td>
      <td>RAM 150</td>
      <td>1993</td>
      <td>regular unleaded</td>
      <td>180.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Regular Cab Pickup</td>
      <td>17</td>
      <td>14</td>
      <td>1851</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>8077</th>
      <td>Dodge</td>
      <td>RAM 150</td>
      <td>1993</td>
      <td>regular unleaded</td>
      <td>230.0</td>
      <td>8.0</td>
      <td>MANUAL</td>
      <td>four wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Extended Cab Pickup</td>
      <td>15</td>
      <td>12</td>
      <td>1851</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>8078</th>
      <td>Dodge</td>
      <td>RAM 150</td>
      <td>1993</td>
      <td>regular unleaded</td>
      <td>230.0</td>
      <td>8.0</td>
      <td>MANUAL</td>
      <td>four wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Extended Cab Pickup</td>
      <td>15</td>
      <td>12</td>
      <td>1851</td>
      <td>2083</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[1289,'Transmission Type'] = 'MANUAL'
df.loc[1290,'Transmission Type'] = 'MANUAL'
df.loc[4691,'Transmission Type'] = 'MANUAL'
df.loc[4692,'Transmission Type'] = 'MANUAL'
df.loc[4693,'Transmission Type'] = 'MANUAL'
df.loc[6158,'Transmission Type'] = 'AUTOMATIC'
df.loc[6160,'Transmission Type'] = 'AUTOMATIC'
df.loc[6165,'Transmission Type'] = 'MANUAL'
df.loc[6174,'Transmission Type'] = 'AUTOMATIC'
df.loc[6366,'Transmission Type'] = 'AUTOMATIC'
df.loc[6368,'Transmission Type'] = 'AUTOMATIC'
df.loc[8042,'Transmission Type'] = 'MANUAL'
```


```python
df[df['Transmission Type'] == 'UNKNOWN']
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



<a id='ab-visualsclean'></a>

# Visuals of clean data 


```python
df.head(10)
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BMW</td>
      <td>1 Series M</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>335.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>26</td>
      <td>19</td>
      <td>3916</td>
      <td>46135</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>28</td>
      <td>19</td>
      <td>3916</td>
      <td>40650</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>20</td>
      <td>3916</td>
      <td>36350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>29450</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>34500</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2012</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>31200</td>
    </tr>
    <tr>
      <th>6</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2012</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>26</td>
      <td>17</td>
      <td>3916</td>
      <td>44100</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2012</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>20</td>
      <td>3916</td>
      <td>39300</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2012</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>36900</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2013</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>27</td>
      <td>18</td>
      <td>3916</td>
      <td>37200</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.corr()
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
      <th>Year</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Number of Doors</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Year</th>
      <td>1.000000</td>
      <td>0.334092</td>
      <td>-0.033038</td>
      <td>0.247648</td>
      <td>0.244972</td>
      <td>0.188417</td>
      <td>0.085874</td>
      <td>0.209635</td>
    </tr>
    <tr>
      <th>Engine HP</th>
      <td>0.334092</td>
      <td>1.000000</td>
      <td>0.771169</td>
      <td>-0.129639</td>
      <td>-0.374234</td>
      <td>-0.373267</td>
      <td>0.039185</td>
      <td>0.658245</td>
    </tr>
    <tr>
      <th>Engine Cylinders</th>
      <td>-0.033038</td>
      <td>0.771169</td>
      <td>1.000000</td>
      <td>-0.152048</td>
      <td>-0.610338</td>
      <td>-0.585333</td>
      <td>0.043010</td>
      <td>0.533431</td>
    </tr>
    <tr>
      <th>Number of Doors</th>
      <td>0.247648</td>
      <td>-0.129639</td>
      <td>-0.152048</td>
      <td>1.000000</td>
      <td>0.115311</td>
      <td>0.121194</td>
      <td>-0.057379</td>
      <td>-0.145179</td>
    </tr>
    <tr>
      <th>highway MPG</th>
      <td>0.244972</td>
      <td>-0.374234</td>
      <td>-0.610338</td>
      <td>0.115311</td>
      <td>1.000000</td>
      <td>0.886299</td>
      <td>-0.017159</td>
      <td>-0.166631</td>
    </tr>
    <tr>
      <th>city mpg</th>
      <td>0.188417</td>
      <td>-0.373267</td>
      <td>-0.585333</td>
      <td>0.121194</td>
      <td>0.886299</td>
      <td>1.000000</td>
      <td>-0.000549</td>
      <td>-0.162343</td>
    </tr>
    <tr>
      <th>Popularity</th>
      <td>0.085874</td>
      <td>0.039185</td>
      <td>0.043010</td>
      <td>-0.057379</td>
      <td>-0.017159</td>
      <td>-0.000549</td>
      <td>1.000000</td>
      <td>-0.048371</td>
    </tr>
    <tr>
      <th>MSRP</th>
      <td>0.209635</td>
      <td>0.658245</td>
      <td>0.533431</td>
      <td>-0.145179</td>
      <td>-0.166631</td>
      <td>-0.162343</td>
      <td>-0.048371</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Can see that Engine HP, Engine cylinders, and popularity have strongest correlation with price 
# Make the figsize 7 x 6
plt.figure(figsize=(7,6))

# Plot heatmap of correlations
#sns.heatmap(correlations)
sns.heatmap(df.corr(), annot = True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a13def320>




![png](testcarprediction_files/testcarprediction_132_1.png)



```python
EC = df['Engine Cylinders'].value_counts()
EC.plot.barh()
plt.xlabel('Count')
plt.ylabel('Engine Cylinders')
```




    Text(0,0.5,'Engine Cylinders')




![png](testcarprediction_files/testcarprediction_133_1.png)



```python
sns.set_style('whitegrid')
df['Popularity'].hist(bins=30)
plt.xlabel('Popularity')
plt.ylabel('Count')
```




    Text(0,0.5,'Count')




![png](testcarprediction_files/testcarprediction_134_1.png)



```python
# Create a jointplot showing MSRP versus Popularity.
sns.jointplot(x='Popularity',y='MSRP',data=df)
```




    <seaborn.axisgrid.JointGrid at 0x1a13f45da0>




![png](testcarprediction_files/testcarprediction_135_1.png)



```python
# plot showing the relationship between the independent
# and dependent variables.

sns.lmplot(x='Engine HP', y='MSRP', data=df)
plt.show()
sns.lmplot(x='Engine Cylinders', y='MSRP', data=df)
plt.show()
sns.lmplot(x='Popularity', y='MSRP', data=df)
plt.show()
```


![png](testcarprediction_files/testcarprediction_136_0.png)



![png](testcarprediction_files/testcarprediction_136_1.png)



![png](testcarprediction_files/testcarprediction_136_2.png)



```python
# Create a jointplot showing MSRP versus engine cylinders.
sns.jointplot(x='Engine Cylinders',y='MSRP',data=df)
```




    <seaborn.axisgrid.JointGrid at 0x1a11bc6c88>




![png](testcarprediction_files/testcarprediction_137_1.png)



```python
# Create a jointplot showing MSRP versus Engine HP.
sns.jointplot(x='Engine HP',y='MSRP',data=df)
```




    <seaborn.axisgrid.JointGrid at 0x1a136ceb70>




![png](testcarprediction_files/testcarprediction_138_1.png)



```python
sns.set_style('whitegrid')
df['Engine HP'].hist(bins=30)
plt.xlabel('Engine HP')
```




    Text(0.5,0,'Engine HP')




![png](testcarprediction_files/testcarprediction_139_1.png)



```python
sns.set_style('whitegrid')
df['Engine HP'].hist(bins=30)
plt.xlabel('Engine HP')
```




    Text(0.5,0,'Engine HP')




![png](testcarprediction_files/testcarprediction_140_1.png)


<a id='ab-hp^2'></a>

# Create Engine HP squared column

Seeing if Engine HP squared would perform better fit in model


```python
df['Engine HP^2'] = (df['Engine HP'] * df['Engine HP']).astype(int)
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
      <th>Engine HP^2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BMW</td>
      <td>1 Series M</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>335.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>26</td>
      <td>19</td>
      <td>3916</td>
      <td>46135</td>
      <td>112225</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>28</td>
      <td>19</td>
      <td>3916</td>
      <td>40650</td>
      <td>90000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>20</td>
      <td>3916</td>
      <td>36350</td>
      <td>90000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>29450</td>
      <td>52900</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>34500</td>
      <td>52900</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a jointplot showing MSRP versus Engine HP.
sns.jointplot(x='Engine HP^2',y='MSRP',data=df)
```




    <seaborn.axisgrid.JointGrid at 0x1a13e31128>




![png](testcarprediction_files/testcarprediction_144_1.png)



```python
sns.set_style('whitegrid')
df['Engine HP^2'].hist(bins=30)
plt.xlabel('Engine HP^2')
```




    Text(0.5,0,'Engine HP^2')




![png](testcarprediction_files/testcarprediction_145_1.png)



```python
sns.lmplot(x='Engine HP^2', y='MSRP', data=df)
plt.show()
```


![png](testcarprediction_files/testcarprediction_146_0.png)


<a id='ab-outliers'></a>

# Identifying outliers


```python
# outliers are cars such as the Bugatti which have 16 cylinders and 0ver a thousand HP
# which will have higher MSRP
df[df['Engine HP'] >= 1000]
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
      <th>Engine HP^2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11362</th>
      <td>Bugatti</td>
      <td>Veyron 16.4</td>
      <td>2008</td>
      <td>premium unleaded (required)</td>
      <td>1001.0</td>
      <td>16.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>all wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>14</td>
      <td>8</td>
      <td>820</td>
      <td>2065902</td>
      <td>1002001</td>
    </tr>
    <tr>
      <th>11363</th>
      <td>Bugatti</td>
      <td>Veyron 16.4</td>
      <td>2008</td>
      <td>premium unleaded (required)</td>
      <td>1001.0</td>
      <td>16.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>all wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>14</td>
      <td>8</td>
      <td>820</td>
      <td>1500000</td>
      <td>1002001</td>
    </tr>
    <tr>
      <th>11364</th>
      <td>Bugatti</td>
      <td>Veyron 16.4</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>1001.0</td>
      <td>16.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>all wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>14</td>
      <td>8</td>
      <td>820</td>
      <td>1705769</td>
      <td>1002001</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['MSRP'] >= 150000]
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
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
      <th>Engine HP^2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>294</th>
      <td>Ferrari</td>
      <td>360</td>
      <td>2002</td>
      <td>premium unleaded (required)</td>
      <td>400.0</td>
      <td>8.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>15</td>
      <td>10</td>
      <td>2774</td>
      <td>160829</td>
      <td>160000</td>
    </tr>
    <tr>
      <th>296</th>
      <td>Ferrari</td>
      <td>360</td>
      <td>2002</td>
      <td>premium unleaded (required)</td>
      <td>400.0</td>
      <td>8.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>15</td>
      <td>10</td>
      <td>2774</td>
      <td>150694</td>
      <td>160000</td>
    </tr>
    <tr>
      <th>297</th>
      <td>Ferrari</td>
      <td>360</td>
      <td>2002</td>
      <td>premium unleaded (required)</td>
      <td>400.0</td>
      <td>8.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>15</td>
      <td>10</td>
      <td>2774</td>
      <td>170829</td>
      <td>160000</td>
    </tr>
    <tr>
      <th>298</th>
      <td>Ferrari</td>
      <td>360</td>
      <td>2003</td>
      <td>premium unleaded (required)</td>
      <td>400.0</td>
      <td>8.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>15</td>
      <td>10</td>
      <td>2774</td>
      <td>165986</td>
      <td>160000</td>
    </tr>
    <tr>
      <th>299</th>
      <td>Ferrari</td>
      <td>360</td>
      <td>2003</td>
      <td>premium unleaded (required)</td>
      <td>400.0</td>
      <td>8.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>15</td>
      <td>10</td>
      <td>2774</td>
      <td>154090</td>
      <td>160000</td>
    </tr>
    <tr>
      <th>301</th>
      <td>Ferrari</td>
      <td>360</td>
      <td>2003</td>
      <td>premium unleaded (required)</td>
      <td>400.0</td>
      <td>8.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>15</td>
      <td>10</td>
      <td>2774</td>
      <td>176287</td>
      <td>160000</td>
    </tr>
    <tr>
      <th>302</th>
      <td>Ferrari</td>
      <td>360</td>
      <td>2004</td>
      <td>premium unleaded (required)</td>
      <td>400.0</td>
      <td>8.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>15</td>
      <td>10</td>
      <td>2774</td>
      <td>157767</td>
      <td>160000</td>
    </tr>
    <tr>
      <th>303</th>
      <td>Ferrari</td>
      <td>360</td>
      <td>2004</td>
      <td>premium unleaded (required)</td>
      <td>425.0</td>
      <td>8.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>15</td>
      <td>10</td>
      <td>2774</td>
      <td>187124</td>
      <td>180625</td>
    </tr>
    <tr>
      <th>305</th>
      <td>Ferrari</td>
      <td>360</td>
      <td>2004</td>
      <td>premium unleaded (required)</td>
      <td>400.0</td>
      <td>8.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>15</td>
      <td>10</td>
      <td>2774</td>
      <td>169900</td>
      <td>160000</td>
    </tr>
    <tr>
      <th>306</th>
      <td>Ferrari</td>
      <td>360</td>
      <td>2004</td>
      <td>premium unleaded (required)</td>
      <td>400.0</td>
      <td>8.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>15</td>
      <td>10</td>
      <td>2774</td>
      <td>180408</td>
      <td>160000</td>
    </tr>
    <tr>
      <th>460</th>
      <td>Ferrari</td>
      <td>456M</td>
      <td>2001</td>
      <td>premium unleaded (required)</td>
      <td>442.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>14</td>
      <td>9</td>
      <td>2774</td>
      <td>223970</td>
      <td>195364</td>
    </tr>
    <tr>
      <th>461</th>
      <td>Ferrari</td>
      <td>456M</td>
      <td>2001</td>
      <td>premium unleaded (required)</td>
      <td>442.0</td>
      <td>12.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>15</td>
      <td>9</td>
      <td>2774</td>
      <td>219775</td>
      <td>195364</td>
    </tr>
    <tr>
      <th>462</th>
      <td>Ferrari</td>
      <td>456M</td>
      <td>2002</td>
      <td>premium unleaded (required)</td>
      <td>442.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>14</td>
      <td>9</td>
      <td>2774</td>
      <td>228625</td>
      <td>195364</td>
    </tr>
    <tr>
      <th>463</th>
      <td>Ferrari</td>
      <td>456M</td>
      <td>2002</td>
      <td>premium unleaded (required)</td>
      <td>442.0</td>
      <td>12.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>15</td>
      <td>9</td>
      <td>2774</td>
      <td>224585</td>
      <td>195364</td>
    </tr>
    <tr>
      <th>464</th>
      <td>Ferrari</td>
      <td>456M</td>
      <td>2003</td>
      <td>premium unleaded (required)</td>
      <td>442.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>14</td>
      <td>9</td>
      <td>2774</td>
      <td>228625</td>
      <td>195364</td>
    </tr>
    <tr>
      <th>465</th>
      <td>Ferrari</td>
      <td>456M</td>
      <td>2003</td>
      <td>premium unleaded (required)</td>
      <td>442.0</td>
      <td>12.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>15</td>
      <td>9</td>
      <td>2774</td>
      <td>224585</td>
      <td>195364</td>
    </tr>
    <tr>
      <th>466</th>
      <td>Ferrari</td>
      <td>458 Italia</td>
      <td>2013</td>
      <td>premium unleaded (required)</td>
      <td>562.0</td>
      <td>8.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>17</td>
      <td>13</td>
      <td>2774</td>
      <td>257412</td>
      <td>315844</td>
    </tr>
    <tr>
      <th>467</th>
      <td>Ferrari</td>
      <td>458 Italia</td>
      <td>2013</td>
      <td>premium unleaded (required)</td>
      <td>562.0</td>
      <td>8.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>17</td>
      <td>13</td>
      <td>2774</td>
      <td>233509</td>
      <td>315844</td>
    </tr>
    <tr>
      <th>468</th>
      <td>Ferrari</td>
      <td>458 Italia</td>
      <td>2014</td>
      <td>premium unleaded (required)</td>
      <td>562.0</td>
      <td>8.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>17</td>
      <td>13</td>
      <td>2774</td>
      <td>233509</td>
      <td>315844</td>
    </tr>
    <tr>
      <th>469</th>
      <td>Ferrari</td>
      <td>458 Italia</td>
      <td>2014</td>
      <td>premium unleaded (required)</td>
      <td>597.0</td>
      <td>8.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>17</td>
      <td>13</td>
      <td>2774</td>
      <td>288000</td>
      <td>356409</td>
    </tr>
    <tr>
      <th>470</th>
      <td>Ferrari</td>
      <td>458 Italia</td>
      <td>2014</td>
      <td>premium unleaded (required)</td>
      <td>562.0</td>
      <td>8.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>17</td>
      <td>13</td>
      <td>2774</td>
      <td>257412</td>
      <td>315844</td>
    </tr>
    <tr>
      <th>471</th>
      <td>Ferrari</td>
      <td>458 Italia</td>
      <td>2015</td>
      <td>premium unleaded (required)</td>
      <td>562.0</td>
      <td>8.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>17</td>
      <td>13</td>
      <td>2774</td>
      <td>239340</td>
      <td>315844</td>
    </tr>
    <tr>
      <th>472</th>
      <td>Ferrari</td>
      <td>458 Italia</td>
      <td>2015</td>
      <td>premium unleaded (required)</td>
      <td>562.0</td>
      <td>8.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>17</td>
      <td>13</td>
      <td>2774</td>
      <td>263553</td>
      <td>315844</td>
    </tr>
    <tr>
      <th>473</th>
      <td>Ferrari</td>
      <td>458 Italia</td>
      <td>2015</td>
      <td>premium unleaded (required)</td>
      <td>597.0</td>
      <td>8.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>17</td>
      <td>13</td>
      <td>2774</td>
      <td>291744</td>
      <td>356409</td>
    </tr>
    <tr>
      <th>598</th>
      <td>Ferrari</td>
      <td>550</td>
      <td>2001</td>
      <td>premium unleaded (required)</td>
      <td>485.0</td>
      <td>12.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>12</td>
      <td>8</td>
      <td>2774</td>
      <td>248500</td>
      <td>235225</td>
    </tr>
    <tr>
      <th>599</th>
      <td>Ferrari</td>
      <td>550</td>
      <td>2001</td>
      <td>premium unleaded (required)</td>
      <td>485.0</td>
      <td>12.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>12</td>
      <td>8</td>
      <td>2774</td>
      <td>205840</td>
      <td>235225</td>
    </tr>
    <tr>
      <th>604</th>
      <td>McLaren</td>
      <td>570S</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>562.0</td>
      <td>8.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>23</td>
      <td>16</td>
      <td>416</td>
      <td>184900</td>
      <td>315844</td>
    </tr>
    <tr>
      <th>605</th>
      <td>Ferrari</td>
      <td>575M</td>
      <td>2002</td>
      <td>premium unleaded (required)</td>
      <td>515.0</td>
      <td>12.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>15</td>
      <td>9</td>
      <td>2774</td>
      <td>214670</td>
      <td>265225</td>
    </tr>
    <tr>
      <th>606</th>
      <td>Ferrari</td>
      <td>575M</td>
      <td>2002</td>
      <td>premium unleaded (required)</td>
      <td>515.0</td>
      <td>12.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>16</td>
      <td>9</td>
      <td>2774</td>
      <td>224670</td>
      <td>265225</td>
    </tr>
    <tr>
      <th>607</th>
      <td>Ferrari</td>
      <td>575M</td>
      <td>2003</td>
      <td>premium unleaded (required)</td>
      <td>515.0</td>
      <td>12.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>15</td>
      <td>9</td>
      <td>2774</td>
      <td>217890</td>
      <td>265225</td>
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
    </tr>
    <tr>
      <th>11092</th>
      <td>Aston Martin</td>
      <td>V12 Vanquish</td>
      <td>2004</td>
      <td>premium unleaded (required)</td>
      <td>460.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>18</td>
      <td>11</td>
      <td>259</td>
      <td>234260</td>
      <td>211600</td>
    </tr>
    <tr>
      <th>11093</th>
      <td>Aston Martin</td>
      <td>V12 Vanquish</td>
      <td>2005</td>
      <td>premium unleaded (required)</td>
      <td>460.0</td>
      <td>12.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>16</td>
      <td>10</td>
      <td>259</td>
      <td>234260</td>
      <td>211600</td>
    </tr>
    <tr>
      <th>11094</th>
      <td>Aston Martin</td>
      <td>V12 Vanquish</td>
      <td>2005</td>
      <td>premium unleaded (required)</td>
      <td>520.0</td>
      <td>12.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>16</td>
      <td>10</td>
      <td>259</td>
      <td>255000</td>
      <td>270400</td>
    </tr>
    <tr>
      <th>11095</th>
      <td>Aston Martin</td>
      <td>V12 Vanquish</td>
      <td>2006</td>
      <td>premium unleaded (required)</td>
      <td>520.0</td>
      <td>12.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>16</td>
      <td>10</td>
      <td>259</td>
      <td>260000</td>
      <td>270400</td>
    </tr>
    <tr>
      <th>11096</th>
      <td>Aston Martin</td>
      <td>V12 Vantage S</td>
      <td>2015</td>
      <td>premium unleaded (required)</td>
      <td>565.0</td>
      <td>12.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>18</td>
      <td>12</td>
      <td>259</td>
      <td>182395</td>
      <td>319225</td>
    </tr>
    <tr>
      <th>11097</th>
      <td>Aston Martin</td>
      <td>V12 Vantage S</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>565.0</td>
      <td>12.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>18</td>
      <td>12</td>
      <td>259</td>
      <td>198195</td>
      <td>319225</td>
    </tr>
    <tr>
      <th>11098</th>
      <td>Aston Martin</td>
      <td>V12 Vantage S</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>565.0</td>
      <td>12.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>18</td>
      <td>12</td>
      <td>259</td>
      <td>183695</td>
      <td>319225</td>
    </tr>
    <tr>
      <th>11099</th>
      <td>Aston Martin</td>
      <td>V12 Vantage</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>510.0</td>
      <td>12.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>17</td>
      <td>11</td>
      <td>259</td>
      <td>191995</td>
      <td>260100</td>
    </tr>
    <tr>
      <th>11100</th>
      <td>Aston Martin</td>
      <td>V12 Vantage</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>510.0</td>
      <td>12.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>17</td>
      <td>11</td>
      <td>259</td>
      <td>180535</td>
      <td>260100</td>
    </tr>
    <tr>
      <th>11101</th>
      <td>Aston Martin</td>
      <td>V12 Vantage</td>
      <td>2012</td>
      <td>premium unleaded (required)</td>
      <td>510.0</td>
      <td>12.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>17</td>
      <td>11</td>
      <td>259</td>
      <td>180535</td>
      <td>260100</td>
    </tr>
    <tr>
      <th>11102</th>
      <td>Aston Martin</td>
      <td>V12 Vantage</td>
      <td>2012</td>
      <td>premium unleaded (required)</td>
      <td>510.0</td>
      <td>12.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>17</td>
      <td>11</td>
      <td>259</td>
      <td>195895</td>
      <td>260100</td>
    </tr>
    <tr>
      <th>11153</th>
      <td>Aston Martin</td>
      <td>V8 Vantage</td>
      <td>2014</td>
      <td>premium unleaded (required)</td>
      <td>430.0</td>
      <td>8.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>21</td>
      <td>14</td>
      <td>259</td>
      <td>150900</td>
      <td>184900</td>
    </tr>
    <tr>
      <th>11159</th>
      <td>Aston Martin</td>
      <td>V8 Vantage</td>
      <td>2015</td>
      <td>premium unleaded (required)</td>
      <td>430.0</td>
      <td>8.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>21</td>
      <td>14</td>
      <td>259</td>
      <td>153195</td>
      <td>184900</td>
    </tr>
    <tr>
      <th>11173</th>
      <td>Aston Martin</td>
      <td>V8 Vantage</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>430.0</td>
      <td>8.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>21</td>
      <td>14</td>
      <td>259</td>
      <td>154495</td>
      <td>184900</td>
    </tr>
    <tr>
      <th>11206</th>
      <td>Aston Martin</td>
      <td>Vanquish</td>
      <td>2014</td>
      <td>premium unleaded (required)</td>
      <td>565.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>Convertible</td>
      <td>19</td>
      <td>13</td>
      <td>259</td>
      <td>296295</td>
      <td>319225</td>
    </tr>
    <tr>
      <th>11207</th>
      <td>Aston Martin</td>
      <td>Vanquish</td>
      <td>2014</td>
      <td>premium unleaded (required)</td>
      <td>565.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>Coupe</td>
      <td>19</td>
      <td>13</td>
      <td>259</td>
      <td>278295</td>
      <td>319225</td>
    </tr>
    <tr>
      <th>11208</th>
      <td>Aston Martin</td>
      <td>Vanquish</td>
      <td>2015</td>
      <td>premium unleaded (required)</td>
      <td>568.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>Convertible</td>
      <td>21</td>
      <td>13</td>
      <td>259</td>
      <td>301695</td>
      <td>322624</td>
    </tr>
    <tr>
      <th>11209</th>
      <td>Aston Martin</td>
      <td>Vanquish</td>
      <td>2015</td>
      <td>premium unleaded (required)</td>
      <td>568.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>Coupe</td>
      <td>21</td>
      <td>13</td>
      <td>259</td>
      <td>283695</td>
      <td>322624</td>
    </tr>
    <tr>
      <th>11210</th>
      <td>Aston Martin</td>
      <td>Vanquish</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>568.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>Coupe</td>
      <td>21</td>
      <td>13</td>
      <td>259</td>
      <td>287650</td>
      <td>322624</td>
    </tr>
    <tr>
      <th>11211</th>
      <td>Aston Martin</td>
      <td>Vanquish</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>568.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>Convertible</td>
      <td>21</td>
      <td>13</td>
      <td>259</td>
      <td>305650</td>
      <td>322624</td>
    </tr>
    <tr>
      <th>11212</th>
      <td>Aston Martin</td>
      <td>Vanquish</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>568.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>Coupe</td>
      <td>21</td>
      <td>13</td>
      <td>259</td>
      <td>302695</td>
      <td>322624</td>
    </tr>
    <tr>
      <th>11213</th>
      <td>Aston Martin</td>
      <td>Vanquish</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>568.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>Convertible</td>
      <td>21</td>
      <td>13</td>
      <td>259</td>
      <td>320695</td>
      <td>322624</td>
    </tr>
    <tr>
      <th>11362</th>
      <td>Bugatti</td>
      <td>Veyron 16.4</td>
      <td>2008</td>
      <td>premium unleaded (required)</td>
      <td>1001.0</td>
      <td>16.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>all wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>14</td>
      <td>8</td>
      <td>820</td>
      <td>2065902</td>
      <td>1002001</td>
    </tr>
    <tr>
      <th>11363</th>
      <td>Bugatti</td>
      <td>Veyron 16.4</td>
      <td>2008</td>
      <td>premium unleaded (required)</td>
      <td>1001.0</td>
      <td>16.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>all wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>14</td>
      <td>8</td>
      <td>820</td>
      <td>1500000</td>
      <td>1002001</td>
    </tr>
    <tr>
      <th>11364</th>
      <td>Bugatti</td>
      <td>Veyron 16.4</td>
      <td>2009</td>
      <td>premium unleaded (required)</td>
      <td>1001.0</td>
      <td>16.0</td>
      <td>AUTOMATED_MANUAL</td>
      <td>all wheel drive</td>
      <td>2.0</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>14</td>
      <td>8</td>
      <td>820</td>
      <td>1705769</td>
      <td>1002001</td>
    </tr>
    <tr>
      <th>11394</th>
      <td>Aston Martin</td>
      <td>Virage</td>
      <td>2012</td>
      <td>premium unleaded (required)</td>
      <td>490.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>Coupe</td>
      <td>18</td>
      <td>13</td>
      <td>259</td>
      <td>208295</td>
      <td>240100</td>
    </tr>
    <tr>
      <th>11395</th>
      <td>Aston Martin</td>
      <td>Virage</td>
      <td>2012</td>
      <td>premium unleaded (required)</td>
      <td>490.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Midsize</td>
      <td>Convertible</td>
      <td>18</td>
      <td>13</td>
      <td>259</td>
      <td>223295</td>
      <td>240100</td>
    </tr>
    <tr>
      <th>11448</th>
      <td>Rolls-Royce</td>
      <td>Wraith</td>
      <td>2014</td>
      <td>premium unleaded (required)</td>
      <td>624.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Coupe</td>
      <td>21</td>
      <td>13</td>
      <td>86</td>
      <td>284900</td>
      <td>389376</td>
    </tr>
    <tr>
      <th>11449</th>
      <td>Rolls-Royce</td>
      <td>Wraith</td>
      <td>2015</td>
      <td>premium unleaded (required)</td>
      <td>624.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Coupe</td>
      <td>21</td>
      <td>13</td>
      <td>86</td>
      <td>294025</td>
      <td>389376</td>
    </tr>
    <tr>
      <th>11450</th>
      <td>Rolls-Royce</td>
      <td>Wraith</td>
      <td>2016</td>
      <td>premium unleaded (required)</td>
      <td>624.0</td>
      <td>12.0</td>
      <td>AUTOMATIC</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Large</td>
      <td>Coupe</td>
      <td>21</td>
      <td>13</td>
      <td>86</td>
      <td>304350</td>
      <td>389376</td>
    </tr>
  </tbody>
</table>
<p>408 rows × 16 columns</p>
</div>



# Save data


```python
# Save undummy data
df1 = df.to_csv('carnotdummied.csv')
```

<a id='ab-dummies'></a>

# Dummies for categorical columns


```python
df = pd.get_dummies(df, columns=['Make','Model', 'Engine Fuel Type', 'Transmission Type', 'Driven_Wheels',
                            'Vehicle Size', 'Vehicle Style'])
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
      <th>Year</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Number of Doors</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
      <th>Engine HP^2</th>
      <th>Make_Acura</th>
      <th>Make_Alfa Romeo</th>
      <th>Make_Aston Martin</th>
      <th>Make_Audi</th>
      <th>Make_BMW</th>
      <th>Make_Bentley</th>
      <th>Make_Bugatti</th>
      <th>Make_Buick</th>
      <th>Make_Cadillac</th>
      <th>Make_Chevrolet</th>
      <th>Make_Chrysler</th>
      <th>Make_Dodge</th>
      <th>Make_FIAT</th>
      <th>Make_Ferrari</th>
      <th>Make_Ford</th>
      <th>Make_GMC</th>
      <th>Make_Genesis</th>
      <th>Make_HUMMER</th>
      <th>Make_Honda</th>
      <th>Make_Hyundai</th>
      <th>Make_Infiniti</th>
      <th>Make_Kia</th>
      <th>Make_Lamborghini</th>
      <th>Make_Land Rover</th>
      <th>Make_Lexus</th>
      <th>Make_Lincoln</th>
      <th>Make_Lotus</th>
      <th>Make_Maserati</th>
      <th>Make_Maybach</th>
      <th>Make_Mazda</th>
      <th>Make_McLaren</th>
      <th>Make_Mercedes-Benz</th>
      <th>Make_Mitsubishi</th>
      <th>Make_Nissan</th>
      <th>Make_Oldsmobile</th>
      <th>Make_Plymouth</th>
      <th>Make_Pontiac</th>
      <th>Make_Porsche</th>
      <th>Make_Rolls-Royce</th>
      <th>Make_Saab</th>
      <th>Make_Scion</th>
      <th>...</th>
      <th>Model_Zephyr</th>
      <th>Model_allroad</th>
      <th>Model_allroad quattro</th>
      <th>Model_e-Golf</th>
      <th>Model_i-MiEV</th>
      <th>Model_i3</th>
      <th>Model_iA</th>
      <th>Model_iM</th>
      <th>Model_iQ</th>
      <th>Model_tC</th>
      <th>Model_xA</th>
      <th>Model_xB</th>
      <th>Model_xD</th>
      <th>Engine Fuel Type_diesel</th>
      <th>Engine Fuel Type_electric</th>
      <th>Engine Fuel Type_flex-fuel (premium unleaded recommended/E85)</th>
      <th>Engine Fuel Type_flex-fuel (premium unleaded required/E85)</th>
      <th>Engine Fuel Type_flex-fuel (unleaded/E85)</th>
      <th>Engine Fuel Type_flex-fuel (unleaded/natural gas)</th>
      <th>Engine Fuel Type_natural gas</th>
      <th>Engine Fuel Type_premium unleaded (recommended)</th>
      <th>Engine Fuel Type_premium unleaded (required)</th>
      <th>Engine Fuel Type_regular unleaded</th>
      <th>Transmission Type_AUTOMATED_MANUAL</th>
      <th>Transmission Type_AUTOMATIC</th>
      <th>Transmission Type_DIRECT_DRIVE</th>
      <th>Transmission Type_MANUAL</th>
      <th>Driven_Wheels_all wheel drive</th>
      <th>Driven_Wheels_four wheel drive</th>
      <th>Driven_Wheels_front wheel drive</th>
      <th>Driven_Wheels_rear wheel drive</th>
      <th>Vehicle Size_Compact</th>
      <th>Vehicle Size_Large</th>
      <th>Vehicle Size_Midsize</th>
      <th>Vehicle Style_2dr Hatchback</th>
      <th>Vehicle Style_2dr SUV</th>
      <th>Vehicle Style_4dr Hatchback</th>
      <th>Vehicle Style_4dr SUV</th>
      <th>Vehicle Style_Cargo Minivan</th>
      <th>Vehicle Style_Cargo Van</th>
      <th>Vehicle Style_Convertible</th>
      <th>Vehicle Style_Convertible SUV</th>
      <th>Vehicle Style_Coupe</th>
      <th>Vehicle Style_Crew Cab Pickup</th>
      <th>Vehicle Style_Extended Cab Pickup</th>
      <th>Vehicle Style_Passenger Minivan</th>
      <th>Vehicle Style_Passenger Van</th>
      <th>Vehicle Style_Regular Cab Pickup</th>
      <th>Vehicle Style_Sedan</th>
      <th>Vehicle Style_Wagon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011</td>
      <td>335.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>26</td>
      <td>19</td>
      <td>3916</td>
      <td>46135</td>
      <td>112225</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>28</td>
      <td>19</td>
      <td>3916</td>
      <td>40650</td>
      <td>90000</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>28</td>
      <td>20</td>
      <td>3916</td>
      <td>36350</td>
      <td>90000</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>29450</td>
      <td>52900</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>34500</td>
      <td>52900</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1014 columns</p>
</div>




```python
df.to_csv('cardummied.csv')
```

<a id='ab-train/test'></a>

# Train test spit


```python
# Create separate object for target variable
y = df.MSRP

# Create separate object for input features
X = df.drop('MSRP', axis=1)
```


```python
# Split X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, # set aside 20% of observations for the test set.
                                                    random_state=1234)
```


```python
# verify test and train set shape 
X_test.shape,y_train.shape
```




    ((2240, 1013), (8959,))




```python
# Print number of observations in X_train, X_test, y_train, and y_test
print(len(X_train), len(X_test), len(y_train), len(y_test))
```

    8959 2240 8959 2240



```python
X_cols = df.drop('MSRP', axis=1).columns
```


```python
# MSE - the average absolute difference between predicted and actual values for our target variable.
# R2 - The percentt of the variation in the target variable that can be explained by the model
```

<a id='ab-rf'></a>


# Pipeline with Random Forest Model


```python
#initialize randomforest and selectKbest
selector = SelectKBest(f_regression)  # select k best
clf = RandomForestRegressor() # Model I want to use 

#place SelectKbest transformer and RandomForest estimator into Pipeine
pipe = Pipeline(steps=[
#     ('poly', PolynomialFeatures()), # Did not need because I created dummies for categorial columns which made to many columns 
    ('selector', selector),  # feature selection
    ('clf', clf) # Model
])

#Create the parameter grid, entering the values to use for each parameter selected in the RandomForest estimator
parameters = {
    'selector__k':[50,100], # params to search through
#     'poly__degree': [2],
    'clf__n_estimators':[20, 100,150], # Start, stop, number of trees
    'clf__min_samples_split': [5], # max number of samples required to split an internal node:
    'clf__max_features': ['auto'], # max number of features considered for splitting a node
    'clf__max_depth': [ 3, 5, 7] # max number of splits per tree
} 


#Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
g_search = GridSearchCV(pipe, parameters, cv=3, n_jobs=1, verbose=2)

#Fit the grid search object to the training data and find the optimal parameters using fit()
g_fit = g_search.fit(X_train, y_train)

#Get the best estimator and print out the estimator model
best_clf = g_fit.best_estimator_
print (best_clf)

#Use best estimator to make predictions on the test set
best_predictions = best_clf.predict(X_test)


#metrics
#print(mean_absolute_error(y_true = y_test, y_pred = best_predictions))
#print(r2_score(y_true = y_test, y_pred = best_predictions))


print("MAE: " + str(mean_absolute_error(y_true = y_test, y_pred = best_predictions)))
print("R2 Score: " + str(r2_score(y_true = y_test, y_pred = best_predictions)))
```

    Fitting 3 folds for each of 18 candidates, totalling 54 fits
    [CV] clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)
    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.3s remaining:    0.0s


    [CV]  clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=50, total=   0.3s
    [CV] clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=50, total=   0.3s
    [CV] clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=50, total=   0.2s
    [CV] clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=100, total=   0.3s
    [CV] clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=100, total=   0.2s
    [CV] clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=100, total=   0.3s
    [CV] clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=50, total=   0.7s
    [CV] clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=50, total=   0.6s
    [CV] clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=50, total=   0.6s
    [CV] clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=100, total=   0.8s
    [CV] clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=100, total=   0.7s
    [CV] clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=100, total=   0.8s
    [CV] clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=50, total=   0.8s
    [CV] clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=50, total=   0.9s
    [CV] clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=50, total=   0.8s
    [CV] clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=100, total=   1.1s
    [CV] clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=100, total=   1.1s
    [CV] clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=3, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=100, total=   1.1s
    [CV] clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=50, total=   0.2s
    [CV] clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=50, total=   0.2s
    [CV] clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=50, total=   0.2s
    [CV] clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=100, total=   0.3s
    [CV] clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=100, total=   0.3s
    [CV] clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=100, total=   0.3s
    [CV] clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=50, total=   0.8s
    [CV] clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=50, total=   0.9s
    [CV] clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=50, total=   0.8s
    [CV] clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=100, total=   1.1s
    [CV] clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=100, total=   1.0s
    [CV] clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=100, total=   1.1s
    [CV] clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=50, total=   1.2s
    [CV] clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=50, total=   1.1s
    [CV] clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=50, total=   1.2s
    [CV] clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=100, total=   1.5s
    [CV] clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=100, total=   1.4s
    [CV] clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=5, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=100, total=   1.5s
    [CV] clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=50, total=   0.3s
    [CV] clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=50, total=   0.4s
    [CV] clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=50, total=   0.3s
    [CV] clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=100, total=   0.4s
    [CV] clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=100, total=   0.3s
    [CV] clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=20, selector__k=100, total=   0.4s
    [CV] clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=50, total=   1.1s
    [CV] clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=50, total=   1.2s
    [CV] clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=50, total=   1.1s
    [CV] clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=100, total=   1.6s
    [CV] clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=100, total=   1.4s
    [CV] clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=100, selector__k=100, total=   1.3s
    [CV] clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=50, total=   1.4s
    [CV] clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=50, total=   1.4s
    [CV] clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=50, total=   1.4s
    [CV] clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=100, total=   2.2s
    [CV] clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=100, total=   2.1s
    [CV] clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__max_depth=7, clf__max_features=auto, clf__min_samples_split=5, clf__n_estimators=150, selector__k=100, total=   2.1s


    [Parallel(n_jobs=1)]: Done  54 out of  54 | elapsed:   49.7s finished
    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    Pipeline(memory=None,
         steps=[('selector', SelectKBest(k=100, score_func=<function f_regression at 0x109eeb950>)), ('clf', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=7,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=5,
               min_weight_fraction_leaf=0.0, n_estimators=150, n_jobs=1,
               oob_score=False, random_state=None, verbose=0, warm_start=False))])
    MAE: 6898.5361470355265
    R2 Score: 0.9249219699040897



```python
rf_pred = g_search.predict(X_test)
plt.scatter(rf_pred, y_test)
plt.plot(y_test, y_test)
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()
```


![png](testcarprediction_files/testcarprediction_164_0.png)


<a id='ab-gb'></a>

# Pipeline with Gradient Boost


```python
#initialize gradient boosting and selectKbest
selector = SelectKBest(f_regression)  # select k best
clf = GradientBoostingRegressor() # Model I want to use 

#place SelectKbest transformer and RandomForest estimator into Pipeine
pipe = Pipeline(steps=[
    ('Scale',StandardScaler()),
    #('poly', PolynomialFeatures()),
    ('selector', selector), 
    ('clf', clf)
])

#Create the parameter grid, entering the values to use for each parameter selected in the RandomForest estimator
parameters = {
    'selector__k':[50,100], # params to search through
    #'poly__degree': [2], 
    'clf__n_estimators': [20, 100], # num of boosting stages to perform. larger number usually better performance
    'clf__learning_rate':[0.05, 0.1, 0.2], # shrinks the contibution of each tree. trade off between n_estimators and learning rate
    'clf__max_depth': [1, 3, 5] # max depth of the individual regression estimators
}

#Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
g_search = GridSearchCV(pipe, parameters, cv=3, n_jobs=1, verbose=2)

#Fit the grid search object to the training data and find the optimal parameters using fit()
g_fit = g_search.fit(X_train, y_train)

#Get the best estimator and print out the estimator model
best_clf = g_fit.best_estimator_
print (best_clf)

#Use best estimator to make predictions on the test set
best_predictions = best_clf.predict(X_test)


#metrics
#print(mean_absolute_error(y_true = y_test, y_pred = best_predictions))
#print(r2_score(y_true = y_test, y_pred = best_predictions))

print("MAE: " + str(mean_absolute_error(y_true = y_test, y_pred = best_predictions)))
print("R2 Score: " + str(r2_score(y_true = y_test, y_pred = best_predictions)))
```

    Fitting 3 folds for each of 36 candidates, totalling 108 fits
    [CV] clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=20, selector__k=50, total=   0.4s
    [CV] clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=20, selector__k=50 


    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.5s remaining:    0.0s
    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=20, selector__k=50, total=   0.4s
    [CV] clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=20, selector__k=50, total=   0.3s
    [CV] clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=20, selector__k=100, total=   0.4s
    [CV] clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=20, selector__k=100, total=   0.4s
    [CV] clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=20, selector__k=100, total=   0.3s
    [CV] clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=100, selector__k=50, total=   0.4s
    [CV] clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=100, selector__k=50, total=   0.4s
    [CV] clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=100, selector__k=50, total=   0.4s
    [CV] clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=100, selector__k=100, total=   0.6s
    [CV] clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=100, selector__k=100, total=   0.7s
    [CV] clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=100, selector__k=100, total=   0.6s
    [CV] clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=20, selector__k=50, total=   0.5s
    [CV] clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=20, selector__k=50, total=   0.4s
    [CV] clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=20, selector__k=50, total=   0.4s
    [CV] clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=20, selector__k=100, total=   0.5s
    [CV] clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=20, selector__k=100, total=   0.5s
    [CV] clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=20, selector__k=100, total=   0.5s
    [CV] clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=100, selector__k=50, total=   0.9s
    [CV] clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=100, selector__k=50, total=   0.9s
    [CV] clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=100, selector__k=50, total=   0.8s
    [CV] clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=100, selector__k=100, total=   1.6s
    [CV] clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=100, selector__k=100, total=   1.5s
    [CV] clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=100, selector__k=100, total=   1.2s
    [CV] clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=20, selector__k=50, total=   0.5s
    [CV] clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=20, selector__k=50, total=   0.7s
    [CV] clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=20, selector__k=50, total=   0.6s
    [CV] clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=20, selector__k=100, total=   0.7s
    [CV] clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=20, selector__k=100, total=   0.7s
    [CV] clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=20, selector__k=100, total=   0.7s
    [CV] clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=100, selector__k=50, total=   1.9s
    [CV] clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=100, selector__k=50, total=   1.4s
    [CV] clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=100, selector__k=50, total=   1.4s
    [CV] clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=100, selector__k=100, total=   2.5s
    [CV] clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=100, selector__k=100, total=   2.5s
    [CV] clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=100, selector__k=100, total=   2.7s
    [CV] clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=20, selector__k=50, total=   0.3s
    [CV] clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=20, selector__k=50, total=   0.3s
    [CV] clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=20, selector__k=50, total=   0.3s
    [CV] clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=20, selector__k=100, total=   0.3s
    [CV] clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=20, selector__k=100, total=   0.3s
    [CV] clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=20, selector__k=100, total=   0.3s
    [CV] clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=100, selector__k=50, total=   0.4s
    [CV] clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=100, selector__k=50, total=   0.4s
    [CV] clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=100, selector__k=50, total=   0.4s
    [CV] clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=100, selector__k=100, total=   0.5s
    [CV] clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=100, selector__k=100, total=   0.5s
    [CV] clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=100, selector__k=100, total=   0.5s
    [CV] clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=20, selector__k=50, total=   0.4s
    [CV] clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=20, selector__k=50, total=   0.4s
    [CV] clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=20, selector__k=50, total=   0.4s
    [CV] clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=20, selector__k=100, total=   0.5s
    [CV] clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=20, selector__k=100, total=   0.5s
    [CV] clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=20, selector__k=100, total=   0.5s
    [CV] clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=100, selector__k=50, total=   0.8s
    [CV] clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=100, selector__k=50, total=   0.8s
    [CV] clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=100, selector__k=50, total=   0.8s
    [CV] clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=100, selector__k=100, total=   1.5s
    [CV] clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=100, selector__k=100, total=   1.3s
    [CV] clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=100, selector__k=100, total=   1.3s
    [CV] clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=20, selector__k=50, total=   0.5s
    [CV] clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=20, selector__k=50, total=   0.5s
    [CV] clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=20, selector__k=50, total=   0.6s
    [CV] clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=20, selector__k=100, total=   0.8s
    [CV] clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=20, selector__k=100, total=   0.7s
    [CV] clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=20, selector__k=100, total=   0.7s
    [CV] clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=100, selector__k=50, total=   1.4s
    [CV] clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=100, selector__k=50, total=   1.3s
    [CV] clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=100, selector__k=50, total=   1.5s
    [CV] clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=100, selector__k=100, total=   2.5s
    [CV] clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=100, selector__k=100, total=   2.1s
    [CV] clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=100, selector__k=100, total=   2.2s
    [CV] clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=20, selector__k=50, total=   0.3s
    [CV] clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=20, selector__k=50, total=   0.3s
    [CV] clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=20, selector__k=50, total=   0.3s
    [CV] clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=20, selector__k=100, total=   0.3s
    [CV] clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=20, selector__k=100, total=   0.4s
    [CV] clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=20, selector__k=100, total=   0.4s
    [CV] clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=100, selector__k=50, total=   0.5s
    [CV] clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=100, selector__k=50, total=   0.5s
    [CV] clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=100, selector__k=50, total=   0.5s
    [CV] clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=100, selector__k=100, total=   0.7s
    [CV] clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=100, selector__k=100, total=   0.6s
    [CV] clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=100, selector__k=100, total=   0.6s
    [CV] clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=20, selector__k=50, total=   0.4s
    [CV] clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=20, selector__k=50, total=   0.4s
    [CV] clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=20, selector__k=50, total=   0.4s
    [CV] clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=20, selector__k=100, total=   0.5s
    [CV] clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=20, selector__k=100, total=   0.5s
    [CV] clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=20, selector__k=100, total=   0.5s
    [CV] clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=100, selector__k=50, total=   0.8s
    [CV] clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=100, selector__k=50, total=   0.8s
    [CV] clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=100, selector__k=50, total=   0.8s
    [CV] clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=100, selector__k=100, total=   1.2s
    [CV] clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=100, selector__k=100, total=   1.4s
    [CV] clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=100, selector__k=100, total=   1.2s
    [CV] clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=20, selector__k=50, total=   0.6s
    [CV] clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=20, selector__k=50, total=   0.6s
    [CV] clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=20, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=20, selector__k=50, total=   0.8s
    [CV] clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=20, selector__k=100, total=   1.0s
    [CV] clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=20, selector__k=100, total=   0.7s
    [CV] clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=20, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=20, selector__k=100, total=   0.9s
    [CV] clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=100, selector__k=50, total=   1.7s
    [CV] clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=100, selector__k=50, total=   1.4s
    [CV] clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=100, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=100, selector__k=50, total=   1.4s
    [CV] clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=100, selector__k=100, total=   2.2s
    [CV] clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=100, selector__k=100, total=   2.1s
    [CV] clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=100, selector__k=100 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=100, selector__k=100, total=   2.4s


    [Parallel(n_jobs=1)]: Done 108 out of 108 | elapsed:  1.7min finished
    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    Pipeline(memory=None,
         steps=[('Scale', StandardScaler(copy=True, with_mean=True, with_std=True)), ('selector', SelectKBest(k=50, score_func=<function f_regression at 0x109eeb950>)), ('clf', GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.2, loss='ls', max_depth=5, ma...s=100, presort='auto', random_state=None,
                 subsample=1.0, verbose=0, warm_start=False))])
    MAE: 4569.530409739521
    R2 Score: 0.9531386762746651



```python
len(best_clf.steps[1][1].get_support())
```




    1013




```python
feature = list(X_cols[best_clf.steps[1][1].get_support()])
```


```python
X_cols = df.drop('MSRP', axis=1).columns
```


```python
feature_import = best_clf.steps[2][1].feature_importances_
```


```python
dict(zip(feature,feature_import))
```




    {'Driven_Wheels_all wheel drive': 0.010180044520391722,
     'Driven_Wheels_front wheel drive': 0.0030040474201914486,
     'Driven_Wheels_rear wheel drive': 0.006939920130830467,
     'Engine Cylinders': 0.056492723143649226,
     'Engine Fuel Type_flex-fuel (premium unleaded required/E85)': 0.004622118920666116,
     'Engine Fuel Type_premium unleaded (required)': 0.02054458099422069,
     'Engine Fuel Type_regular unleaded': 0.009821706110858336,
     'Engine HP': 0.16784824407025872,
     'Engine HP^2': 0.19046619308817184,
     'Make_Aston Martin': 0.005356143350255724,
     'Make_Bentley': 0.012146657996343466,
     'Make_Bugatti': 0.0007619216530423492,
     'Make_Ferrari': 0.008570128324786444,
     'Make_Lamborghini': 0.004904571887663365,
     'Make_Maybach': 0.004338514012278021,
     'Make_Porsche': 0.008641787530394593,
     'Make_Rolls-Royce': 0.014741114807662128,
     'Model_458 Italia': 0.005405439308733585,
     'Model_57': 0.0018458020543991557,
     'Model_62': 0.005726411807380986,
     'Model_911': 0.0028952290716332068,
     'Model_Arnage': 0.0013392577032392242,
     'Model_Aventador': 0.0018176714628395,
     'Model_Carrera GT': 5.697533669976291e-06,
     'Model_Continental GT': 0.0004635586330599047,
     'Model_DBS': 0.010426194135688379,
     'Model_Enzo': 0.0009188950381704875,
     'Model_F430': 0.0006565191018580336,
     'Model_Gallardo': 0.0006422988939341604,
     'Model_Ghost': 0.0007490001611653036,
     'Model_Ghost Series II': 1.4323460133123379e-05,
     'Model_Landaulet': 0.006437036957277748,
     'Model_Murcielago': 0.0024864400711675848,
     'Model_Phantom': 0.0005154999674636062,
     'Model_Phantom Coupe': 1.0202661652350821e-06,
     'Model_Phantom Drophead Coupe': 6.784444922574109e-05,
     'Model_R8': 0.0014865141821076694,
     'Model_Reventon': 0.007777224543114406,
     'Model_SLR McLaren': 0.00503380396179973,
     'Model_Vanquish': 0.008172928102421984,
     'Model_Veyron 16.4': 0.0014414448847538625,
     'Number of Doors': 0.013405820394224634,
     'Transmission Type_AUTOMATED_MANUAL': 0.02306096094915099,
     'Transmission Type_MANUAL': 0.026773955122458785,
     'Vehicle Size_Large': 0.040271703766337305,
     'Vehicle Style_Convertible': 0.012202993884355669,
     'Vehicle Style_Coupe': 0.014985916031109263,
     'Year': 0.0644912715023895,
     'city mpg': 0.10877401504437749,
     'highway MPG': 0.1003268895925291}




```python
pd.DataFrame(feature_import, 
             index=feature).sort_values(0, 
                          ascending = False).plot.barh()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a160acda0>




![png](testcarprediction_files/testcarprediction_172_1.png)



```python
pd.DataFrame(feature_import, 
             index=feature).sort_values(0, 
                          ascending = False)
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Engine HP^2</th>
      <td>0.190466</td>
    </tr>
    <tr>
      <th>Engine HP</th>
      <td>0.167848</td>
    </tr>
    <tr>
      <th>city mpg</th>
      <td>0.108774</td>
    </tr>
    <tr>
      <th>highway MPG</th>
      <td>0.100327</td>
    </tr>
    <tr>
      <th>Year</th>
      <td>0.064491</td>
    </tr>
    <tr>
      <th>Engine Cylinders</th>
      <td>0.056493</td>
    </tr>
    <tr>
      <th>Vehicle Size_Large</th>
      <td>0.040272</td>
    </tr>
    <tr>
      <th>Transmission Type_MANUAL</th>
      <td>0.026774</td>
    </tr>
    <tr>
      <th>Transmission Type_AUTOMATED_MANUAL</th>
      <td>0.023061</td>
    </tr>
    <tr>
      <th>Engine Fuel Type_premium unleaded (required)</th>
      <td>0.020545</td>
    </tr>
    <tr>
      <th>Vehicle Style_Coupe</th>
      <td>0.014986</td>
    </tr>
    <tr>
      <th>Make_Rolls-Royce</th>
      <td>0.014741</td>
    </tr>
    <tr>
      <th>Number of Doors</th>
      <td>0.013406</td>
    </tr>
    <tr>
      <th>Vehicle Style_Convertible</th>
      <td>0.012203</td>
    </tr>
    <tr>
      <th>Make_Bentley</th>
      <td>0.012147</td>
    </tr>
    <tr>
      <th>Model_DBS</th>
      <td>0.010426</td>
    </tr>
    <tr>
      <th>Driven_Wheels_all wheel drive</th>
      <td>0.010180</td>
    </tr>
    <tr>
      <th>Engine Fuel Type_regular unleaded</th>
      <td>0.009822</td>
    </tr>
    <tr>
      <th>Make_Porsche</th>
      <td>0.008642</td>
    </tr>
    <tr>
      <th>Make_Ferrari</th>
      <td>0.008570</td>
    </tr>
    <tr>
      <th>Model_Vanquish</th>
      <td>0.008173</td>
    </tr>
    <tr>
      <th>Model_Reventon</th>
      <td>0.007777</td>
    </tr>
    <tr>
      <th>Driven_Wheels_rear wheel drive</th>
      <td>0.006940</td>
    </tr>
    <tr>
      <th>Model_Landaulet</th>
      <td>0.006437</td>
    </tr>
    <tr>
      <th>Model_62</th>
      <td>0.005726</td>
    </tr>
    <tr>
      <th>Model_458 Italia</th>
      <td>0.005405</td>
    </tr>
    <tr>
      <th>Make_Aston Martin</th>
      <td>0.005356</td>
    </tr>
    <tr>
      <th>Model_SLR McLaren</th>
      <td>0.005034</td>
    </tr>
    <tr>
      <th>Make_Lamborghini</th>
      <td>0.004905</td>
    </tr>
    <tr>
      <th>Engine Fuel Type_flex-fuel (premium unleaded required/E85)</th>
      <td>0.004622</td>
    </tr>
    <tr>
      <th>Make_Maybach</th>
      <td>0.004339</td>
    </tr>
    <tr>
      <th>Driven_Wheels_front wheel drive</th>
      <td>0.003004</td>
    </tr>
    <tr>
      <th>Model_911</th>
      <td>0.002895</td>
    </tr>
    <tr>
      <th>Model_Murcielago</th>
      <td>0.002486</td>
    </tr>
    <tr>
      <th>Model_57</th>
      <td>0.001846</td>
    </tr>
    <tr>
      <th>Model_Aventador</th>
      <td>0.001818</td>
    </tr>
    <tr>
      <th>Model_R8</th>
      <td>0.001487</td>
    </tr>
    <tr>
      <th>Model_Veyron 16.4</th>
      <td>0.001441</td>
    </tr>
    <tr>
      <th>Model_Arnage</th>
      <td>0.001339</td>
    </tr>
    <tr>
      <th>Model_Enzo</th>
      <td>0.000919</td>
    </tr>
    <tr>
      <th>Make_Bugatti</th>
      <td>0.000762</td>
    </tr>
    <tr>
      <th>Model_Ghost</th>
      <td>0.000749</td>
    </tr>
    <tr>
      <th>Model_F430</th>
      <td>0.000657</td>
    </tr>
    <tr>
      <th>Model_Gallardo</th>
      <td>0.000642</td>
    </tr>
    <tr>
      <th>Model_Phantom</th>
      <td>0.000515</td>
    </tr>
    <tr>
      <th>Model_Continental GT</th>
      <td>0.000464</td>
    </tr>
    <tr>
      <th>Model_Phantom Drophead Coupe</th>
      <td>0.000068</td>
    </tr>
    <tr>
      <th>Model_Ghost Series II</th>
      <td>0.000014</td>
    </tr>
    <tr>
      <th>Model_Carrera GT</th>
      <td>0.000006</td>
    </tr>
    <tr>
      <th>Model_Phantom Coupe</th>
      <td>0.000001</td>
    </tr>
  </tbody>
</table>
</div>




```python
gb_pred = g_search.predict(X_test)
plt.scatter(gb_pred, y_test)
plt.plot(y_test, y_test)
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()
```


![png](testcarprediction_files/testcarprediction_174_0.png)


<a id='ab-kn'></a>

# Pipeline with Kneighbors


```python
#initialize k-neighbors and selectKbest
selector = SelectKBest(f_regression)  # select k best
clf = KNeighborsRegressor()

#place SelectKbest transformer and RandomForest estimator into Pipeine
pipe = Pipeline(steps=[
    ('Scale',StandardScaler()),
   # ('poly', PolynomialFeatures()),
    ('selector', selector), 
    ('clf', clf)
])

#Create the parameter grid, entering the values to use for each parameter selected in the RandomForest estimator
parameters = {
    'selector__k':[50], # params to search through
    #'poly__degree': [2], 
    'clf__n_neighbors':[3,5,7], # number of neighbors to use 
    'clf__weights': ['uniform'], # weight function used in prediction 
    'clf__algorithm':['auto']}  # Algorithm used to compute the nearest neighbors 

#Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
g_search = GridSearchCV(pipe, parameters, cv=3, n_jobs=1, verbose=2)

#Fit the grid search object to the training data and find the optimal parameters using fit()
g_fit = g_search.fit(X_train, y_train)

#Get the best estimator and print out the estimator model
best_clf = g_fit.best_estimator_
print (best_clf)

#Use best estimator to make predictions on the test set
best_predictions = best_clf.predict(X_test)

#metrics
#print(mean_absolute_error(y_true = y_test, y_pred = best_predictions))
#print(r2_score(y_true = y_test, y_pred = best_predictions))

print("MAE: " + str(mean_absolute_error(y_true = y_test, y_pred = best_predictions)))
print("R2 Score: " + str(r2_score(y_true = y_test, y_pred = best_predictions)))

```

    Fitting 3 folds for each of 3 candidates, totalling 9 fits
    [CV] clf__algorithm=auto, clf__n_neighbors=3, clf__weights=uniform, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__algorithm=auto, clf__n_neighbors=3, clf__weights=uniform, selector__k=50, total=   1.5s
    [CV] clf__algorithm=auto, clf__n_neighbors=3, clf__weights=uniform, selector__k=50 


    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    3.8s remaining:    0.0s
    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__algorithm=auto, clf__n_neighbors=3, clf__weights=uniform, selector__k=50, total=   1.5s
    [CV] clf__algorithm=auto, clf__n_neighbors=3, clf__weights=uniform, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__algorithm=auto, clf__n_neighbors=3, clf__weights=uniform, selector__k=50, total=   1.6s
    [CV] clf__algorithm=auto, clf__n_neighbors=5, clf__weights=uniform, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__algorithm=auto, clf__n_neighbors=5, clf__weights=uniform, selector__k=50, total=   1.8s
    [CV] clf__algorithm=auto, clf__n_neighbors=5, clf__weights=uniform, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__algorithm=auto, clf__n_neighbors=5, clf__weights=uniform, selector__k=50, total=   1.9s
    [CV] clf__algorithm=auto, clf__n_neighbors=5, clf__weights=uniform, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__algorithm=auto, clf__n_neighbors=5, clf__weights=uniform, selector__k=50, total=   1.7s
    [CV] clf__algorithm=auto, clf__n_neighbors=7, clf__weights=uniform, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__algorithm=auto, clf__n_neighbors=7, clf__weights=uniform, selector__k=50, total=   1.7s
    [CV] clf__algorithm=auto, clf__n_neighbors=7, clf__weights=uniform, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__algorithm=auto, clf__n_neighbors=7, clf__weights=uniform, selector__k=50, total=   1.8s
    [CV] clf__algorithm=auto, clf__n_neighbors=7, clf__weights=uniform, selector__k=50 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__algorithm=auto, clf__n_neighbors=7, clf__weights=uniform, selector__k=50, total=   1.6s


    [Parallel(n_jobs=1)]: Done   9 out of   9 | elapsed:   38.2s finished
    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    Pipeline(memory=None,
         steps=[('Scale', StandardScaler(copy=True, with_mean=True, with_std=True)), ('selector', SelectKBest(k=50, score_func=<function f_regression at 0x109eeb950>)), ('clf', KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
              metric_params=None, n_jobs=1, n_neighbors=3, p=2,
              weights='uniform'))])
    MAE: 4840.068601190476
    R2 Score: 0.8896617420551712



```python
kn_pred = g_search.predict(X_test)
plt.scatter(kn_pred, y_test)
plt.plot(y_test, y_test)
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()
```


![png](testcarprediction_files/testcarprediction_177_0.png)


<a id='ab-visualizepred'></a>

# Visualize predictions 


```python
# plot y-pred vs y
# plot residulas vs y
# Residual histogram  (see if it looks normal)
# Y-hat vs y + y vs y
# Hyperparameter tuning curves
# Metrics vs model
```


```python
def lin_reg(x,y):
    # SLR, the correlation coefficient multiplied by the standard
    # deviation of y divided by standard deviation of x is the optimal slope.
    beta_1 = (scipy.stats.pearsonr(x,y)[0])*(np.std(y)/np.std(x))
    
    # The optimal beta is found by: mean(y) - b1 * mean(x).
    beta_0 = np.mean(y)-(beta_1*np.mean(x)) 
    
    return beta_0, beta_1
```


```python
x = df['Engine Cylinders'].values
y = df['MSRP'].values
beta0, beta1 = lin_reg(x,y)

#Print the optimal values.
print('The Optimal Y Intercept is ', beta0)
print('The Optimal slope is ', beta1)
```

    The Optimal Y Intercept is  -60081.168704419215
    The Optimal slope is  18051.885440336122



```python
y_pred = beta0 + beta1*x
```


```python
# Appending the predicted values:
df['Pred'] = y_pred

# Residuals equals the difference between Y-True and Y-Pred:
df['Residuals'] = abs(df['MSRP']-df['Pred'])
```


```python
# how our predictions compare to the true values.
sns.lmplot(x='MSRP', y='Pred', data=df)
```




    <seaborn.axisgrid.FacetGrid at 0x1a15302860>




![png](testcarprediction_files/testcarprediction_184_1.png)



```python
x = df['Engine HP'].values
y = df['MSRP'].values
beta0, beta1 = lin_reg(x,y)

#Print the optimal values.
print('The Optimal Y Intercept is ', beta0)
print('The Optimal slope is ', beta1)
```

    The Optimal Y Intercept is  -51242.12399479592
    The Optimal slope is  367.99756098271956



```python
# Appending the predicted values
df['Pred'] = y_pred

# Residuals equals the difference between Y-True and Y-Pred:
df['Residuals'] = abs(df['MSRP']-df['Pred'])
```


```python
sns.lmplot(x='MSRP', y='Pred', data=df)
```




    <seaborn.axisgrid.FacetGrid at 0x1a15320c88>




![png](testcarprediction_files/testcarprediction_187_1.png)



```python
# Assumptions for my model
plt.figure(figsize=(8,5))
df['Residuals'] = df['MSRP'] - df['Pred']
sns.distplot(df['Residuals'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a15630828>




![png](testcarprediction_files/testcarprediction_188_1.png)


<a id='ab-bestmodel'></a>

# Best Model for less features


```python
#initialize gradient boosting and selectKbest
selector = SelectKBest(f_regression)  # select k best
clf = GradientBoostingRegressor() # Model I want to use 

#place SelectKbest transformer and RandomForest estimator into Pipeine
pipe = Pipeline(steps=[
    ('Scale',StandardScaler()),
    ('selector', selector), 
    ('clf', clf)
])

#Create the parameter grid, entering the values to use for each parameter selected in the RandomForest estimator
parameters = {
    'selector__k':[12], # params to search through
    'clf__n_estimators': [20, 100], # num of boosting stages to perform. larger number usually better performance
    'clf__learning_rate':[0.05, 0.1, 0.2], # shrinks the contibution of each tree. trade off between n_estimators and learning rate
    'clf__max_depth': [1, 3, 5] # max depth of the individual regression estimators
}

#Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
g_search = GridSearchCV(pipe, parameters, cv=3, n_jobs=1, verbose=2)

#Fit the grid search object to the training data and find the optimal parameters using fit()
g_fit = g_search.fit(X_train, y_train)

#Get the best estimator and print out the estimator model
best_clf = g_fit.best_estimator_
print (best_clf)

#Use best estimator to make predictions on the test set
best_predictions = best_clf.predict(X_test)


#metrics
#print(mean_absolute_error(y_true = y_test, y_pred = best_predictions))
#print(r2_score(y_true = y_test, y_pred = best_predictions))

print("MAE: " + str(mean_absolute_error(y_true = y_test, y_pred = best_predictions)))
print("R2 Score: " + str(r2_score(y_true = y_test, y_pred = best_predictions)))
```

    Fitting 3 folds for each of 18 candidates, totalling 54 fits
    [CV] clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)
    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.5s remaining:    0.0s


    [CV]  clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=20, selector__k=12, total=   0.4s
    [CV] clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=20, selector__k=12, total=   0.4s
    [CV] clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=20, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=100, selector__k=12, total=   0.4s
    [CV] clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=100, selector__k=12, total=   0.4s
    [CV] clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=1, clf__n_estimators=100, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=20, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=20, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=20, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=100, selector__k=12, total=   0.5s
    [CV] clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=100, selector__k=12, total=   0.5s
    [CV] clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=3, clf__n_estimators=100, selector__k=12, total=   0.5s
    [CV] clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=20, selector__k=12, total=   0.4s
    [CV] clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=20, selector__k=12, total=   0.4s
    [CV] clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=20, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=100, selector__k=12, total=   0.6s
    [CV] clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=100, selector__k=12, total=   0.7s
    [CV] clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.05, clf__max_depth=5, clf__n_estimators=100, selector__k=12, total=   0.7s
    [CV] clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=20, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=20, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=20, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=100, selector__k=12, total=   0.4s
    [CV] clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=100, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=1, clf__n_estimators=100, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=20, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=20, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=20, selector__k=12, total=   0.4s
    [CV] clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=100, selector__k=12, total=   0.5s
    [CV] clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=100, selector__k=12, total=   0.5s
    [CV] clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=3, clf__n_estimators=100, selector__k=12, total=   0.5s
    [CV] clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=20, selector__k=12, total=   0.4s
    [CV] clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=20, selector__k=12, total=   0.4s
    [CV] clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=20, selector__k=12, total=   0.4s
    [CV] clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=100, selector__k=12, total=   0.6s
    [CV] clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=100, selector__k=12, total=   0.6s
    [CV] clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.1, clf__max_depth=5, clf__n_estimators=100, selector__k=12, total=   0.6s
    [CV] clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=20, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=20, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=20, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=100, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=100, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=1, clf__n_estimators=100, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=20, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=20, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=20, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=100, selector__k=12, total=   0.5s
    [CV] clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=100, selector__k=12, total=   0.4s
    [CV] clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=3, clf__n_estimators=100, selector__k=12, total=   0.4s
    [CV] clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=20, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=20, selector__k=12, total=   0.4s
    [CV] clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=20, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=20, selector__k=12, total=   0.3s
    [CV] clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=100, selector__k=12, total=   0.9s
    [CV] clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=100, selector__k=12, total=   0.8s
    [CV] clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=100, selector__k=12 


    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    [CV]  clf__learning_rate=0.2, clf__max_depth=5, clf__n_estimators=100, selector__k=12, total=   0.7s


    [Parallel(n_jobs=1)]: Done  54 out of  54 | elapsed:   28.0s finished
    /anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/univariate_selection.py:298: RuntimeWarning: invalid value encountered in true_divide
      corr /= X_norms
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /anaconda3/lib/python3.6/site-packages/scipy/stats/_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)


    Pipeline(memory=None,
         steps=[('Scale', StandardScaler(copy=True, with_mean=True, with_std=True)), ('selector', SelectKBest(k=12, score_func=<function f_regression at 0x109eeb950>)), ('clf', GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.2, loss='ls', max_depth=5, ma...s=100, presort='auto', random_state=None,
                 subsample=1.0, verbose=0, warm_start=False))])
    MAE: 7224.197106401388
    R2 Score: 0.944446769451272



```python
best_clf.steps[1][1].get_support()
```




    array([False,  True,  True, ..., False, False, False])




```python
feature = list(X_cols[best_clf.steps[1][1].get_support()])
```


```python
feature_import = best_clf.steps[2][1].feature_importances_
```


```python
dict(zip(feature,feature_import))
```




    {'Engine Cylinders': 0.161652154815887,
     'Engine Fuel Type_premium unleaded (required)': 0.03686347996660804,
     'Engine Fuel Type_regular unleaded': 0.022067087420454934,
     'Engine HP': 0.3288154443057195,
     'Engine HP^2': 0.39578180065792545,
     'Make_Bentley': 0.011090018954312117,
     'Make_Bugatti': 0.0011565166088123209,
     'Make_Lamborghini': 0.016016273040587527,
     'Make_Maybach': 0.007658665277021095,
     'Make_Rolls-Royce': 0.009172297191849426,
     'Model_Landaulet': 0.008039050208676577,
     'Model_Veyron 16.4': 0.0016872115521459321}




```python
pd.DataFrame(feature_import, 
             index=feature).sort_values(0, 
                          ascending = False).plot.barh()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a118de978>




![png](testcarprediction_files/testcarprediction_195_1.png)



```python
pd.DataFrame(feature_import, 
             index=feature).sort_values(0, 
                          ascending = False)
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Engine HP^2</th>
      <td>0.395782</td>
    </tr>
    <tr>
      <th>Engine HP</th>
      <td>0.328815</td>
    </tr>
    <tr>
      <th>Engine Cylinders</th>
      <td>0.161652</td>
    </tr>
    <tr>
      <th>Engine Fuel Type_premium unleaded (required)</th>
      <td>0.036863</td>
    </tr>
    <tr>
      <th>Engine Fuel Type_regular unleaded</th>
      <td>0.022067</td>
    </tr>
    <tr>
      <th>Make_Lamborghini</th>
      <td>0.016016</td>
    </tr>
    <tr>
      <th>Make_Bentley</th>
      <td>0.011090</td>
    </tr>
    <tr>
      <th>Make_Rolls-Royce</th>
      <td>0.009172</td>
    </tr>
    <tr>
      <th>Model_Landaulet</th>
      <td>0.008039</td>
    </tr>
    <tr>
      <th>Make_Maybach</th>
      <td>0.007659</td>
    </tr>
    <tr>
      <th>Model_Veyron 16.4</th>
      <td>0.001687</td>
    </tr>
    <tr>
      <th>Make_Bugatti</th>
      <td>0.001157</td>
    </tr>
  </tbody>
</table>
</div>




```python
import statsmodels.api as sm
```


```python
model = sm.OLS(y,X).fit()
```


```python
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th>  <td>   0.988</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.986</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   847.9</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 17 Jul 2018</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>00:34:14</td>     <th>  Log-Likelihood:    </th> <td>-1.1478e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 11199</td>      <th>  AIC:               </th>  <td>2.315e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 10233</td>      <th>  BIC:               </th>  <td>2.386e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>   965</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
                                <td></td>                                   <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Year</th>                                                          <td>  744.0213</td> <td>   81.465</td> <td>    9.133</td> <td> 0.000</td> <td>  584.334</td> <td>  903.709</td>
</tr>
<tr>
  <th>Engine HP</th>                                                     <td> -127.7246</td> <td>    8.198</td> <td>  -15.579</td> <td> 0.000</td> <td> -143.795</td> <td> -111.654</td>
</tr>
<tr>
  <th>Engine Cylinders</th>                                              <td> 1195.3531</td> <td>  183.652</td> <td>    6.509</td> <td> 0.000</td> <td>  835.359</td> <td> 1555.347</td>
</tr>
<tr>
  <th>Number of Doors</th>                                               <td>  943.7598</td> <td>  508.433</td> <td>    1.856</td> <td> 0.063</td> <td>  -52.869</td> <td> 1940.389</td>
</tr>
<tr>
  <th>highway MPG</th>                                                   <td>   -5.4721</td> <td>   22.103</td> <td>   -0.248</td> <td> 0.804</td> <td>  -48.798</td> <td>   37.854</td>
</tr>
<tr>
  <th>city mpg</th>                                                      <td>  -61.3339</td> <td>   63.666</td> <td>   -0.963</td> <td> 0.335</td> <td> -186.131</td> <td>   63.463</td>
</tr>
<tr>
  <th>Popularity</th>                                                    <td>  -39.2920</td> <td>    2.871</td> <td>  -13.686</td> <td> 0.000</td> <td>  -44.920</td> <td>  -33.664</td>
</tr>
<tr>
  <th>Engine HP^2</th>                                                   <td>    0.3414</td> <td>    0.010</td> <td>   34.979</td> <td> 0.000</td> <td>    0.322</td> <td>    0.361</td>
</tr>
<tr>
  <th>Make_Acura</th>                                                    <td>-1.135e+05</td> <td> 5682.850</td> <td>  -19.980</td> <td> 0.000</td> <td>-1.25e+05</td> <td>-1.02e+05</td>
</tr>
<tr>
  <th>Make_Alfa Romeo</th>                                               <td>-5.201e+04</td> <td> 3788.710</td> <td>  -13.727</td> <td> 0.000</td> <td>-5.94e+04</td> <td>-4.46e+04</td>
</tr>
<tr>
  <th>Make_Aston Martin</th>                                             <td> 1.192e+04</td> <td> 5828.697</td> <td>    2.045</td> <td> 0.041</td> <td>  492.905</td> <td> 2.33e+04</td>
</tr>
<tr>
  <th>Make_Audi</th>                                                     <td>-1.102e+04</td> <td> 4096.029</td> <td>   -2.691</td> <td> 0.007</td> <td>-1.91e+04</td> <td>-2994.038</td>
</tr>
<tr>
  <th>Make_BMW</th>                                                      <td> 3.836e+04</td> <td> 5130.895</td> <td>    7.476</td> <td> 0.000</td> <td> 2.83e+04</td> <td> 4.84e+04</td>
</tr>
<tr>
  <th>Make_Bentley</th>                                                  <td> 1.088e+05</td> <td> 5137.073</td> <td>   21.184</td> <td> 0.000</td> <td> 9.88e+04</td> <td> 1.19e+05</td>
</tr>
<tr>
  <th>Make_Bugatti</th>                                                  <td> 6.922e+05</td> <td> 3637.595</td> <td>  190.291</td> <td> 0.000</td> <td> 6.85e+05</td> <td> 6.99e+05</td>
</tr>
<tr>
  <th>Make_Buick</th>                                                    <td>-1.244e+05</td> <td> 5587.367</td> <td>  -22.272</td> <td> 0.000</td> <td>-1.35e+05</td> <td>-1.13e+05</td>
</tr>
<tr>
  <th>Make_Cadillac</th>                                                 <td>-5.911e+04</td> <td> 2011.354</td> <td>  -29.391</td> <td> 0.000</td> <td>-6.31e+04</td> <td>-5.52e+04</td>
</tr>
<tr>
  <th>Make_Chevrolet</th>                                                <td>-7.326e+04</td> <td> 3239.041</td> <td>  -22.618</td> <td> 0.000</td> <td>-7.96e+04</td> <td>-6.69e+04</td>
</tr>
<tr>
  <th>Make_Chrysler</th>                                                 <td>-8.716e+04</td> <td> 5579.228</td> <td>  -15.622</td> <td> 0.000</td> <td>-9.81e+04</td> <td>-7.62e+04</td>
</tr>
<tr>
  <th>Make_Dodge</th>                                                    <td>-6.135e+04</td> <td> 4591.941</td> <td>  -13.360</td> <td> 0.000</td> <td>-7.03e+04</td> <td>-5.23e+04</td>
</tr>
<tr>
  <th>Make_FIAT</th>                                                     <td> -9.24e+04</td> <td> 4262.418</td> <td>  -21.678</td> <td> 0.000</td> <td>-1.01e+05</td> <td> -8.4e+04</td>
</tr>
<tr>
  <th>Make_Ferrari</th>                                                  <td> 1.577e+05</td> <td> 1845.857</td> <td>   85.440</td> <td> 0.000</td> <td> 1.54e+05</td> <td> 1.61e+05</td>
</tr>
<tr>
  <th>Make_Ford</th>                                                     <td> 8.029e+04</td> <td> 9493.408</td> <td>    8.457</td> <td> 0.000</td> <td> 6.17e+04</td> <td> 9.89e+04</td>
</tr>
<tr>
  <th>Make_GMC</th>                                                      <td>-1.397e+05</td> <td> 4730.569</td> <td>  -29.537</td> <td> 0.000</td> <td>-1.49e+05</td> <td> -1.3e+05</td>
</tr>
<tr>
  <th>Make_Genesis</th>                                                  <td>-6.559e+04</td> <td> 4125.760</td> <td>  -15.898</td> <td> 0.000</td> <td>-7.37e+04</td> <td>-5.75e+04</td>
</tr>
<tr>
  <th>Make_HUMMER</th>                                                   <td>-8.554e+04</td> <td> 4542.425</td> <td>  -18.832</td> <td> 0.000</td> <td>-9.44e+04</td> <td>-7.66e+04</td>
</tr>
<tr>
  <th>Make_Honda</th>                                                    <td>-4.774e+04</td> <td>  847.841</td> <td>  -56.307</td> <td> 0.000</td> <td>-4.94e+04</td> <td>-4.61e+04</td>
</tr>
<tr>
  <th>Make_Hyundai</th>                                                  <td>-7.791e+04</td> <td> 2552.363</td> <td>  -30.526</td> <td> 0.000</td> <td>-8.29e+04</td> <td>-7.29e+04</td>
</tr>
<tr>
  <th>Make_Infiniti</th>                                                 <td>-1.151e+05</td> <td> 6802.790</td> <td>  -16.924</td> <td> 0.000</td> <td>-1.28e+05</td> <td>-1.02e+05</td>
</tr>
<tr>
  <th>Make_Kia</th>                                                      <td>-6.594e+04</td> <td> 1928.668</td> <td>  -34.190</td> <td> 0.000</td> <td>-6.97e+04</td> <td>-6.22e+04</td>
</tr>
<tr>
  <th>Make_Lamborghini</th>                                              <td> 2.758e+05</td> <td> 3477.301</td> <td>   79.324</td> <td> 0.000</td> <td> 2.69e+05</td> <td> 2.83e+05</td>
</tr>
<tr>
  <th>Make_Land Rover</th>                                               <td> -1.01e+05</td> <td> 5289.941</td> <td>  -19.098</td> <td> 0.000</td> <td>-1.11e+05</td> <td>-9.07e+04</td>
</tr>
<tr>
  <th>Make_Lexus</th>                                                    <td>-9.733e+04</td> <td> 5290.597</td> <td>  -18.396</td> <td> 0.000</td> <td>-1.08e+05</td> <td> -8.7e+04</td>
</tr>
<tr>
  <th>Make_Lincoln</th>                                                  <td>-1.662e+05</td> <td> 6841.451</td> <td>  -24.292</td> <td> 0.000</td> <td> -1.8e+05</td> <td>-1.53e+05</td>
</tr>
<tr>
  <th>Make_Lotus</th>                                                    <td>-5.633e+04</td> <td> 4512.174</td> <td>  -12.484</td> <td> 0.000</td> <td>-6.52e+04</td> <td>-4.75e+04</td>
</tr>
<tr>
  <th>Make_Maserati</th>                                                 <td>-6.931e+04</td> <td> 8053.531</td> <td>   -8.606</td> <td> 0.000</td> <td>-8.51e+04</td> <td>-5.35e+04</td>
</tr>
<tr>
  <th>Make_Maybach</th>                                                  <td> 3.867e+05</td> <td> 5252.038</td> <td>   73.632</td> <td> 0.000</td> <td> 3.76e+05</td> <td> 3.97e+05</td>
</tr>
<tr>
  <th>Make_Mazda</th>                                                    <td>-1.081e+05</td> <td> 5126.187</td> <td>  -21.080</td> <td> 0.000</td> <td>-1.18e+05</td> <td> -9.8e+04</td>
</tr>
<tr>
  <th>Make_McLaren</th>                                                  <td> 1.821e+04</td> <td> 5448.645</td> <td>    3.343</td> <td> 0.001</td> <td> 7534.517</td> <td> 2.89e+04</td>
</tr>
<tr>
  <th>Make_Mercedes-Benz</th>                                            <td>-7.765e+04</td> <td> 4660.746</td> <td>  -16.660</td> <td> 0.000</td> <td>-8.68e+04</td> <td>-6.85e+04</td>
</tr>
<tr>
  <th>Make_Mitsubishi</th>                                               <td>-1.187e+05</td> <td> 5027.187</td> <td>  -23.609</td> <td> 0.000</td> <td>-1.29e+05</td> <td>-1.09e+05</td>
</tr>
<tr>
  <th>Make_Nissan</th>                                                   <td>-6.219e+04</td> <td> 1327.931</td> <td>  -46.830</td> <td> 0.000</td> <td>-6.48e+04</td> <td>-5.96e+04</td>
</tr>
<tr>
  <th>Make_Oldsmobile</th>                                               <td>-1.345e+05</td> <td> 5311.273</td> <td>  -25.325</td> <td> 0.000</td> <td>-1.45e+05</td> <td>-1.24e+05</td>
</tr>
<tr>
  <th>Make_Plymouth</th>                                                 <td>-1.176e+05</td> <td> 6913.364</td> <td>  -17.006</td> <td> 0.000</td> <td>-1.31e+05</td> <td>-1.04e+05</td>
</tr>
<tr>
  <th>Make_Pontiac</th>                                                  <td>-1.242e+05</td> <td> 5275.567</td> <td>  -23.541</td> <td> 0.000</td> <td>-1.35e+05</td> <td>-1.14e+05</td>
</tr>
<tr>
  <th>Make_Porsche</th>                                                  <td> -1.65e+04</td> <td> 1861.879</td> <td>   -8.859</td> <td> 0.000</td> <td>-2.01e+04</td> <td>-1.28e+04</td>
</tr>
<tr>
  <th>Make_Rolls-Royce</th>                                              <td> 1.306e+05</td> <td> 6175.348</td> <td>   21.146</td> <td> 0.000</td> <td> 1.18e+05</td> <td> 1.43e+05</td>
</tr>
<tr>
  <th>Make_Saab</th>                                                     <td>-1.044e+05</td> <td> 4746.725</td> <td>  -22.004</td> <td> 0.000</td> <td>-1.14e+05</td> <td>-9.51e+04</td>
</tr>
<tr>
  <th>Make_Scion</th>                                                    <td>-1.238e+05</td> <td> 6088.127</td> <td>  -20.334</td> <td> 0.000</td> <td>-1.36e+05</td> <td>-1.12e+05</td>
</tr>
<tr>
  <th>Make_Spyker</th>                                                   <td> 1.723e+04</td> <td> 4178.446</td> <td>    4.123</td> <td> 0.000</td> <td> 9038.610</td> <td> 2.54e+04</td>
</tr>
<tr>
  <th>Make_Subaru</th>                                                   <td>-1.084e+05</td> <td> 4497.695</td> <td>  -24.107</td> <td> 0.000</td> <td>-1.17e+05</td> <td>-9.96e+04</td>
</tr>
<tr>
  <th>Make_Suzuki</th>                                                   <td>-1.162e+05</td> <td> 4660.367</td> <td>  -24.925</td> <td> 0.000</td> <td>-1.25e+05</td> <td>-1.07e+05</td>
</tr>
<tr>
  <th>Make_Tesla</th>                                                    <td>-2.708e+04</td> <td> 3844.997</td> <td>   -7.043</td> <td> 0.000</td> <td>-3.46e+04</td> <td>-1.95e+04</td>
</tr>
<tr>
  <th>Make_Toyota</th>                                                   <td> -5.62e+04</td> <td> 1048.016</td> <td>  -53.625</td> <td> 0.000</td> <td>-5.83e+04</td> <td>-5.41e+04</td>
</tr>
<tr>
  <th>Make_Volkswagen</th>                                               <td> -9.44e+04</td> <td> 5930.929</td> <td>  -15.916</td> <td> 0.000</td> <td>-1.06e+05</td> <td>-8.28e+04</td>
</tr>
<tr>
  <th>Make_Volvo</th>                                                    <td>-9.449e+04</td> <td> 8082.457</td> <td>  -11.691</td> <td> 0.000</td> <td> -1.1e+05</td> <td>-7.86e+04</td>
</tr>
<tr>
  <th>Model_1 Series</th>                                                <td>-1.404e+04</td> <td> 2760.376</td> <td>   -5.087</td> <td> 0.000</td> <td>-1.95e+04</td> <td>-8630.641</td>
</tr>
<tr>
  <th>Model_1 Series M</th>                                              <td>-5595.7536</td> <td> 7351.070</td> <td>   -0.761</td> <td> 0.447</td> <td>   -2e+04</td> <td> 8813.783</td>
</tr>
<tr>
  <th>Model_100</th>                                                     <td>-1.348e+04</td> <td> 3384.461</td> <td>   -3.982</td> <td> 0.000</td> <td>-2.01e+04</td> <td>-6841.128</td>
</tr>
<tr>
  <th>Model_124 Spider</th>                                              <td> -1.86e+04</td> <td> 3795.760</td> <td>   -4.900</td> <td> 0.000</td> <td> -2.6e+04</td> <td>-1.12e+04</td>
</tr>
<tr>
  <th>Model_190-Class</th>                                               <td>-4.065e+04</td> <td> 3207.910</td> <td>  -12.671</td> <td> 0.000</td> <td>-4.69e+04</td> <td>-3.44e+04</td>
</tr>
<tr>
  <th>Model_2</th>                                                       <td> -1.63e+04</td> <td> 2599.390</td> <td>   -6.270</td> <td> 0.000</td> <td>-2.14e+04</td> <td>-1.12e+04</td>
</tr>
<tr>
  <th>Model_2 Series</th>                                                <td>-1.557e+04</td> <td> 2776.120</td> <td>   -5.608</td> <td> 0.000</td> <td> -2.1e+04</td> <td>-1.01e+04</td>
</tr>
<tr>
  <th>Model_200</th>                                                     <td>-1.051e+04</td> <td> 3035.543</td> <td>   -3.461</td> <td> 0.001</td> <td>-1.65e+04</td> <td>-4556.092</td>
</tr>
<tr>
  <th>Model_200SX</th>                                                   <td>-4814.4195</td> <td> 3235.365</td> <td>   -1.488</td> <td> 0.137</td> <td>-1.12e+04</td> <td> 1527.529</td>
</tr>
<tr>
  <th>Model_240</th>                                                     <td> -1.38e+04</td> <td> 7953.630</td> <td>   -1.735</td> <td> 0.083</td> <td>-2.94e+04</td> <td> 1787.295</td>
</tr>
<tr>
  <th>Model_240SX</th>                                                   <td>-1407.3579</td> <td> 2911.295</td> <td>   -0.483</td> <td> 0.629</td> <td>-7114.066</td> <td> 4299.350</td>
</tr>
<tr>
  <th>Model_3</th>                                                       <td>-8919.4721</td> <td> 1833.143</td> <td>   -4.866</td> <td> 0.000</td> <td>-1.25e+04</td> <td>-5326.154</td>
</tr>
<tr>
  <th>Model_3 Series</th>                                                <td>-1.019e+04</td> <td> 2573.842</td> <td>   -3.960</td> <td> 0.000</td> <td>-1.52e+04</td> <td>-5148.171</td>
</tr>
<tr>
  <th>Model_3 Series Gran Turismo</th>                                   <td>-9297.1967</td> <td> 3712.991</td> <td>   -2.504</td> <td> 0.012</td> <td>-1.66e+04</td> <td>-2019.007</td>
</tr>
<tr>
  <th>Model_300</th>                                                     <td>-6248.6887</td> <td> 3460.540</td> <td>   -1.806</td> <td> 0.071</td> <td> -1.3e+04</td> <td>  534.647</td>
</tr>
<tr>
  <th>Model_300-Class</th>                                               <td>-4.478e+04</td> <td> 1898.369</td> <td>  -23.591</td> <td> 0.000</td> <td>-4.85e+04</td> <td>-4.11e+04</td>
</tr>
<tr>
  <th>Model_3000GT</th>                                                  <td>-1.399e+04</td> <td> 2581.584</td> <td>   -5.420</td> <td> 0.000</td> <td>-1.91e+04</td> <td>-8931.928</td>
</tr>
<tr>
  <th>Model_300M</th>                                                    <td>-1652.7505</td> <td> 4550.296</td> <td>   -0.363</td> <td> 0.716</td> <td>-1.06e+04</td> <td> 7266.721</td>
</tr>
<tr>
  <th>Model_300ZX</th>                                                   <td>-6729.5740</td> <td> 2575.053</td> <td>   -2.613</td> <td> 0.009</td> <td>-1.18e+04</td> <td>-1681.967</td>
</tr>
<tr>
  <th>Model_323</th>                                                     <td>-1.602e+04</td> <td> 4084.128</td> <td>   -3.923</td> <td> 0.000</td> <td> -2.4e+04</td> <td>-8014.785</td>
</tr>
<tr>
  <th>Model_350-Class</th>                                               <td> -4.81e+04</td> <td> 3900.158</td> <td>  -12.334</td> <td> 0.000</td> <td>-5.57e+04</td> <td>-4.05e+04</td>
</tr>
<tr>
  <th>Model_350Z</th>                                                    <td> 1.104e+04</td> <td> 2000.519</td> <td>    5.517</td> <td> 0.000</td> <td> 7115.830</td> <td>  1.5e+04</td>
</tr>
<tr>
  <th>Model_360</th>                                                     <td>-6.618e+04</td> <td> 2283.939</td> <td>  -28.974</td> <td> 0.000</td> <td>-7.07e+04</td> <td>-6.17e+04</td>
</tr>
<tr>
  <th>Model_370Z</th>                                                    <td> 7372.2050</td> <td> 2173.103</td> <td>    3.392</td> <td> 0.001</td> <td> 3112.498</td> <td> 1.16e+04</td>
</tr>
<tr>
  <th>Model_4 Series</th>                                                <td>-6421.2427</td> <td> 2619.523</td> <td>   -2.451</td> <td> 0.014</td> <td>-1.16e+04</td> <td>-1286.464</td>
</tr>
<tr>
  <th>Model_4 Series Gran Coupe</th>                                     <td>-7047.1577</td> <td> 3021.806</td> <td>   -2.332</td> <td> 0.020</td> <td> -1.3e+04</td> <td>-1123.827</td>
</tr>
<tr>
  <th>Model_400-Class</th>                                               <td>-4.859e+04</td> <td> 3757.763</td> <td>  -12.931</td> <td> 0.000</td> <td> -5.6e+04</td> <td>-4.12e+04</td>
</tr>
<tr>
  <th>Model_420-Class</th>                                               <td>-4.581e+04</td> <td> 5206.282</td> <td>   -8.799</td> <td> 0.000</td> <td> -5.6e+04</td> <td>-3.56e+04</td>
</tr>
<tr>
  <th>Model_456M</th>                                                    <td>-7616.5847</td> <td> 3057.726</td> <td>   -2.491</td> <td> 0.013</td> <td>-1.36e+04</td> <td>-1622.842</td>
</tr>
<tr>
  <th>Model_458 Italia</th>                                              <td>-1.459e+04</td> <td> 2677.492</td> <td>   -5.450</td> <td> 0.000</td> <td>-1.98e+04</td> <td>-9344.111</td>
</tr>
<tr>
  <th>Model_4C</th>                                                      <td>-5.201e+04</td> <td> 3788.710</td> <td>  -13.727</td> <td> 0.000</td> <td>-5.94e+04</td> <td>-4.46e+04</td>
</tr>
<tr>
  <th>Model_4Runner</th>                                                 <td> 4077.9483</td> <td> 1541.771</td> <td>    2.645</td> <td> 0.008</td> <td> 1055.774</td> <td> 7100.122</td>
</tr>
<tr>
  <th>Model_5</th>                                                       <td>-7275.5055</td> <td> 4330.227</td> <td>   -1.680</td> <td> 0.093</td> <td>-1.58e+04</td> <td> 1212.587</td>
</tr>
<tr>
  <th>Model_5 Series</th>                                                <td>-1965.2143</td> <td> 3009.560</td> <td>   -0.653</td> <td> 0.514</td> <td>-7864.541</td> <td> 3934.112</td>
</tr>
<tr>
  <th>Model_5 Series Gran Turismo</th>                                   <td>-1275.4174</td> <td> 3366.882</td> <td>   -0.379</td> <td> 0.705</td> <td>-7875.166</td> <td> 5324.331</td>
</tr>
<tr>
  <th>Model_500</th>                                                     <td> -2.28e+04</td> <td> 2000.608</td> <td>  -11.397</td> <td> 0.000</td> <td>-2.67e+04</td> <td>-1.89e+04</td>
</tr>
<tr>
  <th>Model_500-Class</th>                                               <td> -5.35e+04</td> <td> 2812.166</td> <td>  -19.025</td> <td> 0.000</td> <td> -5.9e+04</td> <td> -4.8e+04</td>
</tr>
<tr>
  <th>Model_500L</th>                                                    <td>-1.676e+04</td> <td> 2385.210</td> <td>   -7.028</td> <td> 0.000</td> <td>-2.14e+04</td> <td>-1.21e+04</td>
</tr>
<tr>
  <th>Model_500X</th>                                                    <td>-1.734e+04</td> <td> 2171.919</td> <td>   -7.985</td> <td> 0.000</td> <td>-2.16e+04</td> <td>-1.31e+04</td>
</tr>
<tr>
  <th>Model_500e</th>                                                    <td>-1.689e+04</td> <td> 5274.069</td> <td>   -3.203</td> <td> 0.001</td> <td>-2.72e+04</td> <td>-6553.978</td>
</tr>
<tr>
  <th>Model_550</th>                                                     <td>-1.615e+04</td> <td> 4906.165</td> <td>   -3.291</td> <td> 0.001</td> <td>-2.58e+04</td> <td>-6530.178</td>
</tr>
<tr>
  <th>Model_560-Class</th>                                               <td>-4.649e+04</td> <td> 3859.938</td> <td>  -12.045</td> <td> 0.000</td> <td>-5.41e+04</td> <td>-3.89e+04</td>
</tr>
<tr>
  <th>Model_57</th>                                                      <td>-2.083e+05</td> <td> 3008.203</td> <td>  -69.252</td> <td> 0.000</td> <td>-2.14e+05</td> <td>-2.02e+05</td>
</tr>
<tr>
  <th>Model_570S</th>                                                    <td> -3.67e+04</td> <td> 6292.221</td> <td>   -5.833</td> <td> 0.000</td> <td> -4.9e+04</td> <td>-2.44e+04</td>
</tr>
<tr>
  <th>Model_575M</th>                                                    <td>-2.795e+04</td> <td> 2998.765</td> <td>   -9.322</td> <td> 0.000</td> <td>-3.38e+04</td> <td>-2.21e+04</td>
</tr>
<tr>
  <th>Model_599</th>                                                     <td>  5.16e+04</td> <td> 3248.072</td> <td>   15.887</td> <td> 0.000</td> <td> 4.52e+04</td> <td>  5.8e+04</td>
</tr>
<tr>
  <th>Model_6</th>                                                       <td>-5439.0689</td> <td> 2423.321</td> <td>   -2.244</td> <td> 0.025</td> <td>-1.02e+04</td> <td> -688.885</td>
</tr>
<tr>
  <th>Model_6 Series</th>                                                <td> 1.794e+04</td> <td> 2638.790</td> <td>    6.797</td> <td> 0.000</td> <td> 1.28e+04</td> <td> 2.31e+04</td>
</tr>
<tr>
  <th>Model_6 Series Gran Coupe</th>                                     <td> 1.838e+04</td> <td> 3184.342</td> <td>    5.773</td> <td> 0.000</td> <td> 1.21e+04</td> <td> 2.46e+04</td>
</tr>
<tr>
  <th>Model_600-Class</th>                                               <td>-6.699e+04</td> <td> 3796.974</td> <td>  -17.643</td> <td> 0.000</td> <td>-7.44e+04</td> <td>-5.95e+04</td>
</tr>
<tr>
  <th>Model_6000</th>                                                    <td> -1.23e+04</td> <td> 2853.952</td> <td>   -4.311</td> <td> 0.000</td> <td>-1.79e+04</td> <td>-6708.436</td>
</tr>
<tr>
  <th>Model_612 Scaglietti</th>                                          <td> 4.971e+04</td> <td> 4112.969</td> <td>   12.086</td> <td> 0.000</td> <td> 4.16e+04</td> <td> 5.78e+04</td>
</tr>
<tr>
  <th>Model_62</th>                                                      <td>-1.572e+05</td> <td> 3008.203</td> <td>  -52.260</td> <td> 0.000</td> <td>-1.63e+05</td> <td>-1.51e+05</td>
</tr>
<tr>
  <th>Model_626</th>                                                     <td>-7121.8836</td> <td> 2688.252</td> <td>   -2.649</td> <td> 0.008</td> <td>-1.24e+04</td> <td>-1852.383</td>
</tr>
<tr>
  <th>Model_650S Coupe</th>                                              <td> 2.228e+04</td> <td> 6277.261</td> <td>    3.549</td> <td> 0.000</td> <td> 9975.894</td> <td> 3.46e+04</td>
</tr>
<tr>
  <th>Model_650S Spider</th>                                             <td> 2.891e+04</td> <td> 6284.110</td> <td>    4.600</td> <td> 0.000</td> <td> 1.66e+04</td> <td> 4.12e+04</td>
</tr>
<tr>
  <th>Model_7 Series</th>                                                <td> 2.032e+04</td> <td> 3011.819</td> <td>    6.748</td> <td> 0.000</td> <td> 1.44e+04</td> <td> 2.62e+04</td>
</tr>
<tr>
  <th>Model_718 Cayman</th>                                              <td>-2.447e+04</td> <td> 4865.942</td> <td>   -5.029</td> <td> 0.000</td> <td> -3.4e+04</td> <td>-1.49e+04</td>
</tr>
<tr>
  <th>Model_740</th>                                                     <td>-1.295e+04</td> <td> 7733.650</td> <td>   -1.675</td> <td> 0.094</td> <td>-2.81e+04</td> <td> 2208.655</td>
</tr>
<tr>
  <th>Model_760</th>                                                     <td>-1.321e+04</td> <td> 8530.321</td> <td>   -1.549</td> <td> 0.121</td> <td>-2.99e+04</td> <td> 3506.309</td>
</tr>
<tr>
  <th>Model_780</th>                                                     <td>-1.327e+04</td> <td> 9002.131</td> <td>   -1.474</td> <td> 0.141</td> <td>-3.09e+04</td> <td> 4378.901</td>
</tr>
<tr>
  <th>Model_8 Series</th>                                                <td>-3.779e+04</td> <td> 3546.699</td> <td>  -10.656</td> <td> 0.000</td> <td>-4.47e+04</td> <td>-3.08e+04</td>
</tr>
<tr>
  <th>Model_80</th>                                                      <td>-1.095e+04</td> <td> 3927.950</td> <td>   -2.787</td> <td> 0.005</td> <td>-1.86e+04</td> <td>-3246.699</td>
</tr>
<tr>
  <th>Model_850</th>                                                     <td>-1.852e+04</td> <td> 7639.575</td> <td>   -2.424</td> <td> 0.015</td> <td>-3.35e+04</td> <td>-3543.743</td>
</tr>
<tr>
  <th>Model_86</th>                                                      <td> -258.4745</td> <td> 5058.406</td> <td>   -0.051</td> <td> 0.959</td> <td>-1.02e+04</td> <td> 9656.992</td>
</tr>
<tr>
  <th>Model_9-2X</th>                                                    <td> -1.26e+04</td> <td> 3400.664</td> <td>   -3.705</td> <td> 0.000</td> <td>-1.93e+04</td> <td>-5933.393</td>
</tr>
<tr>
  <th>Model_9-3</th>                                                     <td>-5468.5127</td> <td> 1697.810</td> <td>   -3.221</td> <td> 0.001</td> <td>-8796.552</td> <td>-2140.473</td>
</tr>
<tr>
  <th>Model_9-3 Griffin</th>                                             <td>-7178.5647</td> <td> 2518.016</td> <td>   -2.851</td> <td> 0.004</td> <td>-1.21e+04</td> <td>-2242.760</td>
</tr>
<tr>
  <th>Model_9-4X</th>                                                    <td>-1.025e+04</td> <td> 3077.219</td> <td>   -3.330</td> <td> 0.001</td> <td>-1.63e+04</td> <td>-4216.049</td>
</tr>
<tr>
  <th>Model_9-5</th>                                                     <td>-1258.6110</td> <td> 2309.845</td> <td>   -0.545</td> <td> 0.586</td> <td>-5786.359</td> <td> 3269.137</td>
</tr>
<tr>
  <th>Model_9-7X</th>                                                    <td>-9297.0307</td> <td> 2471.521</td> <td>   -3.762</td> <td> 0.000</td> <td>-1.41e+04</td> <td>-4452.367</td>
</tr>
<tr>
  <th>Model_90</th>                                                      <td>-1.338e+04</td> <td> 3909.331</td> <td>   -3.423</td> <td> 0.001</td> <td> -2.1e+04</td> <td>-5719.795</td>
</tr>
<tr>
  <th>Model_900</th>                                                     <td>-3.069e+04</td> <td> 1709.977</td> <td>  -17.946</td> <td> 0.000</td> <td> -3.4e+04</td> <td>-2.73e+04</td>
</tr>
<tr>
  <th>Model_9000</th>                                                    <td>-2.771e+04</td> <td> 2288.093</td> <td>  -12.109</td> <td> 0.000</td> <td>-3.22e+04</td> <td>-2.32e+04</td>
</tr>
<tr>
  <th>Model_911</th>                                                     <td> 1.634e+04</td> <td> 1581.258</td> <td>   10.333</td> <td> 0.000</td> <td> 1.32e+04</td> <td> 1.94e+04</td>
</tr>
<tr>
  <th>Model_928</th>                                                     <td>-6.793e+04</td> <td> 4156.843</td> <td>  -16.342</td> <td> 0.000</td> <td>-7.61e+04</td> <td>-5.98e+04</td>
</tr>
<tr>
  <th>Model_929</th>                                                     <td>-1.471e+04</td> <td> 4512.682</td> <td>   -3.260</td> <td> 0.001</td> <td>-2.36e+04</td> <td>-5864.662</td>
</tr>
<tr>
  <th>Model_940</th>                                                     <td>-1.594e+04</td> <td> 7744.494</td> <td>   -2.058</td> <td> 0.040</td> <td>-3.11e+04</td> <td> -756.001</td>
</tr>
<tr>
  <th>Model_944</th>                                                     <td>-5.969e+04</td> <td> 3722.740</td> <td>  -16.033</td> <td> 0.000</td> <td> -6.7e+04</td> <td>-5.24e+04</td>
</tr>
<tr>
  <th>Model_960</th>                                                     <td>-1.902e+04</td> <td> 8013.282</td> <td>   -2.374</td> <td> 0.018</td> <td>-3.47e+04</td> <td>-3314.807</td>
</tr>
<tr>
  <th>Model_968</th>                                                     <td>-6.054e+04</td> <td> 3092.200</td> <td>  -19.578</td> <td> 0.000</td> <td>-6.66e+04</td> <td>-5.45e+04</td>
</tr>
<tr>
  <th>Model_A3</th>                                                      <td> -811.0673</td> <td> 3299.922</td> <td>   -0.246</td> <td> 0.806</td> <td>-7279.560</td> <td> 5657.425</td>
</tr>
<tr>
  <th>Model_A4</th>                                                      <td> 6722.7995</td> <td> 3418.730</td> <td>    1.966</td> <td> 0.049</td> <td>   21.419</td> <td> 1.34e+04</td>
</tr>
<tr>
  <th>Model_A4 allroad</th>                                              <td> 8357.5461</td> <td> 5199.247</td> <td>    1.607</td> <td> 0.108</td> <td>-1833.997</td> <td> 1.85e+04</td>
</tr>
<tr>
  <th>Model_A5</th>                                                      <td> 9898.2838</td> <td> 4043.286</td> <td>    2.448</td> <td> 0.014</td> <td> 1972.651</td> <td> 1.78e+04</td>
</tr>
<tr>
  <th>Model_A6</th>                                                      <td> 1.799e+04</td> <td> 3459.859</td> <td>    5.200</td> <td> 0.000</td> <td> 1.12e+04</td> <td> 2.48e+04</td>
</tr>
<tr>
  <th>Model_A7</th>                                                      <td> 2.808e+04</td> <td> 3868.644</td> <td>    7.259</td> <td> 0.000</td> <td> 2.05e+04</td> <td> 3.57e+04</td>
</tr>
<tr>
  <th>Model_A8</th>                                                      <td> 3.955e+04</td> <td> 4298.463</td> <td>    9.201</td> <td> 0.000</td> <td> 3.11e+04</td> <td>  4.8e+04</td>
</tr>
<tr>
  <th>Model_ALPINA B6 Gran Coupe</th>                                    <td> 1.109e+04</td> <td> 4801.175</td> <td>    2.311</td> <td> 0.021</td> <td> 1682.027</td> <td> 2.05e+04</td>
</tr>
<tr>
  <th>Model_ALPINA B7</th>                                               <td> 3.364e+04</td> <td> 3411.585</td> <td>    9.862</td> <td> 0.000</td> <td>  2.7e+04</td> <td> 4.03e+04</td>
</tr>
<tr>
  <th>Model_AMG GT</th>                                                  <td> 2.548e+04</td> <td> 4221.282</td> <td>    6.036</td> <td> 0.000</td> <td> 1.72e+04</td> <td> 3.38e+04</td>
</tr>
<tr>
  <th>Model_ATS</th>                                                     <td>-2993.8122</td> <td> 1535.403</td> <td>   -1.950</td> <td> 0.051</td> <td>-6003.504</td> <td>   15.879</td>
</tr>
<tr>
  <th>Model_ATS Coupe</th>                                               <td> -261.0055</td> <td> 1702.778</td> <td>   -0.153</td> <td> 0.878</td> <td>-3598.783</td> <td> 3076.772</td>
</tr>
<tr>
  <th>Model_ATS-V</th>                                                   <td>-6017.3505</td> <td> 3704.529</td> <td>   -1.624</td> <td> 0.104</td> <td>-1.33e+04</td> <td> 1244.252</td>
</tr>
<tr>
  <th>Model_Acadia</th>                                                  <td> 3.041e+04</td> <td> 3015.387</td> <td>   10.083</td> <td> 0.000</td> <td> 2.45e+04</td> <td> 3.63e+04</td>
</tr>
<tr>
  <th>Model_Acadia Limited</th>                                          <td> 3.364e+04</td> <td> 5707.609</td> <td>    5.893</td> <td> 0.000</td> <td> 2.24e+04</td> <td> 4.48e+04</td>
</tr>
<tr>
  <th>Model_Accent</th>                                                  <td>-1.219e+04</td> <td> 1883.235</td> <td>   -6.475</td> <td> 0.000</td> <td>-1.59e+04</td> <td>-8502.325</td>
</tr>
<tr>
  <th>Model_Acclaim</th>                                                 <td>-7741.2907</td> <td> 5874.094</td> <td>   -1.318</td> <td> 0.188</td> <td>-1.93e+04</td> <td> 3773.083</td>
</tr>
<tr>
  <th>Model_Accord</th>                                                  <td> -936.1508</td> <td> 1301.867</td> <td>   -0.719</td> <td> 0.472</td> <td>-3488.064</td> <td> 1615.763</td>
</tr>
<tr>
  <th>Model_Accord Crosstour</th>                                        <td> 2447.1240</td> <td> 2439.955</td> <td>    1.003</td> <td> 0.316</td> <td>-2335.666</td> <td> 7229.914</td>
</tr>
<tr>
  <th>Model_Accord Hybrid</th>                                           <td> 6569.1866</td> <td> 2773.200</td> <td>    2.369</td> <td> 0.018</td> <td> 1133.172</td> <td>  1.2e+04</td>
</tr>
<tr>
  <th>Model_Accord Plug-In Hybrid</th>                                   <td> 1.492e+04</td> <td> 6966.629</td> <td>    2.142</td> <td> 0.032</td> <td> 1267.134</td> <td> 2.86e+04</td>
</tr>
<tr>
  <th>Model_Achieva</th>                                                 <td>-1.016e+04</td> <td> 3184.931</td> <td>   -3.190</td> <td> 0.001</td> <td>-1.64e+04</td> <td>-3915.369</td>
</tr>
<tr>
  <th>Model_ActiveHybrid 5</th>                                          <td> 4109.9401</td> <td> 4751.596</td> <td>    0.865</td> <td> 0.387</td> <td>-5204.119</td> <td> 1.34e+04</td>
</tr>
<tr>
  <th>Model_ActiveHybrid 7</th>                                          <td> 2.623e+04</td> <td> 4738.263</td> <td>    5.536</td> <td> 0.000</td> <td> 1.69e+04</td> <td> 3.55e+04</td>
</tr>
<tr>
  <th>Model_ActiveHybrid X6</th>                                         <td> 6405.2906</td> <td> 5442.959</td> <td>    1.177</td> <td> 0.239</td> <td>-4263.975</td> <td> 1.71e+04</td>
</tr>
<tr>
  <th>Model_Aerio</th>                                                   <td>-4746.3315</td> <td> 1435.568</td> <td>   -3.306</td> <td> 0.001</td> <td>-7560.327</td> <td>-1932.336</td>
</tr>
<tr>
  <th>Model_Aerostar</th>                                                <td>-5570.1393</td> <td> 4624.856</td> <td>   -1.204</td> <td> 0.228</td> <td>-1.46e+04</td> <td> 3495.485</td>
</tr>
<tr>
  <th>Model_Alero</th>                                                   <td> 1890.5518</td> <td> 1856.897</td> <td>    1.018</td> <td> 0.309</td> <td>-1749.329</td> <td> 5530.433</td>
</tr>
<tr>
  <th>Model_Allante</th>                                                 <td>-3.437e+04</td> <td> 4328.608</td> <td>   -7.939</td> <td> 0.000</td> <td>-4.29e+04</td> <td>-2.59e+04</td>
</tr>
<tr>
  <th>Model_Alpina</th>                                                  <td>   7.4e+04</td> <td> 7347.060</td> <td>   10.072</td> <td> 0.000</td> <td> 5.96e+04</td> <td> 8.84e+04</td>
</tr>
<tr>
  <th>Model_Altima</th>                                                  <td> 4138.0542</td> <td> 2323.920</td> <td>    1.781</td> <td> 0.075</td> <td> -417.284</td> <td> 8693.392</td>
</tr>
<tr>
  <th>Model_Altima Hybrid</th>                                           <td> 1.078e+04</td> <td> 4411.113</td> <td>    2.444</td> <td> 0.015</td> <td> 2135.565</td> <td> 1.94e+04</td>
</tr>
<tr>
  <th>Model_Amanti</th>                                                  <td>-1794.8874</td> <td> 3991.615</td> <td>   -0.450</td> <td> 0.653</td> <td>-9619.235</td> <td> 6029.460</td>
</tr>
<tr>
  <th>Model_Armada</th>                                                  <td> 8858.5532</td> <td> 2405.492</td> <td>    3.683</td> <td> 0.000</td> <td> 4143.319</td> <td> 1.36e+04</td>
</tr>
<tr>
  <th>Model_Arnage</th>                                                  <td>-3.292e+04</td> <td> 3112.412</td> <td>  -10.576</td> <td> 0.000</td> <td> -3.9e+04</td> <td>-2.68e+04</td>
</tr>
<tr>
  <th>Model_Aspen</th>                                                   <td>-8972.7260</td> <td> 4341.690</td> <td>   -2.067</td> <td> 0.039</td> <td>-1.75e+04</td> <td> -462.164</td>
</tr>
<tr>
  <th>Model_Aspire</th>                                                  <td>-8417.4378</td> <td> 3326.220</td> <td>   -2.531</td> <td> 0.011</td> <td>-1.49e+04</td> <td>-1897.395</td>
</tr>
<tr>
  <th>Model_Astro</th>                                                   <td>-4501.6512</td> <td> 5081.463</td> <td>   -0.886</td> <td> 0.376</td> <td>-1.45e+04</td> <td> 5459.012</td>
</tr>
<tr>
  <th>Model_Astro Cargo</th>                                             <td>-2940.4214</td> <td> 5301.089</td> <td>   -0.555</td> <td> 0.579</td> <td>-1.33e+04</td> <td> 7450.751</td>
</tr>
<tr>
  <th>Model_Aurora</th>                                                  <td> 9687.7605</td> <td> 3257.563</td> <td>    2.974</td> <td> 0.003</td> <td> 3302.299</td> <td> 1.61e+04</td>
</tr>
<tr>
  <th>Model_Avalanche</th>                                               <td>-1.085e+04</td> <td> 2398.742</td> <td>   -4.525</td> <td> 0.000</td> <td>-1.56e+04</td> <td>-6152.063</td>
</tr>
<tr>
  <th>Model_Avalon</th>                                                  <td> 5349.3984</td> <td> 2047.145</td> <td>    2.613</td> <td> 0.009</td> <td> 1336.593</td> <td> 9362.204</td>
</tr>
<tr>
  <th>Model_Avalon Hybrid</th>                                           <td> 1.392e+04</td> <td> 2612.936</td> <td>    5.327</td> <td> 0.000</td> <td> 8796.440</td> <td>  1.9e+04</td>
</tr>
<tr>
  <th>Model_Avenger</th>                                                 <td>-5058.4362</td> <td> 4746.910</td> <td>   -1.066</td> <td> 0.287</td> <td>-1.44e+04</td> <td> 4246.437</td>
</tr>
<tr>
  <th>Model_Aventador</th>                                               <td>-3.804e+04</td> <td> 2688.112</td> <td>  -14.151</td> <td> 0.000</td> <td>-4.33e+04</td> <td>-3.28e+04</td>
</tr>
<tr>
  <th>Model_Aveo</th>                                                    <td>-1.755e+04</td> <td> 2211.886</td> <td>   -7.936</td> <td> 0.000</td> <td>-2.19e+04</td> <td>-1.32e+04</td>
</tr>
<tr>
  <th>Model_Aviator</th>                                                 <td> 4.394e+04</td> <td> 3143.731</td> <td>   13.976</td> <td> 0.000</td> <td> 3.78e+04</td> <td> 5.01e+04</td>
</tr>
<tr>
  <th>Model_Axxess</th>                                                  <td>  179.8388</td> <td> 5433.977</td> <td>    0.033</td> <td> 0.974</td> <td>-1.05e+04</td> <td> 1.08e+04</td>
</tr>
<tr>
  <th>Model_Azera</th>                                                   <td> 2371.4562</td> <td> 2954.695</td> <td>    0.803</td> <td> 0.422</td> <td>-3420.326</td> <td> 8163.238</td>
</tr>
<tr>
  <th>Model_Aztek</th>                                                   <td>-4599.4181</td> <td> 2893.852</td> <td>   -1.589</td> <td> 0.112</td> <td>-1.03e+04</td> <td> 1073.098</td>
</tr>
<tr>
  <th>Model_Azure</th>                                                   <td> 4.783e+04</td> <td> 4565.942</td> <td>   10.475</td> <td> 0.000</td> <td> 3.89e+04</td> <td> 5.68e+04</td>
</tr>
<tr>
  <th>Model_Azure T</th>                                                 <td> 6.783e+04</td> <td> 7295.792</td> <td>    9.298</td> <td> 0.000</td> <td> 5.35e+04</td> <td> 8.21e+04</td>
</tr>
<tr>
  <th>Model_B-Class Electric Drive</th>                                  <td>-3.068e+04</td> <td> 6228.531</td> <td>   -4.925</td> <td> 0.000</td> <td>-4.29e+04</td> <td>-1.85e+04</td>
</tr>
<tr>
  <th>Model_B-Series</th>                                                <td>-9485.9804</td> <td> 2447.952</td> <td>   -3.875</td> <td> 0.000</td> <td>-1.43e+04</td> <td>-4687.516</td>
</tr>
<tr>
  <th>Model_B-Series Pickup</th>                                         <td>-2.235e+04</td> <td> 1872.186</td> <td>  -11.936</td> <td> 0.000</td> <td> -2.6e+04</td> <td>-1.87e+04</td>
</tr>
<tr>
  <th>Model_B-Series Truck</th>                                          <td>-1.199e+04</td> <td> 2640.882</td> <td>   -4.540</td> <td> 0.000</td> <td>-1.72e+04</td> <td>-6812.004</td>
</tr>
<tr>
  <th>Model_B9 Tribeca</th>                                              <td> 2858.9747</td> <td> 1875.199</td> <td>    1.525</td> <td> 0.127</td> <td> -816.782</td> <td> 6534.732</td>
</tr>
<tr>
  <th>Model_BRZ</th>                                                     <td>  -84.9446</td> <td> 2338.316</td> <td>   -0.036</td> <td> 0.971</td> <td>-4668.501</td> <td> 4498.612</td>
</tr>
<tr>
  <th>Model_Baja</th>                                                    <td>-8044.0389</td> <td> 2148.427</td> <td>   -3.744</td> <td> 0.000</td> <td>-1.23e+04</td> <td>-3832.702</td>
</tr>
<tr>
  <th>Model_Beetle</th>                                                  <td>-1.086e+04</td> <td> 5732.510</td> <td>   -1.894</td> <td> 0.058</td> <td>-2.21e+04</td> <td>  380.365</td>
</tr>
<tr>
  <th>Model_Beetle Convertible</th>                                      <td>-1.314e+04</td> <td> 5695.114</td> <td>   -2.308</td> <td> 0.021</td> <td>-2.43e+04</td> <td>-1981.114</td>
</tr>
<tr>
  <th>Model_Beretta</th>                                                 <td>-1.767e+04</td> <td> 3527.827</td> <td>   -5.007</td> <td> 0.000</td> <td>-2.46e+04</td> <td>-1.08e+04</td>
</tr>
<tr>
  <th>Model_Black Diamond Avalanche</th>                                 <td>-1.321e+04</td> <td> 3283.858</td> <td>   -4.024</td> <td> 0.000</td> <td>-1.96e+04</td> <td>-6775.921</td>
</tr>
<tr>
  <th>Model_Blackwood</th>                                               <td> 4.745e+04</td> <td> 7167.034</td> <td>    6.620</td> <td> 0.000</td> <td> 3.34e+04</td> <td> 6.15e+04</td>
</tr>
<tr>
  <th>Model_Blazer</th>                                                  <td>-5671.0180</td> <td> 2617.027</td> <td>   -2.167</td> <td> 0.030</td> <td>-1.08e+04</td> <td> -541.133</td>
</tr>
<tr>
  <th>Model_Bolt EV</th>                                                 <td>-5034.7444</td> <td> 6877.099</td> <td>   -0.732</td> <td> 0.464</td> <td>-1.85e+04</td> <td> 8445.717</td>
</tr>
<tr>
  <th>Model_Bonneville</th>                                              <td> 3693.4302</td> <td> 2589.120</td> <td>    1.427</td> <td> 0.154</td> <td>-1381.753</td> <td> 8768.613</td>
</tr>
<tr>
  <th>Model_Borrego</th>                                                 <td>-4688.9815</td> <td> 2404.845</td> <td>   -1.950</td> <td> 0.051</td> <td>-9402.948</td> <td>   24.985</td>
</tr>
<tr>
  <th>Model_Boxster</th>                                                 <td> -3.02e+04</td> <td> 2687.040</td> <td>  -11.239</td> <td> 0.000</td> <td>-3.55e+04</td> <td>-2.49e+04</td>
</tr>
<tr>
  <th>Model_Bravada</th>                                                 <td> 8562.3898</td> <td> 2767.956</td> <td>    3.093</td> <td> 0.002</td> <td> 3136.654</td> <td>  1.4e+04</td>
</tr>
<tr>
  <th>Model_Breeze</th>                                                  <td>-8275.6075</td> <td> 5767.933</td> <td>   -1.435</td> <td> 0.151</td> <td>-1.96e+04</td> <td> 3030.670</td>
</tr>
<tr>
  <th>Model_Bronco</th>                                                  <td>-9022.1320</td> <td> 3243.733</td> <td>   -2.781</td> <td> 0.005</td> <td>-1.54e+04</td> <td>-2663.780</td>
</tr>
<tr>
  <th>Model_Bronco II</th>                                               <td>-2713.5384</td> <td> 5491.270</td> <td>   -0.494</td> <td> 0.621</td> <td>-1.35e+04</td> <td> 8050.426</td>
</tr>
<tr>
  <th>Model_Brooklands</th>                                              <td>  4.76e+04</td> <td> 5374.330</td> <td>    8.858</td> <td> 0.000</td> <td> 3.71e+04</td> <td> 5.81e+04</td>
</tr>
<tr>
  <th>Model_Brougham</th>                                                <td> -2.54e+04</td> <td> 4333.065</td> <td>   -5.862</td> <td> 0.000</td> <td>-3.39e+04</td> <td>-1.69e+04</td>
</tr>
<tr>
  <th>Model_C-Class</th>                                                 <td>-2.297e+04</td> <td> 1617.539</td> <td>  -14.201</td> <td> 0.000</td> <td>-2.61e+04</td> <td>-1.98e+04</td>
</tr>
<tr>
  <th>Model_C-Max Hybrid</th>                                            <td> 5595.1606</td> <td> 3310.469</td> <td>    1.690</td> <td> 0.091</td> <td> -894.007</td> <td> 1.21e+04</td>
</tr>
<tr>
  <th>Model_C/K 1500 Series</th>                                         <td>-2.686e+04</td> <td> 1883.631</td> <td>  -14.260</td> <td> 0.000</td> <td>-3.06e+04</td> <td>-2.32e+04</td>
</tr>
<tr>
  <th>Model_C/K 2500 Series</th>                                         <td>-2.869e+04</td> <td> 2974.113</td> <td>   -9.647</td> <td> 0.000</td> <td>-3.45e+04</td> <td>-2.29e+04</td>
</tr>
<tr>
  <th>Model_C30</th>                                                     <td>-7302.2787</td> <td> 8305.760</td> <td>   -0.879</td> <td> 0.379</td> <td>-2.36e+04</td> <td> 8978.638</td>
</tr>
<tr>
  <th>Model_C36 AMG</th>                                                 <td>-4.647e+04</td> <td> 4213.632</td> <td>  -11.027</td> <td> 0.000</td> <td>-5.47e+04</td> <td>-3.82e+04</td>
</tr>
<tr>
  <th>Model_C43 AMG</th>                                                 <td>-5.205e+04</td> <td> 4173.485</td> <td>  -12.472</td> <td> 0.000</td> <td>-6.02e+04</td> <td>-4.39e+04</td>
</tr>
<tr>
  <th>Model_C70</th>                                                     <td>-1174.6724</td> <td> 8693.229</td> <td>   -0.135</td> <td> 0.893</td> <td>-1.82e+04</td> <td> 1.59e+04</td>
</tr>
<tr>
  <th>Model_C8</th>                                                      <td> 1.723e+04</td> <td> 4178.446</td> <td>    4.123</td> <td> 0.000</td> <td> 9038.610</td> <td> 2.54e+04</td>
</tr>
<tr>
  <th>Model_CC</th>                                                      <td>-1909.6530</td> <td> 5909.877</td> <td>   -0.323</td> <td> 0.747</td> <td>-1.35e+04</td> <td> 9674.862</td>
</tr>
<tr>
  <th>Model_CL</th>                                                      <td>-4630.3642</td> <td> 2437.461</td> <td>   -1.900</td> <td> 0.058</td> <td>-9408.266</td> <td>  147.537</td>
</tr>
<tr>
  <th>Model_CL-Class</th>                                                <td> 5.177e+04</td> <td> 2344.802</td> <td>   22.078</td> <td> 0.000</td> <td> 4.72e+04</td> <td> 5.64e+04</td>
</tr>
<tr>
  <th>Model_CLA-Class</th>                                               <td>-3.034e+04</td> <td> 2639.690</td> <td>  -11.495</td> <td> 0.000</td> <td>-3.55e+04</td> <td>-2.52e+04</td>
</tr>
<tr>
  <th>Model_CLK-Class</th>                                               <td>-1.008e+04</td> <td> 1983.465</td> <td>   -5.084</td> <td> 0.000</td> <td> -1.4e+04</td> <td>-6195.382</td>
</tr>
<tr>
  <th>Model_CLS-Class</th>                                               <td>-7135.9362</td> <td> 2241.011</td> <td>   -3.184</td> <td> 0.001</td> <td>-1.15e+04</td> <td>-2743.116</td>
</tr>
<tr>
  <th>Model_CR-V</th>                                                    <td>-3213.6652</td> <td> 1473.669</td> <td>   -2.181</td> <td> 0.029</td> <td>-6102.345</td> <td> -324.985</td>
</tr>
<tr>
  <th>Model_CR-Z</th>                                                    <td>-6523.0908</td> <td> 1979.873</td> <td>   -3.295</td> <td> 0.001</td> <td>-1.04e+04</td> <td>-2642.153</td>
</tr>
<tr>
  <th>Model_CT 200h</th>                                                 <td>-1.698e+04</td> <td> 4380.215</td> <td>   -3.877</td> <td> 0.000</td> <td>-2.56e+04</td> <td>-8397.809</td>
</tr>
<tr>
  <th>Model_CT6</th>                                                     <td> 1.138e+04</td> <td> 1914.544</td> <td>    5.946</td> <td> 0.000</td> <td> 7630.399</td> <td> 1.51e+04</td>
</tr>
<tr>
  <th>Model_CTS</th>                                                     <td> 7077.5706</td> <td> 1447.444</td> <td>    4.890</td> <td> 0.000</td> <td> 4240.298</td> <td> 9914.843</td>
</tr>
<tr>
  <th>Model_CTS Coupe</th>                                               <td>-1386.5680</td> <td> 1927.828</td> <td>   -0.719</td> <td> 0.472</td> <td>-5165.488</td> <td> 2392.352</td>
</tr>
<tr>
  <th>Model_CTS Wagon</th>                                               <td>   36.2807</td> <td> 1865.185</td> <td>    0.019</td> <td> 0.984</td> <td>-3619.846</td> <td> 3692.408</td>
</tr>
<tr>
  <th>Model_CTS-V</th>                                                   <td>-2.941e+04</td> <td> 4172.870</td> <td>   -7.048</td> <td> 0.000</td> <td>-3.76e+04</td> <td>-2.12e+04</td>
</tr>
<tr>
  <th>Model_CTS-V Coupe</th>                                             <td>-2.227e+04</td> <td> 4179.548</td> <td>   -5.329</td> <td> 0.000</td> <td>-3.05e+04</td> <td>-1.41e+04</td>
</tr>
<tr>
  <th>Model_CTS-V Wagon</th>                                             <td>-2.453e+04</td> <td> 4206.155</td> <td>   -5.832</td> <td> 0.000</td> <td>-3.28e+04</td> <td>-1.63e+04</td>
</tr>
<tr>
  <th>Model_CX-3</th>                                                    <td>-1.287e+04</td> <td> 2522.807</td> <td>   -5.102</td> <td> 0.000</td> <td>-1.78e+04</td> <td>-7927.327</td>
</tr>
<tr>
  <th>Model_CX-5</th>                                                    <td>-8339.9586</td> <td> 1949.773</td> <td>   -4.277</td> <td> 0.000</td> <td>-1.22e+04</td> <td>-4518.022</td>
</tr>
<tr>
  <th>Model_CX-7</th>                                                    <td>-5210.5659</td> <td> 2109.468</td> <td>   -2.470</td> <td> 0.014</td> <td>-9345.536</td> <td>-1075.596</td>
</tr>
<tr>
  <th>Model_CX-9</th>                                                    <td>-4694.9725</td> <td> 2277.174</td> <td>   -2.062</td> <td> 0.039</td> <td>-9158.679</td> <td> -231.266</td>
</tr>
<tr>
  <th>Model_Cabrio</th>                                                  <td>-1.333e+04</td> <td> 5650.137</td> <td>   -2.360</td> <td> 0.018</td> <td>-2.44e+04</td> <td>-2256.611</td>
</tr>
<tr>
  <th>Model_Cabriolet</th>                                               <td>-2.342e+04</td> <td> 4145.391</td> <td>   -5.650</td> <td> 0.000</td> <td>-3.15e+04</td> <td>-1.53e+04</td>
</tr>
<tr>
  <th>Model_Cadenza</th>                                                 <td> 3311.6281</td> <td> 2580.484</td> <td>    1.283</td> <td> 0.199</td> <td>-1746.626</td> <td> 8369.882</td>
</tr>
<tr>
  <th>Model_Caliber</th>                                                 <td>-5587.4684</td> <td> 4647.545</td> <td>   -1.202</td> <td> 0.229</td> <td>-1.47e+04</td> <td> 3522.630</td>
</tr>
<tr>
  <th>Model_California</th>                                              <td> -5.71e+04</td> <td> 4054.406</td> <td>  -14.084</td> <td> 0.000</td> <td>-6.51e+04</td> <td>-4.92e+04</td>
</tr>
<tr>
  <th>Model_California T</th>                                            <td>-7.455e+04</td> <td> 6789.074</td> <td>  -10.981</td> <td> 0.000</td> <td>-8.79e+04</td> <td>-6.12e+04</td>
</tr>
<tr>
  <th>Model_Camaro</th>                                                  <td>-1.982e+04</td> <td> 1935.692</td> <td>  -10.240</td> <td> 0.000</td> <td>-2.36e+04</td> <td> -1.6e+04</td>
</tr>
<tr>
  <th>Model_Camry</th>                                                   <td> -768.9230</td> <td> 1973.190</td> <td>   -0.390</td> <td> 0.697</td> <td>-4636.763</td> <td> 3098.917</td>
</tr>
<tr>
  <th>Model_Camry Hybrid</th>                                            <td> 3239.0193</td> <td> 2631.128</td> <td>    1.231</td> <td> 0.218</td> <td>-1918.507</td> <td> 8396.546</td>
</tr>
<tr>
  <th>Model_Camry Solara</th>                                            <td> 1462.3325</td> <td> 1561.648</td> <td>    0.936</td> <td> 0.349</td> <td>-1598.803</td> <td> 4523.467</td>
</tr>
<tr>
  <th>Model_Canyon</th>                                                  <td> 1.762e+04</td> <td> 3396.372</td> <td>    5.188</td> <td> 0.000</td> <td>  1.1e+04</td> <td> 2.43e+04</td>
</tr>
<tr>
  <th>Model_Caprice</th>                                                 <td>-2.454e+04</td> <td> 3552.930</td> <td>   -6.907</td> <td> 0.000</td> <td>-3.15e+04</td> <td>-1.76e+04</td>
</tr>
<tr>
  <th>Model_Captiva Sport</th>                                           <td>-8903.6780</td> <td> 2440.098</td> <td>   -3.649</td> <td> 0.000</td> <td>-1.37e+04</td> <td>-4120.608</td>
</tr>
<tr>
  <th>Model_Caravan</th>                                                 <td>-2256.5822</td> <td> 6507.581</td> <td>   -0.347</td> <td> 0.729</td> <td> -1.5e+04</td> <td> 1.05e+04</td>
</tr>
<tr>
  <th>Model_Carrera GT</th>                                              <td> 2.958e+05</td> <td> 4838.435</td> <td>   61.129</td> <td> 0.000</td> <td> 2.86e+05</td> <td> 3.05e+05</td>
</tr>
<tr>
  <th>Model_Cascada</th>                                                 <td>-6397.6226</td> <td> 3467.733</td> <td>   -1.845</td> <td> 0.065</td> <td>-1.32e+04</td> <td>  399.813</td>
</tr>
<tr>
  <th>Model_Catera</th>                                                  <td> -2.02e+04</td> <td> 3626.343</td> <td>   -5.571</td> <td> 0.000</td> <td>-2.73e+04</td> <td>-1.31e+04</td>
</tr>
<tr>
  <th>Model_Cavalier</th>                                                <td>-9759.2314</td> <td> 2220.297</td> <td>   -4.395</td> <td> 0.000</td> <td>-1.41e+04</td> <td>-5407.015</td>
</tr>
<tr>
  <th>Model_Cayenne</th>                                                 <td>-1.671e+04</td> <td> 2186.647</td> <td>   -7.644</td> <td> 0.000</td> <td> -2.1e+04</td> <td>-1.24e+04</td>
</tr>
<tr>
  <th>Model_Cayman</th>                                                  <td>-2.183e+04</td> <td> 2682.490</td> <td>   -8.139</td> <td> 0.000</td> <td>-2.71e+04</td> <td>-1.66e+04</td>
</tr>
<tr>
  <th>Model_Cayman S</th>                                                <td>-1.731e+04</td> <td> 6669.614</td> <td>   -2.595</td> <td> 0.009</td> <td>-3.04e+04</td> <td>-4233.505</td>
</tr>
<tr>
  <th>Model_Celebrity</th>                                               <td>-1.675e+04</td> <td> 5549.932</td> <td>   -3.018</td> <td> 0.003</td> <td>-2.76e+04</td> <td>-5871.722</td>
</tr>
<tr>
  <th>Model_Celica</th>                                                  <td> 1453.2343</td> <td> 2092.012</td> <td>    0.695</td> <td> 0.487</td> <td>-2647.519</td> <td> 5553.988</td>
</tr>
<tr>
  <th>Model_Century</th>                                                 <td>-3744.3741</td> <td> 4041.879</td> <td>   -0.926</td> <td> 0.354</td> <td>-1.17e+04</td> <td> 4178.500</td>
</tr>
<tr>
  <th>Model_Challenger</th>                                              <td>-1.753e+04</td> <td> 4466.184</td> <td>   -3.924</td> <td> 0.000</td> <td>-2.63e+04</td> <td>-8771.066</td>
</tr>
<tr>
  <th>Model_Charger</th>                                                 <td> -1.71e+04</td> <td> 4440.526</td> <td>   -3.850</td> <td> 0.000</td> <td>-2.58e+04</td> <td>-8393.928</td>
</tr>
<tr>
  <th>Model_Chevy Van</th>                                               <td>-3.153e+04</td> <td> 3826.739</td> <td>   -8.240</td> <td> 0.000</td> <td> -3.9e+04</td> <td> -2.4e+04</td>
</tr>
<tr>
  <th>Model_Ciera</th>                                                   <td>-1.249e+04</td> <td> 4031.859</td> <td>   -3.097</td> <td> 0.002</td> <td>-2.04e+04</td> <td>-4585.243</td>
</tr>
<tr>
  <th>Model_Cirrus</th>                                                  <td>-2.324e+04</td> <td> 5463.644</td> <td>   -4.253</td> <td> 0.000</td> <td>-3.39e+04</td> <td>-1.25e+04</td>
</tr>
<tr>
  <th>Model_City Express</th>                                            <td>-9749.5732</td> <td> 4894.168</td> <td>   -1.992</td> <td> 0.046</td> <td>-1.93e+04</td> <td> -156.046</td>
</tr>
<tr>
  <th>Model_Civic</th>                                                   <td>-5528.5440</td> <td> 1225.838</td> <td>   -4.510</td> <td> 0.000</td> <td>-7931.427</td> <td>-3125.661</td>
</tr>
<tr>
  <th>Model_Civic CRX</th>                                               <td>-1.046e+04</td> <td> 3353.808</td> <td>   -3.120</td> <td> 0.002</td> <td> -1.7e+04</td> <td>-3888.493</td>
</tr>
<tr>
  <th>Model_Civic del Sol</th>                                           <td>-1.065e+04</td> <td> 2667.870</td> <td>   -3.994</td> <td> 0.000</td> <td>-1.59e+04</td> <td>-5425.350</td>
</tr>
<tr>
  <th>Model_Classic</th>                                                 <td>-7115.7111</td> <td> 5226.247</td> <td>   -1.362</td> <td> 0.173</td> <td>-1.74e+04</td> <td> 3128.757</td>
</tr>
<tr>
  <th>Model_Cobalt</th>                                                  <td>-1.143e+04</td> <td> 2064.231</td> <td>   -5.539</td> <td> 0.000</td> <td>-1.55e+04</td> <td>-7386.484</td>
</tr>
<tr>
  <th>Model_Colorado</th>                                                <td>-1.735e+04</td> <td> 2040.887</td> <td>   -8.499</td> <td> 0.000</td> <td>-2.13e+04</td> <td>-1.33e+04</td>
</tr>
<tr>
  <th>Model_Colt</th>                                                    <td>-7737.6769</td> <td> 4513.986</td> <td>   -1.714</td> <td> 0.087</td> <td>-1.66e+04</td> <td> 1110.621</td>
</tr>
<tr>
  <th>Model_Concorde</th>                                                <td>-5574.0592</td> <td> 4230.577</td> <td>   -1.318</td> <td> 0.188</td> <td>-1.39e+04</td> <td> 2718.700</td>
</tr>
<tr>
  <th>Model_Continental</th>                                             <td>  4.42e+04</td> <td> 2083.440</td> <td>   21.216</td> <td> 0.000</td> <td> 4.01e+04</td> <td> 4.83e+04</td>
</tr>
<tr>
  <th>Model_Continental Flying Spur</th>                                 <td>-1.219e+05</td> <td> 4780.760</td> <td>  -25.501</td> <td> 0.000</td> <td>-1.31e+05</td> <td>-1.13e+05</td>
</tr>
<tr>
  <th>Model_Continental Flying Spur Speed</th>                           <td>-1.103e+05</td> <td> 4689.084</td> <td>  -23.525</td> <td> 0.000</td> <td> -1.2e+05</td> <td>-1.01e+05</td>
</tr>
<tr>
  <th>Model_Continental GT</th>                                          <td>  -9.3e+04</td> <td> 2825.916</td> <td>  -32.911</td> <td> 0.000</td> <td>-9.85e+04</td> <td>-8.75e+04</td>
</tr>
<tr>
  <th>Model_Continental GT Speed</th>                                    <td>-1.072e+05</td> <td> 5449.821</td> <td>  -19.673</td> <td> 0.000</td> <td>-1.18e+05</td> <td>-9.65e+04</td>
</tr>
<tr>
  <th>Model_Continental GT Speed Convertible</th>                        <td> -9.42e+04</td> <td> 7450.412</td> <td>  -12.643</td> <td> 0.000</td> <td>-1.09e+05</td> <td>-7.96e+04</td>
</tr>
<tr>
  <th>Model_Continental GT3-R</th>                                       <td> 2.856e+04</td> <td> 7340.429</td> <td>    3.890</td> <td> 0.000</td> <td> 1.42e+04</td> <td> 4.29e+04</td>
</tr>
<tr>
  <th>Model_Continental GTC</th>                                         <td>-9.997e+04</td> <td> 3675.191</td> <td>  -27.202</td> <td> 0.000</td> <td>-1.07e+05</td> <td>-9.28e+04</td>
</tr>
<tr>
  <th>Model_Continental GTC Speed</th>                                   <td>-9.059e+04</td> <td> 5441.016</td> <td>  -16.650</td> <td> 0.000</td> <td>-1.01e+05</td> <td>-7.99e+04</td>
</tr>
<tr>
  <th>Model_Continental R</th>                                           <td> 4.282e+04</td> <td> 5397.907</td> <td>    7.932</td> <td> 0.000</td> <td> 3.22e+04</td> <td> 5.34e+04</td>
</tr>
<tr>
  <th>Model_Continental SR</th>                                          <td> 5.084e+04</td> <td> 5392.769</td> <td>    9.428</td> <td> 0.000</td> <td> 4.03e+04</td> <td> 6.14e+04</td>
</tr>
<tr>
  <th>Model_Continental Supersports</th>                                 <td>-5.408e+04</td> <td> 4769.635</td> <td>  -11.339</td> <td> 0.000</td> <td>-6.34e+04</td> <td>-4.47e+04</td>
</tr>
<tr>
  <th>Model_Continental Supersports Convertible</th>                     <td>-4.915e+04</td> <td> 5565.476</td> <td>   -8.831</td> <td> 0.000</td> <td>-6.01e+04</td> <td>-3.82e+04</td>
</tr>
<tr>
  <th>Model_Contour</th>                                                 <td>-5801.8538</td> <td> 3741.030</td> <td>   -1.551</td> <td> 0.121</td> <td>-1.31e+04</td> <td> 1531.297</td>
</tr>
<tr>
  <th>Model_Contour SVT</th>                                             <td>-5898.0901</td> <td> 4257.310</td> <td>   -1.385</td> <td> 0.166</td> <td>-1.42e+04</td> <td> 2447.071</td>
</tr>
<tr>
  <th>Model_Corniche</th>                                                <td>  5.97e+04</td> <td> 6658.734</td> <td>    8.966</td> <td> 0.000</td> <td> 4.67e+04</td> <td> 7.28e+04</td>
</tr>
<tr>
  <th>Model_Corolla</th>                                                 <td>-6613.6446</td> <td> 1515.544</td> <td>   -4.364</td> <td> 0.000</td> <td>-9584.408</td> <td>-3642.881</td>
</tr>
<tr>
  <th>Model_Corolla iM</th>                                              <td>-9384.7681</td> <td> 5042.362</td> <td>   -1.861</td> <td> 0.063</td> <td>-1.93e+04</td> <td>  499.250</td>
</tr>
<tr>
  <th>Model_Corrado</th>                                                 <td> -1.69e+04</td> <td> 6353.662</td> <td>   -2.659</td> <td> 0.008</td> <td>-2.94e+04</td> <td>-4442.753</td>
</tr>
<tr>
  <th>Model_Corsica</th>                                                 <td> -1.89e+04</td> <td> 5387.116</td> <td>   -3.509</td> <td> 0.000</td> <td>-2.95e+04</td> <td>-8340.982</td>
</tr>
<tr>
  <th>Model_Corvette</th>                                                <td>-1.072e+04</td> <td> 1802.848</td> <td>   -5.948</td> <td> 0.000</td> <td>-1.43e+04</td> <td>-7188.582</td>
</tr>
<tr>
  <th>Model_Corvette Stingray</th>                                       <td>-8236.3542</td> <td> 3843.596</td> <td>   -2.143</td> <td> 0.032</td> <td>-1.58e+04</td> <td> -702.153</td>
</tr>
<tr>
  <th>Model_Coupe</th>                                                   <td> -1.24e+04</td> <td> 4632.458</td> <td>   -2.678</td> <td> 0.007</td> <td>-2.15e+04</td> <td>-3323.919</td>
</tr>
<tr>
  <th>Model_Cressida</th>                                                <td>-7550.9591</td> <td> 4389.473</td> <td>   -1.720</td> <td> 0.085</td> <td>-1.62e+04</td> <td> 1053.267</td>
</tr>
<tr>
  <th>Model_Crossfire</th>                                               <td>  858.5848</td> <td> 3925.820</td> <td>    0.219</td> <td> 0.827</td> <td>-6836.790</td> <td> 8553.960</td>
</tr>
<tr>
  <th>Model_Crosstour</th>                                               <td> 1307.5126</td> <td> 1795.701</td> <td>    0.728</td> <td> 0.467</td> <td>-2212.413</td> <td> 4827.438</td>
</tr>
<tr>
  <th>Model_Crosstrek</th>                                               <td>-9423.0195</td> <td> 2373.482</td> <td>   -3.970</td> <td> 0.000</td> <td>-1.41e+04</td> <td>-4770.529</td>
</tr>
<tr>
  <th>Model_Crown Victoria</th>                                          <td> 7760.4925</td> <td> 4207.571</td> <td>    1.844</td> <td> 0.065</td> <td> -487.171</td> <td>  1.6e+04</td>
</tr>
<tr>
  <th>Model_Cruze</th>                                                   <td>-1.385e+04</td> <td> 1949.323</td> <td>   -7.104</td> <td> 0.000</td> <td>-1.77e+04</td> <td>   -1e+04</td>
</tr>
<tr>
  <th>Model_Cruze Limited</th>                                           <td>-1.416e+04</td> <td> 2719.543</td> <td>   -5.207</td> <td> 0.000</td> <td>-1.95e+04</td> <td>-8830.071</td>
</tr>
<tr>
  <th>Model_Cube</th>                                                    <td>-2906.9453</td> <td> 2839.926</td> <td>   -1.024</td> <td> 0.306</td> <td>-8473.756</td> <td> 2659.866</td>
</tr>
<tr>
  <th>Model_Custom Cruiser</th>                                          <td>-1.348e+04</td> <td> 4104.465</td> <td>   -3.283</td> <td> 0.001</td> <td>-2.15e+04</td> <td>-5429.837</td>
</tr>
<tr>
  <th>Model_Cutlass</th>                                                 <td>-1.448e+04</td> <td> 3195.851</td> <td>   -4.530</td> <td> 0.000</td> <td>-2.07e+04</td> <td>-8211.544</td>
</tr>
<tr>
  <th>Model_Cutlass Calais</th>                                          <td>-5888.1219</td> <td> 2262.419</td> <td>   -2.603</td> <td> 0.009</td> <td>-1.03e+04</td> <td>-1453.338</td>
</tr>
<tr>
  <th>Model_Cutlass Ciera</th>                                           <td>-1.015e+04</td> <td> 2237.890</td> <td>   -4.537</td> <td> 0.000</td> <td>-1.45e+04</td> <td>-5766.258</td>
</tr>
<tr>
  <th>Model_Cutlass Supreme</th>                                         <td>-1.372e+04</td> <td> 2733.388</td> <td>   -5.020</td> <td> 0.000</td> <td>-1.91e+04</td> <td>-8363.661</td>
</tr>
<tr>
  <th>Model_DB7</th>                                                     <td>-3.336e+04</td> <td> 3473.144</td> <td>   -9.604</td> <td> 0.000</td> <td>-4.02e+04</td> <td>-2.65e+04</td>
</tr>
<tr>
  <th>Model_DB9</th>                                                     <td>-1.584e+04</td> <td> 2621.844</td> <td>   -6.043</td> <td> 0.000</td> <td> -2.1e+04</td> <td>-1.07e+04</td>
</tr>
<tr>
  <th>Model_DB9 GT</th>                                                  <td>-3638.7353</td> <td> 4011.196</td> <td>   -0.907</td> <td> 0.364</td> <td>-1.15e+04</td> <td> 4223.994</td>
</tr>
<tr>
  <th>Model_DBS</th>                                                     <td> 7.216e+04</td> <td> 1979.936</td> <td>   36.445</td> <td> 0.000</td> <td> 6.83e+04</td> <td>  7.6e+04</td>
</tr>
<tr>
  <th>Model_DTS</th>                                                     <td> 6246.2727</td> <td> 2049.911</td> <td>    3.047</td> <td> 0.002</td> <td> 2228.046</td> <td> 1.03e+04</td>
</tr>
<tr>
  <th>Model_Dakota</th>                                                  <td>-7758.2896</td> <td> 3802.217</td> <td>   -2.040</td> <td> 0.041</td> <td>-1.52e+04</td> <td> -305.201</td>
</tr>
<tr>
  <th>Model_Dart</th>                                                    <td>-5989.8452</td> <td> 4447.239</td> <td>   -1.347</td> <td> 0.178</td> <td>-1.47e+04</td> <td> 2727.615</td>
</tr>
<tr>
  <th>Model_Dawn</th>                                                    <td>-2.283e+04</td> <td> 6709.398</td> <td>   -3.403</td> <td> 0.001</td> <td> -3.6e+04</td> <td>-9682.055</td>
</tr>
<tr>
  <th>Model_Daytona</th>                                                 <td>-1.076e+04</td> <td> 5258.659</td> <td>   -2.046</td> <td> 0.041</td> <td>-2.11e+04</td> <td> -450.317</td>
</tr>
<tr>
  <th>Model_DeVille</th>                                                 <td> 7899.2486</td> <td> 2370.696</td> <td>    3.332</td> <td> 0.001</td> <td> 3252.221</td> <td> 1.25e+04</td>
</tr>
<tr>
  <th>Model_Defender</th>                                                <td>-1.051e+04</td> <td> 3614.954</td> <td>   -2.908</td> <td> 0.004</td> <td>-1.76e+04</td> <td>-3425.886</td>
</tr>
<tr>
  <th>Model_Diablo</th>                                                  <td>-1.433e+05</td> <td> 6281.066</td> <td>  -22.819</td> <td> 0.000</td> <td>-1.56e+05</td> <td>-1.31e+05</td>
</tr>
<tr>
  <th>Model_Diamante</th>                                                <td> 6455.1585</td> <td> 2559.640</td> <td>    2.522</td> <td> 0.012</td> <td> 1437.762</td> <td> 1.15e+04</td>
</tr>
<tr>
  <th>Model_Discovery</th>                                               <td>-1.502e+04</td> <td> 2844.998</td> <td>   -5.280</td> <td> 0.000</td> <td>-2.06e+04</td> <td>-9443.976</td>
</tr>
<tr>
  <th>Model_Discovery Series II</th>                                     <td>-2.109e+04</td> <td> 2851.318</td> <td>   -7.395</td> <td> 0.000</td> <td>-2.67e+04</td> <td>-1.55e+04</td>
</tr>
<tr>
  <th>Model_Discovery Sport</th>                                         <td>-1.617e+04</td> <td> 2546.141</td> <td>   -6.352</td> <td> 0.000</td> <td>-2.12e+04</td> <td>-1.12e+04</td>
</tr>
<tr>
  <th>Model_Durango</th>                                                 <td>-2589.8977</td> <td> 4379.560</td> <td>   -0.591</td> <td> 0.554</td> <td>-1.12e+04</td> <td> 5994.897</td>
</tr>
<tr>
  <th>Model_Dynasty</th>                                                 <td>-1.138e+04</td> <td> 5583.332</td> <td>   -2.039</td> <td> 0.042</td> <td>-2.23e+04</td> <td> -438.241</td>
</tr>
<tr>
  <th>Model_E-150</th>                                                   <td>-1.652e+04</td> <td> 2565.060</td> <td>   -6.440</td> <td> 0.000</td> <td>-2.15e+04</td> <td>-1.15e+04</td>
</tr>
<tr>
  <th>Model_E-250</th>                                                   <td> -1.77e+04</td> <td> 3764.811</td> <td>   -4.703</td> <td> 0.000</td> <td>-2.51e+04</td> <td>-1.03e+04</td>
</tr>
<tr>
  <th>Model_E-Class</th>                                                 <td>-1.445e+04</td> <td> 1600.698</td> <td>   -9.030</td> <td> 0.000</td> <td>-1.76e+04</td> <td>-1.13e+04</td>
</tr>
<tr>
  <th>Model_E-Series Van</th>                                            <td>-3762.9223</td> <td> 3459.070</td> <td>   -1.088</td> <td> 0.277</td> <td>-1.05e+04</td> <td> 3017.533</td>
</tr>
<tr>
  <th>Model_E-Series Wagon</th>                                          <td>  331.7101</td> <td> 3539.919</td> <td>    0.094</td> <td> 0.925</td> <td>-6607.224</td> <td> 7270.644</td>
</tr>
<tr>
  <th>Model_E55 AMG</th>                                                 <td>-5.546e+04</td> <td> 5043.680</td> <td>  -10.995</td> <td> 0.000</td> <td>-6.53e+04</td> <td>-4.56e+04</td>
</tr>
<tr>
  <th>Model_ECHO</th>                                                    <td>-7482.5604</td> <td> 2198.576</td> <td>   -3.403</td> <td> 0.001</td> <td>-1.18e+04</td> <td>-3172.921</td>
</tr>
<tr>
  <th>Model_ES 250</th>                                                  <td>-2.791e+04</td> <td> 5243.837</td> <td>   -5.323</td> <td> 0.000</td> <td>-3.82e+04</td> <td>-1.76e+04</td>
</tr>
<tr>
  <th>Model_ES 300</th>                                                  <td>-8160.9948</td> <td> 4143.079</td> <td>   -1.970</td> <td> 0.049</td> <td>-1.63e+04</td> <td>  -39.749</td>
</tr>
<tr>
  <th>Model_ES 300h</th>                                                 <td>-4798.6505</td> <td> 4255.719</td> <td>   -1.128</td> <td> 0.260</td> <td>-1.31e+04</td> <td> 3543.393</td>
</tr>
<tr>
  <th>Model_ES 330</th>                                                  <td>-9872.6342</td> <td> 4130.734</td> <td>   -2.390</td> <td> 0.017</td> <td> -1.8e+04</td> <td>-1775.586</td>
</tr>
<tr>
  <th>Model_ES 350</th>                                                  <td>-1.275e+04</td> <td> 3635.742</td> <td>   -3.506</td> <td> 0.000</td> <td>-1.99e+04</td> <td>-5620.774</td>
</tr>
<tr>
  <th>Model_EX</th>                                                      <td>-1.092e+04</td> <td> 2983.006</td> <td>   -3.660</td> <td> 0.000</td> <td>-1.68e+04</td> <td>-5069.887</td>
</tr>
<tr>
  <th>Model_EX35</th>                                                    <td>-1.055e+04</td> <td> 3030.366</td> <td>   -3.480</td> <td> 0.001</td> <td>-1.65e+04</td> <td>-4606.260</td>
</tr>
<tr>
  <th>Model_Eclipse</th>                                                 <td>-2345.9532</td> <td> 2280.931</td> <td>   -1.029</td> <td> 0.304</td> <td>-6817.025</td> <td> 2125.119</td>
</tr>
<tr>
  <th>Model_Eclipse Spyder</th>                                          <td>-4005.2953</td> <td> 2852.147</td> <td>   -1.404</td> <td> 0.160</td> <td>-9596.061</td> <td> 1585.471</td>
</tr>
<tr>
  <th>Model_Edge</th>                                                    <td> 6783.4784</td> <td> 1678.392</td> <td>    4.042</td> <td> 0.000</td> <td> 3493.501</td> <td> 1.01e+04</td>
</tr>
<tr>
  <th>Model_Eighty-Eight</th>                                            <td>-1.617e+04</td> <td> 2936.468</td> <td>   -5.508</td> <td> 0.000</td> <td>-2.19e+04</td> <td>-1.04e+04</td>
</tr>
<tr>
  <th>Model_Eighty-Eight Royale</th>                                     <td>-1.335e+04</td> <td> 3501.397</td> <td>   -3.814</td> <td> 0.000</td> <td>-2.02e+04</td> <td>-6490.251</td>
</tr>
<tr>
  <th>Model_Elantra</th>                                                 <td>-7906.4937</td> <td> 1823.564</td> <td>   -4.336</td> <td> 0.000</td> <td>-1.15e+04</td> <td>-4331.950</td>
</tr>
<tr>
  <th>Model_Elantra Coupe</th>                                           <td>-6295.3732</td> <td> 2591.543</td> <td>   -2.429</td> <td> 0.015</td> <td>-1.14e+04</td> <td>-1215.441</td>
</tr>
<tr>
  <th>Model_Elantra GT</th>                                              <td>-9449.8329</td> <td> 2987.021</td> <td>   -3.164</td> <td> 0.002</td> <td>-1.53e+04</td> <td>-3594.687</td>
</tr>
<tr>
  <th>Model_Elantra Touring</th>                                         <td>-7696.6369</td> <td> 2178.723</td> <td>   -3.533</td> <td> 0.000</td> <td> -1.2e+04</td> <td>-3425.912</td>
</tr>
<tr>
  <th>Model_Eldorado</th>                                                <td>-6665.2227</td> <td> 2837.142</td> <td>   -2.349</td> <td> 0.019</td> <td>-1.22e+04</td> <td>-1103.870</td>
</tr>
<tr>
  <th>Model_Electra</th>                                                 <td>-1.327e+04</td> <td> 6918.733</td> <td>   -1.918</td> <td> 0.055</td> <td>-2.68e+04</td> <td>  288.977</td>
</tr>
<tr>
  <th>Model_Element</th>                                                 <td>-4287.0100</td> <td> 1760.899</td> <td>   -2.435</td> <td> 0.015</td> <td>-7738.718</td> <td> -835.302</td>
</tr>
<tr>
  <th>Model_Elise</th>                                                   <td>-3.272e+04</td> <td> 2758.701</td> <td>  -11.861</td> <td> 0.000</td> <td>-3.81e+04</td> <td>-2.73e+04</td>
</tr>
<tr>
  <th>Model_Enclave</th>                                                 <td> 3583.1762</td> <td> 2172.192</td> <td>    1.650</td> <td> 0.099</td> <td> -674.746</td> <td> 7841.099</td>
</tr>
<tr>
  <th>Model_Encore</th>                                                  <td>-8592.4600</td> <td> 2055.049</td> <td>   -4.181</td> <td> 0.000</td> <td>-1.26e+04</td> <td>-4564.162</td>
</tr>
<tr>
  <th>Model_Endeavor</th>                                                <td>  325.8131</td> <td> 2605.384</td> <td>    0.125</td> <td> 0.900</td> <td>-4781.249</td> <td> 5432.875</td>
</tr>
<tr>
  <th>Model_Entourage</th>                                               <td> 1891.0362</td> <td> 4972.327</td> <td>    0.380</td> <td> 0.704</td> <td>-7855.698</td> <td> 1.16e+04</td>
</tr>
<tr>
  <th>Model_Envision</th>                                                <td> 3204.7407</td> <td> 2751.525</td> <td>    1.165</td> <td> 0.244</td> <td>-2188.788</td> <td> 8598.269</td>
</tr>
<tr>
  <th>Model_Envoy</th>                                                   <td> 2.746e+04</td> <td> 3010.208</td> <td>    9.123</td> <td> 0.000</td> <td> 2.16e+04</td> <td> 3.34e+04</td>
</tr>
<tr>
  <th>Model_Envoy XL</th>                                                <td> 2.905e+04</td> <td> 3052.913</td> <td>    9.517</td> <td> 0.000</td> <td> 2.31e+04</td> <td>  3.5e+04</td>
</tr>
<tr>
  <th>Model_Envoy XUV</th>                                               <td> 2.942e+04</td> <td> 4316.887</td> <td>    6.815</td> <td> 0.000</td> <td>  2.1e+04</td> <td> 3.79e+04</td>
</tr>
<tr>
  <th>Model_Enzo</th>                                                    <td> 3.508e+05</td> <td> 6796.271</td> <td>   51.616</td> <td> 0.000</td> <td> 3.37e+05</td> <td> 3.64e+05</td>
</tr>
<tr>
  <th>Model_Eos</th>                                                     <td>-8147.9780</td> <td> 6229.597</td> <td>   -1.308</td> <td> 0.191</td> <td>-2.04e+04</td> <td> 4063.251</td>
</tr>
<tr>
  <th>Model_Equator</th>                                                 <td>-1.036e+04</td> <td> 2282.731</td> <td>   -4.540</td> <td> 0.000</td> <td>-1.48e+04</td> <td>-5888.426</td>
</tr>
<tr>
  <th>Model_Equinox</th>                                                 <td>-1.094e+04</td> <td> 2054.672</td> <td>   -5.322</td> <td> 0.000</td> <td> -1.5e+04</td> <td>-6907.646</td>
</tr>
<tr>
  <th>Model_Equus</th>                                                   <td>  1.05e+04</td> <td> 3082.777</td> <td>    3.407</td> <td> 0.001</td> <td> 4459.502</td> <td> 1.65e+04</td>
</tr>
<tr>
  <th>Model_Escalade</th>                                                <td> 1.275e+04</td> <td> 1573.645</td> <td>    8.102</td> <td> 0.000</td> <td> 9665.202</td> <td> 1.58e+04</td>
</tr>
<tr>
  <th>Model_Escalade ESV</th>                                            <td> 1.574e+04</td> <td> 1573.292</td> <td>   10.005</td> <td> 0.000</td> <td> 1.27e+04</td> <td> 1.88e+04</td>
</tr>
<tr>
  <th>Model_Escalade EXT</th>                                            <td>-4156.0943</td> <td> 3649.218</td> <td>   -1.139</td> <td> 0.255</td> <td>-1.13e+04</td> <td> 2997.088</td>
</tr>
<tr>
  <th>Model_Escalade Hybrid</th>                                         <td> 2.698e+04</td> <td> 2201.400</td> <td>   12.256</td> <td> 0.000</td> <td> 2.27e+04</td> <td> 3.13e+04</td>
</tr>
<tr>
  <th>Model_Escape</th>                                                  <td> 2596.3741</td> <td> 2322.289</td> <td>    1.118</td> <td> 0.264</td> <td>-1955.768</td> <td> 7148.516</td>
</tr>
<tr>
  <th>Model_Escape Hybrid</th>                                           <td> 1.281e+04</td> <td> 2343.399</td> <td>    5.468</td> <td> 0.000</td> <td> 8220.074</td> <td> 1.74e+04</td>
</tr>
<tr>
  <th>Model_Escape S</th>                                                <td> 2928.3231</td> <td> 5033.343</td> <td>    0.582</td> <td> 0.561</td> <td>-6938.016</td> <td> 1.28e+04</td>
</tr>
<tr>
  <th>Model_Escape SE</th>                                               <td> 2760.5117</td> <td> 5031.701</td> <td>    0.549</td> <td> 0.583</td> <td>-7102.607</td> <td> 1.26e+04</td>
</tr>
<tr>
  <th>Model_Escort</th>                                                  <td> 3561.9762</td> <td> 2613.767</td> <td>    1.363</td> <td> 0.173</td> <td>-1561.518</td> <td> 8685.470</td>
</tr>
<tr>
  <th>Model_Esprit</th>                                                  <td> 6102.7552</td> <td> 3746.255</td> <td>    1.629</td> <td> 0.103</td> <td>-1240.637</td> <td> 1.34e+04</td>
</tr>
<tr>
  <th>Model_Estate Wagon</th>                                            <td>-1.817e+04</td> <td> 6989.214</td> <td>   -2.599</td> <td> 0.009</td> <td>-3.19e+04</td> <td>-4467.168</td>
</tr>
<tr>
  <th>Model_Esteem</th>                                                  <td>-3854.1047</td> <td> 1401.264</td> <td>   -2.750</td> <td> 0.006</td> <td>-6600.856</td> <td>-1107.354</td>
</tr>
<tr>
  <th>Model_EuroVan</th>                                                 <td> -541.6895</td> <td> 7177.792</td> <td>   -0.075</td> <td> 0.940</td> <td>-1.46e+04</td> <td> 1.35e+04</td>
</tr>
<tr>
  <th>Model_Evora</th>                                                   <td>-1.418e+04</td> <td> 2651.137</td> <td>   -5.348</td> <td> 0.000</td> <td>-1.94e+04</td> <td>-8980.507</td>
</tr>
<tr>
  <th>Model_Evora 400</th>                                               <td>-7202.0651</td> <td> 6203.034</td> <td>   -1.161</td> <td> 0.246</td> <td>-1.94e+04</td> <td> 4957.096</td>
</tr>
<tr>
  <th>Model_Excel</th>                                                   <td>-1.176e+04</td> <td> 2648.187</td> <td>   -4.439</td> <td> 0.000</td> <td>-1.69e+04</td> <td>-6564.632</td>
</tr>
<tr>
  <th>Model_Exige</th>                                                   <td>-8331.0739</td> <td> 2994.804</td> <td>   -2.782</td> <td> 0.005</td> <td>-1.42e+04</td> <td>-2460.671</td>
</tr>
<tr>
  <th>Model_Expedition</th>                                              <td> 1.753e+04</td> <td> 1560.147</td> <td>   11.236</td> <td> 0.000</td> <td> 1.45e+04</td> <td> 2.06e+04</td>
</tr>
<tr>
  <th>Model_Explorer</th>                                                <td> 7186.2483</td> <td> 1642.262</td> <td>    4.376</td> <td> 0.000</td> <td> 3967.093</td> <td> 1.04e+04</td>
</tr>
<tr>
  <th>Model_Explorer Sport</th>                                          <td> 1.046e+04</td> <td> 2505.840</td> <td>    4.173</td> <td> 0.000</td> <td> 5545.558</td> <td> 1.54e+04</td>
</tr>
<tr>
  <th>Model_Explorer Sport Trac</th>                                     <td> -698.5569</td> <td> 1809.072</td> <td>   -0.386</td> <td> 0.699</td> <td>-4244.693</td> <td> 2847.579</td>
</tr>
<tr>
  <th>Model_Expo</th>                                                    <td>-1.045e+04</td> <td> 2328.279</td> <td>   -4.488</td> <td> 0.000</td> <td> -1.5e+04</td> <td>-5884.693</td>
</tr>
<tr>
  <th>Model_Express</th>                                                 <td>-1.721e+04</td> <td> 3740.525</td> <td>   -4.602</td> <td> 0.000</td> <td>-2.45e+04</td> <td>-9882.371</td>
</tr>
<tr>
  <th>Model_Express Cargo</th>                                           <td>-1.975e+04</td> <td> 5737.856</td> <td>   -3.441</td> <td> 0.001</td> <td> -3.1e+04</td> <td>-8499.255</td>
</tr>
<tr>
  <th>Model_F-150</th>                                                   <td> -352.4856</td> <td> 1560.728</td> <td>   -0.226</td> <td> 0.821</td> <td>-3411.819</td> <td> 2706.847</td>
</tr>
<tr>
  <th>Model_F-150 Heritage</th>                                          <td> 2366.0724</td> <td> 1780.245</td> <td>    1.329</td> <td> 0.184</td> <td>-1123.556</td> <td> 5855.700</td>
</tr>
<tr>
  <th>Model_F-150 SVT Lightning</th>                                     <td>-1.978e+04</td> <td> 4107.582</td> <td>   -4.816</td> <td> 0.000</td> <td>-2.78e+04</td> <td>-1.17e+04</td>
</tr>
<tr>
  <th>Model_F-250</th>                                                   <td>-1.559e+04</td> <td> 1380.408</td> <td>  -11.290</td> <td> 0.000</td> <td>-1.83e+04</td> <td>-1.29e+04</td>
</tr>
<tr>
  <th>Model_F12 Berlinetta</th>                                          <td>-7410.5426</td> <td> 4183.174</td> <td>   -1.772</td> <td> 0.077</td> <td>-1.56e+04</td> <td>  789.297</td>
</tr>
<tr>
  <th>Model_F430</th>                                                    <td>-4.023e+04</td> <td> 2213.489</td> <td>  -18.177</td> <td> 0.000</td> <td>-4.46e+04</td> <td>-3.59e+04</td>
</tr>
<tr>
  <th>Model_FF</th>                                                      <td>-6825.8426</td> <td> 4161.237</td> <td>   -1.640</td> <td> 0.101</td> <td> -1.5e+04</td> <td> 1330.997</td>
</tr>
<tr>
  <th>Model_FJ Cruiser</th>                                              <td>-3781.4535</td> <td> 2422.343</td> <td>   -1.561</td> <td> 0.119</td> <td>-8529.720</td> <td>  966.813</td>
</tr>
<tr>
  <th>Model_FR-S</th>                                                    <td>-5485.9244</td> <td> 2440.716</td> <td>   -2.248</td> <td> 0.025</td> <td>-1.03e+04</td> <td> -701.642</td>
</tr>
<tr>
  <th>Model_FX</th>                                                      <td>-2478.2846</td> <td> 3032.563</td> <td>   -0.817</td> <td> 0.414</td> <td>-8422.702</td> <td> 3466.133</td>
</tr>
<tr>
  <th>Model_FX35</th>                                                    <td>-3498.8299</td> <td> 3637.542</td> <td>   -0.962</td> <td> 0.336</td> <td>-1.06e+04</td> <td> 3631.465</td>
</tr>
<tr>
  <th>Model_FX45</th>                                                    <td>   97.2683</td> <td> 4670.082</td> <td>    0.021</td> <td> 0.983</td> <td>-9057.006</td> <td> 9251.543</td>
</tr>
<tr>
  <th>Model_FX50</th>                                                    <td> -910.7358</td> <td> 5447.786</td> <td>   -0.167</td> <td> 0.867</td> <td>-1.16e+04</td> <td> 9767.992</td>
</tr>
<tr>
  <th>Model_Festiva</th>                                                 <td>-5513.6229</td> <td> 4514.340</td> <td>   -1.221</td> <td> 0.222</td> <td>-1.44e+04</td> <td> 3335.367</td>
</tr>
<tr>
  <th>Model_Fiesta</th>                                                  <td>-4229.1341</td> <td> 1893.630</td> <td>   -2.233</td> <td> 0.026</td> <td>-7941.019</td> <td> -517.249</td>
</tr>
<tr>
  <th>Model_Firebird</th>                                                <td>-9031.3201</td> <td> 2152.470</td> <td>   -4.196</td> <td> 0.000</td> <td>-1.33e+04</td> <td>-4812.058</td>
</tr>
<tr>
  <th>Model_Fit</th>                                                     <td>-1.146e+04</td> <td> 1919.155</td> <td>   -5.970</td> <td> 0.000</td> <td>-1.52e+04</td> <td>-7694.529</td>
</tr>
<tr>
  <th>Model_Fit EV</th>                                                  <td> 2285.4278</td> <td> 6610.322</td> <td>    0.346</td> <td> 0.730</td> <td>-1.07e+04</td> <td> 1.52e+04</td>
</tr>
<tr>
  <th>Model_Five Hundred</th>                                            <td> 6948.4731</td> <td> 2058.712</td> <td>    3.375</td> <td> 0.001</td> <td> 2912.994</td> <td>  1.1e+04</td>
</tr>
<tr>
  <th>Model_Fleetwood</th>                                               <td>-2.982e+04</td> <td> 4212.156</td> <td>   -7.079</td> <td> 0.000</td> <td>-3.81e+04</td> <td>-2.16e+04</td>
</tr>
<tr>
  <th>Model_Flex</th>                                                    <td> 4699.1277</td> <td> 2069.082</td> <td>    2.271</td> <td> 0.023</td> <td>  643.321</td> <td> 8754.935</td>
</tr>
<tr>
  <th>Model_Flying Spur</th>                                             <td>-1.059e+05</td> <td> 3919.575</td> <td>  -27.012</td> <td> 0.000</td> <td>-1.14e+05</td> <td>-9.82e+04</td>
</tr>
<tr>
  <th>Model_Focus</th>                                                   <td>-1363.4022</td> <td> 2047.707</td> <td>   -0.666</td> <td> 0.506</td> <td>-5377.310</td> <td> 2650.505</td>
</tr>
<tr>
  <th>Model_Focus RS</th>                                                <td> 2537.1329</td> <td> 7143.701</td> <td>    0.355</td> <td> 0.722</td> <td>-1.15e+04</td> <td> 1.65e+04</td>
</tr>
<tr>
  <th>Model_Focus ST</th>                                                <td> 1493.9100</td> <td> 4238.342</td> <td>    0.352</td> <td> 0.724</td> <td>-6814.071</td> <td> 9801.891</td>
</tr>
<tr>
  <th>Model_Forenza</th>                                                 <td>-6333.1441</td> <td> 1457.589</td> <td>   -4.345</td> <td> 0.000</td> <td>-9190.305</td> <td>-3475.984</td>
</tr>
<tr>
  <th>Model_Forester</th>                                                <td>-6057.2423</td> <td> 1670.430</td> <td>   -3.626</td> <td> 0.000</td> <td>-9331.613</td> <td>-2782.872</td>
</tr>
<tr>
  <th>Model_Forte</th>                                                   <td>-9381.3041</td> <td> 1600.719</td> <td>   -5.861</td> <td> 0.000</td> <td>-1.25e+04</td> <td>-6243.581</td>
</tr>
<tr>
  <th>Model_Fox</th>                                                     <td>-1.623e+04</td> <td> 5984.031</td> <td>   -2.712</td> <td> 0.007</td> <td> -2.8e+04</td> <td>-4499.575</td>
</tr>
<tr>
  <th>Model_Freelander</th>                                              <td>-2.222e+04</td> <td> 2409.406</td> <td>   -9.222</td> <td> 0.000</td> <td>-2.69e+04</td> <td>-1.75e+04</td>
</tr>
<tr>
  <th>Model_Freestar</th>                                                <td> 1.141e+04</td> <td> 4141.683</td> <td>    2.755</td> <td> 0.006</td> <td> 3289.774</td> <td> 1.95e+04</td>
</tr>
<tr>
  <th>Model_Freestyle</th>                                               <td> 8772.4949</td> <td> 2138.669</td> <td>    4.102</td> <td> 0.000</td> <td> 4580.285</td> <td>  1.3e+04</td>
</tr>
<tr>
  <th>Model_Frontier</th>                                                <td>-6303.3976</td> <td> 2513.788</td> <td>   -2.508</td> <td> 0.012</td> <td>-1.12e+04</td> <td>-1375.881</td>
</tr>
<tr>
  <th>Model_Fusion</th>                                                  <td> 7236.2215</td> <td> 1983.741</td> <td>    3.648</td> <td> 0.000</td> <td> 3347.700</td> <td> 1.11e+04</td>
</tr>
<tr>
  <th>Model_Fusion Hybrid</th>                                           <td>  1.01e+04</td> <td> 2732.584</td> <td>    3.698</td> <td> 0.000</td> <td> 4747.657</td> <td> 1.55e+04</td>
</tr>
<tr>
  <th>Model_G Convertible</th>                                           <td>-1916.5040</td> <td> 3399.917</td> <td>   -0.564</td> <td> 0.573</td> <td>-8581.007</td> <td> 4747.999</td>
</tr>
<tr>
  <th>Model_G Coupe</th>                                                 <td>-2632.5733</td> <td> 2875.305</td> <td>   -0.916</td> <td> 0.360</td> <td>-8268.734</td> <td> 3003.587</td>
</tr>
<tr>
  <th>Model_G Sedan</th>                                                 <td>-6895.6434</td> <td> 2662.588</td> <td>   -2.590</td> <td> 0.010</td> <td>-1.21e+04</td> <td>-1676.449</td>
</tr>
<tr>
  <th>Model_G-Class</th>                                                 <td> 3.737e+04</td> <td> 3257.596</td> <td>   11.472</td> <td> 0.000</td> <td>  3.1e+04</td> <td> 4.38e+04</td>
</tr>
<tr>
  <th>Model_G20</th>                                                     <td>-1.094e+04</td> <td> 3640.658</td> <td>   -3.005</td> <td> 0.003</td> <td>-1.81e+04</td> <td>-3804.527</td>
</tr>
<tr>
  <th>Model_G3</th>                                                      <td>-1.238e+04</td> <td> 6971.974</td> <td>   -1.776</td> <td> 0.076</td> <td> -2.6e+04</td> <td> 1286.792</td>
</tr>
<tr>
  <th>Model_G35</th>                                                     <td>-6645.3788</td> <td> 2891.747</td> <td>   -2.298</td> <td> 0.022</td> <td>-1.23e+04</td> <td> -976.988</td>
</tr>
<tr>
  <th>Model_G37</th>                                                     <td>-7952.2174</td> <td> 3011.628</td> <td>   -2.641</td> <td> 0.008</td> <td>-1.39e+04</td> <td>-2048.837</td>
</tr>
<tr>
  <th>Model_G37 Convertible</th>                                         <td>-3103.3646</td> <td> 4694.922</td> <td>   -0.661</td> <td> 0.509</td> <td>-1.23e+04</td> <td> 6099.602</td>
</tr>
<tr>
  <th>Model_G37 Coupe</th>                                               <td>-4505.0531</td> <td> 3921.478</td> <td>   -1.149</td> <td> 0.251</td> <td>-1.22e+04</td> <td> 3181.812</td>
</tr>
<tr>
  <th>Model_G37 Sedan</th>                                               <td>-7618.0758</td> <td> 3657.130</td> <td>   -2.083</td> <td> 0.037</td> <td>-1.48e+04</td> <td> -449.384</td>
</tr>
<tr>
  <th>Model_G5</th>                                                      <td>-5050.4474</td> <td> 3086.627</td> <td>   -1.636</td> <td> 0.102</td> <td>-1.11e+04</td> <td>  999.946</td>
</tr>
<tr>
  <th>Model_G6</th>                                                      <td>-2760.3215</td> <td> 1944.223</td> <td>   -1.420</td> <td> 0.156</td> <td>-6571.380</td> <td> 1050.736</td>
</tr>
<tr>
  <th>Model_G8</th>                                                      <td>-7319.2788</td> <td> 3381.497</td> <td>   -2.165</td> <td> 0.030</td> <td>-1.39e+04</td> <td> -690.882</td>
</tr>
<tr>
  <th>Model_G80</th>                                                     <td>-6.559e+04</td> <td> 4125.760</td> <td>  -15.898</td> <td> 0.000</td> <td>-7.37e+04</td> <td>-5.75e+04</td>
</tr>
<tr>
  <th>Model_GL-Class</th>                                                <td>-6939.1568</td> <td> 2281.748</td> <td>   -3.041</td> <td> 0.002</td> <td>-1.14e+04</td> <td>-2466.484</td>
</tr>
<tr>
  <th>Model_GLA-Class</th>                                               <td>-3.341e+04</td> <td> 2640.891</td> <td>  -12.652</td> <td> 0.000</td> <td>-3.86e+04</td> <td>-2.82e+04</td>
</tr>
<tr>
  <th>Model_GLC-Class</th>                                               <td>-2.594e+04</td> <td> 3616.879</td> <td>   -7.171</td> <td> 0.000</td> <td> -3.3e+04</td> <td>-1.88e+04</td>
</tr>
<tr>
  <th>Model_GLE-Class</th>                                               <td>-1.608e+04</td> <td> 2299.582</td> <td>   -6.994</td> <td> 0.000</td> <td>-2.06e+04</td> <td>-1.16e+04</td>
</tr>
<tr>
  <th>Model_GLE-Class Coupe</th>                                         <td>-1.267e+04</td> <td> 3611.975</td> <td>   -3.509</td> <td> 0.000</td> <td>-1.98e+04</td> <td>-5594.667</td>
</tr>
<tr>
  <th>Model_GLI</th>                                                     <td>-6348.7730</td> <td> 5915.289</td> <td>   -1.073</td> <td> 0.283</td> <td>-1.79e+04</td> <td> 5246.352</td>
</tr>
<tr>
  <th>Model_GLK-Class</th>                                               <td>-3.057e+04</td> <td> 2478.874</td> <td>  -12.334</td> <td> 0.000</td> <td>-3.54e+04</td> <td>-2.57e+04</td>
</tr>
<tr>
  <th>Model_GLS-Class</th>                                               <td>-5990.8696</td> <td> 4178.866</td> <td>   -1.434</td> <td> 0.152</td> <td>-1.42e+04</td> <td> 2200.526</td>
</tr>
<tr>
  <th>Model_GS 200t</th>                                                 <td> 1374.5798</td> <td> 3646.163</td> <td>    0.377</td> <td> 0.706</td> <td>-5772.614</td> <td> 8521.774</td>
</tr>
<tr>
  <th>Model_GS 300</th>                                                  <td>-1674.9146</td> <td> 3579.352</td> <td>   -0.468</td> <td> 0.640</td> <td>-8691.145</td> <td> 5341.316</td>
</tr>
<tr>
  <th>Model_GS 350</th>                                                  <td>-3325.2877</td> <td> 2311.111</td> <td>   -1.439</td> <td> 0.150</td> <td>-7855.518</td> <td> 1204.942</td>
</tr>
<tr>
  <th>Model_GS 400</th>                                                  <td>-3.942e+04</td> <td> 4183.473</td> <td>   -9.422</td> <td> 0.000</td> <td>-4.76e+04</td> <td>-3.12e+04</td>
</tr>
<tr>
  <th>Model_GS 430</th>                                                  <td> 1683.5586</td> <td> 4112.592</td> <td>    0.409</td> <td> 0.682</td> <td>-6377.926</td> <td> 9745.044</td>
</tr>
<tr>
  <th>Model_GS 450h</th>                                                 <td> 5607.3364</td> <td> 4175.286</td> <td>    1.343</td> <td> 0.179</td> <td>-2577.042</td> <td> 1.38e+04</td>
</tr>
<tr>
  <th>Model_GS 460</th>                                                  <td>-1581.6376</td> <td> 4109.589</td> <td>   -0.385</td> <td> 0.700</td> <td>-9637.238</td> <td> 6473.963</td>
</tr>
<tr>
  <th>Model_GS F</th>                                                    <td> 5473.4215</td> <td> 7070.125</td> <td>    0.774</td> <td> 0.439</td> <td>-8385.407</td> <td> 1.93e+04</td>
</tr>
<tr>
  <th>Model_GT</th>                                                      <td> 8.848e+04</td> <td> 5140.179</td> <td>   17.214</td> <td> 0.000</td> <td> 7.84e+04</td> <td> 9.86e+04</td>
</tr>
<tr>
  <th>Model_GT-R</th>                                                    <td> 4.061e+04</td> <td> 3240.461</td> <td>   12.531</td> <td> 0.000</td> <td> 3.43e+04</td> <td>  4.7e+04</td>
</tr>
<tr>
  <th>Model_GTI</th>                                                     <td>-7339.7977</td> <td> 5655.325</td> <td>   -1.298</td> <td> 0.194</td> <td>-1.84e+04</td> <td> 3745.746</td>
</tr>
<tr>
  <th>Model_GTO</th>                                                     <td>-8330.6239</td> <td> 4167.056</td> <td>   -1.999</td> <td> 0.046</td> <td>-1.65e+04</td> <td> -162.378</td>
</tr>
<tr>
  <th>Model_GX 460</th>                                                  <td>-5663.9611</td> <td> 2989.359</td> <td>   -1.895</td> <td> 0.058</td> <td>-1.15e+04</td> <td>  195.769</td>
</tr>
<tr>
  <th>Model_GX 470</th>                                                  <td>-6165.4598</td> <td> 4108.080</td> <td>   -1.501</td> <td> 0.133</td> <td>-1.42e+04</td> <td> 1887.182</td>
</tr>
<tr>
  <th>Model_Galant</th>                                                  <td>  -71.3072</td> <td> 3026.543</td> <td>   -0.024</td> <td> 0.981</td> <td>-6003.925</td> <td> 5861.310</td>
</tr>
<tr>
  <th>Model_Gallardo</th>                                                <td>-2.327e+05</td> <td> 2173.279</td> <td> -107.060</td> <td> 0.000</td> <td>-2.37e+05</td> <td>-2.28e+05</td>
</tr>
<tr>
  <th>Model_Genesis</th>                                                 <td> -375.5624</td> <td> 2706.801</td> <td>   -0.139</td> <td> 0.890</td> <td>-5681.423</td> <td> 4930.298</td>
</tr>
<tr>
  <th>Model_Genesis Coupe</th>                                           <td>-5899.4833</td> <td> 2037.314</td> <td>   -2.896</td> <td> 0.004</td> <td>-9893.018</td> <td>-1905.949</td>
</tr>
<tr>
  <th>Model_Ghibli</th>                                                  <td>-2.551e+04</td> <td> 6005.249</td> <td>   -4.248</td> <td> 0.000</td> <td>-3.73e+04</td> <td>-1.37e+04</td>
</tr>
<tr>
  <th>Model_Ghost</th>                                                   <td>  -7.1e+04</td> <td> 3107.976</td> <td>  -22.846</td> <td> 0.000</td> <td>-7.71e+04</td> <td>-6.49e+04</td>
</tr>
<tr>
  <th>Model_Ghost Series II</th>                                         <td>-4.156e+04</td> <td> 3672.688</td> <td>  -11.315</td> <td> 0.000</td> <td>-4.88e+04</td> <td>-3.44e+04</td>
</tr>
<tr>
  <th>Model_Golf</th>                                                    <td>-1.312e+04</td> <td> 5787.441</td> <td>   -2.266</td> <td> 0.023</td> <td>-2.45e+04</td> <td>-1772.193</td>
</tr>
<tr>
  <th>Model_Golf Alltrack</th>                                           <td>-1.003e+04</td> <td> 6492.352</td> <td>   -1.545</td> <td> 0.122</td> <td>-2.28e+04</td> <td> 2696.514</td>
</tr>
<tr>
  <th>Model_Golf GTI</th>                                                <td>-8953.4626</td> <td> 5717.602</td> <td>   -1.566</td> <td> 0.117</td> <td>-2.02e+04</td> <td> 2254.156</td>
</tr>
<tr>
  <th>Model_Golf R</th>                                                  <td>-6399.0442</td> <td> 6090.654</td> <td>   -1.051</td> <td> 0.293</td> <td>-1.83e+04</td> <td> 5539.830</td>
</tr>
<tr>
  <th>Model_Golf SportWagen</th>                                         <td>-9574.5266</td> <td> 5862.176</td> <td>   -1.633</td> <td> 0.102</td> <td>-2.11e+04</td> <td> 1916.486</td>
</tr>
<tr>
  <th>Model_GranSport</th>                                               <td>   52.4676</td> <td> 6464.070</td> <td>    0.008</td> <td> 0.994</td> <td>-1.26e+04</td> <td> 1.27e+04</td>
</tr>
<tr>
  <th>Model_GranTurismo</th>                                             <td> 3.155e+04</td> <td> 6003.930</td> <td>    5.254</td> <td> 0.000</td> <td> 1.98e+04</td> <td> 4.33e+04</td>
</tr>
<tr>
  <th>Model_GranTurismo Convertible</th>                                 <td> 3.573e+04</td> <td> 5858.816</td> <td>    6.099</td> <td> 0.000</td> <td> 2.42e+04</td> <td> 4.72e+04</td>
</tr>
<tr>
  <th>Model_Grand Am</th>                                                <td>-1011.3114</td> <td> 1977.461</td> <td>   -0.511</td> <td> 0.609</td> <td>-4887.522</td> <td> 2864.899</td>
</tr>
<tr>
  <th>Model_Grand Caravan</th>                                           <td>-5622.4215</td> <td> 5970.301</td> <td>   -0.942</td> <td> 0.346</td> <td>-1.73e+04</td> <td> 6080.537</td>
</tr>
<tr>
  <th>Model_Grand Prix</th>                                              <td>-4274.0167</td> <td> 2689.190</td> <td>   -1.589</td> <td> 0.112</td> <td>-9545.355</td> <td>  997.322</td>
</tr>
<tr>
  <th>Model_Grand Vitara</th>                                            <td>-4928.6849</td> <td> 1878.525</td> <td>   -2.624</td> <td> 0.009</td> <td>-8610.962</td> <td>-1246.407</td>
</tr>
<tr>
  <th>Model_Grand Voyager</th>                                           <td> -1.53e+04</td> <td> 6204.352</td> <td>   -2.467</td> <td> 0.014</td> <td>-2.75e+04</td> <td>-3142.967</td>
</tr>
<tr>
  <th>Model_H3</th>                                                      <td>-3.706e+04</td> <td> 2446.062</td> <td>  -15.150</td> <td> 0.000</td> <td>-4.19e+04</td> <td>-3.23e+04</td>
</tr>
<tr>
  <th>Model_H3T</th>                                                     <td>-4.848e+04</td> <td> 3383.000</td> <td>  -14.332</td> <td> 0.000</td> <td>-5.51e+04</td> <td>-4.19e+04</td>
</tr>
<tr>
  <th>Model_HHR</th>                                                     <td>-9131.8275</td> <td> 2404.014</td> <td>   -3.799</td> <td> 0.000</td> <td>-1.38e+04</td> <td>-4419.490</td>
</tr>
<tr>
  <th>Model_HR-V</th>                                                    <td>-9876.8469</td> <td> 1932.098</td> <td>   -5.112</td> <td> 0.000</td> <td>-1.37e+04</td> <td>-6089.555</td>
</tr>
<tr>
  <th>Model_HS 250h</th>                                                 <td>-5107.1942</td> <td> 3039.285</td> <td>   -1.680</td> <td> 0.093</td> <td>-1.11e+04</td> <td>  850.400</td>
</tr>
<tr>
  <th>Model_Highlander</th>                                              <td> 3244.9610</td> <td> 1622.560</td> <td>    2.000</td> <td> 0.046</td> <td>   64.425</td> <td> 6425.497</td>
</tr>
<tr>
  <th>Model_Highlander Hybrid</th>                                       <td> 1.378e+04</td> <td> 2925.260</td> <td>    4.709</td> <td> 0.000</td> <td> 8040.955</td> <td> 1.95e+04</td>
</tr>
<tr>
  <th>Model_Horizon</th>                                                 <td>-5536.0600</td> <td> 8359.220</td> <td>   -0.662</td> <td> 0.508</td> <td>-2.19e+04</td> <td> 1.08e+04</td>
</tr>
<tr>
  <th>Model_Huracan</th>                                                 <td> -2.33e+05</td> <td> 3562.137</td> <td>  -65.409</td> <td> 0.000</td> <td> -2.4e+05</td> <td>-2.26e+05</td>
</tr>
<tr>
  <th>Model_I30</th>                                                     <td>-1.885e+04</td> <td> 3824.607</td> <td>   -4.928</td> <td> 0.000</td> <td>-2.63e+04</td> <td>-1.13e+04</td>
</tr>
<tr>
  <th>Model_I35</th>                                                     <td>-5280.7397</td> <td> 4731.279</td> <td>   -1.116</td> <td> 0.264</td> <td>-1.46e+04</td> <td> 3993.495</td>
</tr>
<tr>
  <th>Model_ILX</th>                                                     <td>-1.447e+04</td> <td> 2161.744</td> <td>   -6.695</td> <td> 0.000</td> <td>-1.87e+04</td> <td>-1.02e+04</td>
</tr>
<tr>
  <th>Model_ILX Hybrid</th>                                              <td>-9875.2888</td> <td> 5031.747</td> <td>   -1.963</td> <td> 0.050</td> <td>-1.97e+04</td> <td>  -12.079</td>
</tr>
<tr>
  <th>Model_IS 200t</th>                                                 <td>-1.081e+04</td> <td> 5052.529</td> <td>   -2.140</td> <td> 0.032</td> <td>-2.07e+04</td> <td> -906.467</td>
</tr>
<tr>
  <th>Model_IS 250</th>                                                  <td>-1.117e+04</td> <td> 2634.811</td> <td>   -4.240</td> <td> 0.000</td> <td>-1.63e+04</td> <td>-6005.766</td>
</tr>
<tr>
  <th>Model_IS 250 C</th>                                                <td>-1.359e+04</td> <td> 4201.047</td> <td>   -3.234</td> <td> 0.001</td> <td>-2.18e+04</td> <td>-5352.556</td>
</tr>
<tr>
  <th>Model_IS 300</th>                                                  <td>-1.257e+04</td> <td> 3230.617</td> <td>   -3.891</td> <td> 0.000</td> <td>-1.89e+04</td> <td>-6238.825</td>
</tr>
<tr>
  <th>Model_IS 350</th>                                                  <td>-1.379e+04</td> <td> 2992.562</td> <td>   -4.608</td> <td> 0.000</td> <td>-1.97e+04</td> <td>-7922.987</td>
</tr>
<tr>
  <th>Model_IS 350 C</th>                                                <td>-1.418e+04</td> <td> 4190.285</td> <td>   -3.383</td> <td> 0.001</td> <td>-2.24e+04</td> <td>-5963.244</td>
</tr>
<tr>
  <th>Model_IS F</th>                                                    <td>-5664.0569</td> <td> 4129.355</td> <td>   -1.372</td> <td> 0.170</td> <td>-1.38e+04</td> <td> 2430.287</td>
</tr>
<tr>
  <th>Model_Impala</th>                                                  <td>-7987.9424</td> <td> 2608.122</td> <td>   -3.063</td> <td> 0.002</td> <td>-1.31e+04</td> <td>-2875.513</td>
</tr>
<tr>
  <th>Model_Impala Limited</th>                                          <td>-1.418e+04</td> <td> 2731.223</td> <td>   -5.193</td> <td> 0.000</td> <td>-1.95e+04</td> <td>-8828.712</td>
</tr>
<tr>
  <th>Model_Imperial</th>                                                <td>-2.059e+04</td> <td> 5713.429</td> <td>   -3.604</td> <td> 0.000</td> <td>-3.18e+04</td> <td>-9394.259</td>
</tr>
<tr>
  <th>Model_Impreza</th>                                                 <td>-1.029e+04</td> <td> 1637.066</td> <td>   -6.285</td> <td> 0.000</td> <td>-1.35e+04</td> <td>-7080.009</td>
</tr>
<tr>
  <th>Model_Impreza WRX</th>                                             <td>-1504.8215</td> <td> 1946.392</td> <td>   -0.773</td> <td> 0.439</td> <td>-5320.131</td> <td> 2310.488</td>
</tr>
<tr>
  <th>Model_Insight</th>                                                 <td>-7854.1296</td> <td> 2309.717</td> <td>   -3.400</td> <td> 0.001</td> <td>-1.24e+04</td> <td>-3326.633</td>
</tr>
<tr>
  <th>Model_Integra</th>                                                 <td>-1.773e+04</td> <td> 1685.356</td> <td>  -10.518</td> <td> 0.000</td> <td> -2.1e+04</td> <td>-1.44e+04</td>
</tr>
<tr>
  <th>Model_Intrepid</th>                                                <td> -132.2169</td> <td> 5109.305</td> <td>   -0.026</td> <td> 0.979</td> <td>-1.01e+04</td> <td> 9883.021</td>
</tr>
<tr>
  <th>Model_Intrigue</th>                                                <td>-1114.0959</td> <td> 2522.393</td> <td>   -0.442</td> <td> 0.659</td> <td>-6058.481</td> <td> 3830.289</td>
</tr>
<tr>
  <th>Model_J30</th>                                                     <td>-2.519e+04</td> <td> 4136.317</td> <td>   -6.091</td> <td> 0.000</td> <td>-3.33e+04</td> <td>-1.71e+04</td>
</tr>
<tr>
  <th>Model_JX</th>                                                      <td>-6960.7729</td> <td> 5410.745</td> <td>   -1.286</td> <td> 0.198</td> <td>-1.76e+04</td> <td> 3645.347</td>
</tr>
<tr>
  <th>Model_Jetta</th>                                                   <td>-1.073e+04</td> <td> 5701.019</td> <td>   -1.882</td> <td> 0.060</td> <td>-2.19e+04</td> <td>  444.510</td>
</tr>
<tr>
  <th>Model_Jetta GLI</th>                                               <td>-6656.5437</td> <td> 5754.146</td> <td>   -1.157</td> <td> 0.247</td> <td>-1.79e+04</td> <td> 4622.709</td>
</tr>
<tr>
  <th>Model_Jetta Hybrid</th>                                            <td>-6747.9146</td> <td> 6219.758</td> <td>   -1.085</td> <td> 0.278</td> <td>-1.89e+04</td> <td> 5444.029</td>
</tr>
<tr>
  <th>Model_Jetta SportWagen</th>                                        <td>-9991.3322</td> <td> 5717.422</td> <td>   -1.748</td> <td> 0.081</td> <td>-2.12e+04</td> <td> 1215.934</td>
</tr>
<tr>
  <th>Model_Jimmy</th>                                                   <td> 1.694e+04</td> <td> 2730.691</td> <td>    6.202</td> <td> 0.000</td> <td> 1.16e+04</td> <td> 2.23e+04</td>
</tr>
<tr>
  <th>Model_Journey</th>                                                 <td>-7462.3776</td> <td> 4296.148</td> <td>   -1.737</td> <td> 0.082</td> <td>-1.59e+04</td> <td>  958.914</td>
</tr>
<tr>
  <th>Model_Juke</th>                                                    <td>  670.7645</td> <td> 2288.463</td> <td>    0.293</td> <td> 0.769</td> <td>-3815.072</td> <td> 5156.601</td>
</tr>
<tr>
  <th>Model_Justy</th>                                                   <td>-1.377e+04</td> <td> 2509.200</td> <td>   -5.487</td> <td> 0.000</td> <td>-1.87e+04</td> <td>-8849.831</td>
</tr>
<tr>
  <th>Model_K900</th>                                                    <td> 8367.4174</td> <td> 3277.709</td> <td>    2.553</td> <td> 0.011</td> <td> 1942.466</td> <td> 1.48e+04</td>
</tr>
<tr>
  <th>Model_Kizashi</th>                                                 <td> -644.2447</td> <td> 1728.018</td> <td>   -0.373</td> <td> 0.709</td> <td>-4031.498</td> <td> 2743.009</td>
</tr>
<tr>
  <th>Model_LFA</th>                                                     <td> 2.732e+05</td> <td> 7117.954</td> <td>   38.381</td> <td> 0.000</td> <td> 2.59e+05</td> <td> 2.87e+05</td>
</tr>
<tr>
  <th>Model_LHS</th>                                                     <td> -1.89e+04</td> <td> 5469.287</td> <td>   -3.455</td> <td> 0.001</td> <td>-2.96e+04</td> <td>-8176.530</td>
</tr>
<tr>
  <th>Model_LR2</th>                                                     <td>-1.803e+04</td> <td> 2560.829</td> <td>   -7.040</td> <td> 0.000</td> <td> -2.3e+04</td> <td> -1.3e+04</td>
</tr>
<tr>
  <th>Model_LR3</th>                                                     <td>-1.087e+04</td> <td> 3896.577</td> <td>   -2.790</td> <td> 0.005</td> <td>-1.85e+04</td> <td>-3232.038</td>
</tr>
<tr>
  <th>Model_LR4</th>                                                     <td>-9743.8066</td> <td> 2425.525</td> <td>   -4.017</td> <td> 0.000</td> <td>-1.45e+04</td> <td>-4989.302</td>
</tr>
<tr>
  <th>Model_LS</th>                                                      <td> 4.732e+04</td> <td> 3080.881</td> <td>   15.358</td> <td> 0.000</td> <td> 4.13e+04</td> <td> 5.34e+04</td>
</tr>
<tr>
  <th>Model_LS 400</th>                                                  <td>-3.778e+04</td> <td> 4181.339</td> <td>   -9.036</td> <td> 0.000</td> <td> -4.6e+04</td> <td>-2.96e+04</td>
</tr>
<tr>
  <th>Model_LS 430</th>                                                  <td> 6705.9389</td> <td> 4192.580</td> <td>    1.599</td> <td> 0.110</td> <td>-1512.338</td> <td> 1.49e+04</td>
</tr>
<tr>
  <th>Model_LS 460</th>                                                  <td> 1.125e+04</td> <td> 2233.772</td> <td>    5.038</td> <td> 0.000</td> <td> 6875.516</td> <td> 1.56e+04</td>
</tr>
<tr>
  <th>Model_LS 600h L</th>                                               <td> 4.284e+04</td> <td> 4217.611</td> <td>   10.158</td> <td> 0.000</td> <td> 3.46e+04</td> <td> 5.11e+04</td>
</tr>
<tr>
  <th>Model_LSS</th>                                                     <td>-1.616e+04</td> <td> 2935.196</td> <td>   -5.507</td> <td> 0.000</td> <td>-2.19e+04</td> <td>-1.04e+04</td>
</tr>
<tr>
  <th>Model_LTD Crown Victoria</th>                                      <td>-6459.1412</td> <td> 4030.310</td> <td>   -1.603</td> <td> 0.109</td> <td>-1.44e+04</td> <td> 1441.055</td>
</tr>
<tr>
  <th>Model_LX 450</th>                                                  <td> -3.63e+04</td> <td> 5139.659</td> <td>   -7.062</td> <td> 0.000</td> <td>-4.64e+04</td> <td>-2.62e+04</td>
</tr>
<tr>
  <th>Model_LX 470</th>                                                  <td> 1.528e+04</td> <td> 4134.820</td> <td>    3.695</td> <td> 0.000</td> <td> 7171.120</td> <td> 2.34e+04</td>
</tr>
<tr>
  <th>Model_LX 570</th>                                                  <td>  1.47e+04</td> <td> 4173.179</td> <td>    3.522</td> <td> 0.000</td> <td> 6516.499</td> <td> 2.29e+04</td>
</tr>
<tr>
  <th>Model_LaCrosse</th>                                                <td>-3084.4680</td> <td> 2172.127</td> <td>   -1.420</td> <td> 0.156</td> <td>-7342.262</td> <td> 1173.326</td>
</tr>
<tr>
  <th>Model_Lancer</th>                                                  <td>-6891.0780</td> <td> 2106.077</td> <td>   -3.272</td> <td> 0.001</td> <td> -1.1e+04</td> <td>-2762.755</td>
</tr>
<tr>
  <th>Model_Lancer Evolution</th>                                        <td> 3637.8815</td> <td> 2966.317</td> <td>    1.226</td> <td> 0.220</td> <td>-2176.680</td> <td> 9452.443</td>
</tr>
<tr>
  <th>Model_Lancer Sportback</th>                                        <td>-6257.2809</td> <td> 3060.269</td> <td>   -2.045</td> <td> 0.041</td> <td>-1.23e+04</td> <td> -258.554</td>
</tr>
<tr>
  <th>Model_Land Cruiser</th>                                            <td> 3.389e+04</td> <td> 4101.230</td> <td>    8.264</td> <td> 0.000</td> <td> 2.59e+04</td> <td> 4.19e+04</td>
</tr>
<tr>
  <th>Model_Landaulet</th>                                               <td> 7.522e+05</td> <td> 4345.128</td> <td>  173.124</td> <td> 0.000</td> <td> 7.44e+05</td> <td> 7.61e+05</td>
</tr>
<tr>
  <th>Model_Laser</th>                                                   <td>-6625.2790</td> <td> 4733.420</td> <td>   -1.400</td> <td> 0.162</td> <td>-1.59e+04</td> <td> 2653.151</td>
</tr>
<tr>
  <th>Model_Le Baron</th>                                                <td>-2.313e+04</td> <td> 4334.749</td> <td>   -5.337</td> <td> 0.000</td> <td>-3.16e+04</td> <td>-1.46e+04</td>
</tr>
<tr>
  <th>Model_Le Mans</th>                                                 <td>-1.273e+04</td> <td> 2598.505</td> <td>   -4.900</td> <td> 0.000</td> <td>-1.78e+04</td> <td>-7638.416</td>
</tr>
<tr>
  <th>Model_LeSabre</th>                                                 <td> 1584.2894</td> <td> 2949.848</td> <td>    0.537</td> <td> 0.591</td> <td>-4197.989</td> <td> 7366.568</td>
</tr>
<tr>
  <th>Model_Leaf</th>                                                    <td> 1390.2705</td> <td> 5347.259</td> <td>    0.260</td> <td> 0.795</td> <td>-9091.404</td> <td> 1.19e+04</td>
</tr>
<tr>
  <th>Model_Legacy</th>                                                  <td>-5599.5718</td> <td> 1956.264</td> <td>   -2.862</td> <td> 0.004</td> <td>-9434.233</td> <td>-1764.910</td>
</tr>
<tr>
  <th>Model_Legend</th>                                                  <td>-2.394e+04</td> <td> 2031.840</td> <td>  -11.784</td> <td> 0.000</td> <td>-2.79e+04</td> <td>   -2e+04</td>
</tr>
<tr>
  <th>Model_Levante</th>                                                 <td>-3.097e+04</td> <td> 7427.546</td> <td>   -4.169</td> <td> 0.000</td> <td>-4.55e+04</td> <td>-1.64e+04</td>
</tr>
<tr>
  <th>Model_Loyale</th>                                                  <td>-1.265e+04</td> <td> 2567.861</td> <td>   -4.927</td> <td> 0.000</td> <td>-1.77e+04</td> <td>-7617.910</td>
</tr>
<tr>
  <th>Model_Lucerne</th>                                                 <td> 1979.4939</td> <td> 1930.216</td> <td>    1.026</td> <td> 0.305</td> <td>-1804.108</td> <td> 5763.096</td>
</tr>
<tr>
  <th>Model_Lumina</th>                                                  <td>-1.896e+04</td> <td> 3945.446</td> <td>   -4.805</td> <td> 0.000</td> <td>-2.67e+04</td> <td>-1.12e+04</td>
</tr>
<tr>
  <th>Model_Lumina Minivan</th>                                          <td>-1.818e+04</td> <td> 5205.160</td> <td>   -3.493</td> <td> 0.000</td> <td>-2.84e+04</td> <td>-7979.107</td>
</tr>
<tr>
  <th>Model_M</th>                                                       <td> -784.9873</td> <td> 2032.232</td> <td>   -0.386</td> <td> 0.699</td> <td>-4768.560</td> <td> 3198.586</td>
</tr>
<tr>
  <th>Model_M-Class</th>                                                 <td> -1.85e+04</td> <td> 1982.054</td> <td>   -9.335</td> <td> 0.000</td> <td>-2.24e+04</td> <td>-1.46e+04</td>
</tr>
<tr>
  <th>Model_M2</th>                                                      <td>-7521.3778</td> <td> 5450.104</td> <td>   -1.380</td> <td> 0.168</td> <td>-1.82e+04</td> <td> 3161.893</td>
</tr>
<tr>
  <th>Model_M3</th>                                                      <td>-3923.3553</td> <td> 4652.308</td> <td>   -0.843</td> <td> 0.399</td> <td> -1.3e+04</td> <td> 5196.079</td>
</tr>
<tr>
  <th>Model_M30</th>                                                     <td>-2.446e+04</td> <td> 4338.824</td> <td>   -5.637</td> <td> 0.000</td> <td> -3.3e+04</td> <td> -1.6e+04</td>
</tr>
<tr>
  <th>Model_M35</th>                                                     <td> 1832.5974</td> <td> 3591.863</td> <td>    0.510</td> <td> 0.610</td> <td>-5208.157</td> <td> 8873.352</td>
</tr>
<tr>
  <th>Model_M37</th>                                                     <td>-1105.1860</td> <td> 5425.624</td> <td>   -0.204</td> <td> 0.839</td> <td>-1.17e+04</td> <td> 9530.099</td>
</tr>
<tr>
  <th>Model_M4</th>                                                      <td>-1461.1448</td> <td> 3634.788</td> <td>   -0.402</td> <td> 0.688</td> <td>-8586.040</td> <td> 5663.751</td>
</tr>
<tr>
  <th>Model_M4 GTS</th>                                                  <td> 4.784e+04</td> <td> 7420.256</td> <td>    6.447</td> <td> 0.000</td> <td> 3.33e+04</td> <td> 6.24e+04</td>
</tr>
<tr>
  <th>Model_M45</th>                                                     <td> 3298.6865</td> <td> 3593.629</td> <td>    0.918</td> <td> 0.359</td> <td>-3745.530</td> <td> 1.03e+04</td>
</tr>
<tr>
  <th>Model_M5</th>                                                      <td>-1.051e+04</td> <td> 4790.744</td> <td>   -2.194</td> <td> 0.028</td> <td>-1.99e+04</td> <td>-1121.862</td>
</tr>
<tr>
  <th>Model_M56</th>                                                     <td>-3202.6369</td> <td> 5415.114</td> <td>   -0.591</td> <td> 0.554</td> <td>-1.38e+04</td> <td> 7412.046</td>
</tr>
<tr>
  <th>Model_M6</th>                                                      <td> 9363.6697</td> <td> 3691.472</td> <td>    2.537</td> <td> 0.011</td> <td> 2127.662</td> <td> 1.66e+04</td>
</tr>
<tr>
  <th>Model_M6 Gran Coupe</th>                                           <td> 1.201e+04</td> <td> 4802.251</td> <td>    2.501</td> <td> 0.012</td> <td> 2596.634</td> <td> 2.14e+04</td>
</tr>
<tr>
  <th>Model_MDX</th>                                                     <td>-1416.6606</td> <td> 1529.375</td> <td>   -0.926</td> <td> 0.354</td> <td>-4414.536</td> <td> 1581.215</td>
</tr>
<tr>
  <th>Model_MKC</th>                                                     <td> 4.121e+04</td> <td> 2759.702</td> <td>   14.934</td> <td> 0.000</td> <td> 3.58e+04</td> <td> 4.66e+04</td>
</tr>
<tr>
  <th>Model_MKS</th>                                                     <td> 3.501e+04</td> <td> 3172.592</td> <td>   11.036</td> <td> 0.000</td> <td> 2.88e+04</td> <td> 4.12e+04</td>
</tr>
<tr>
  <th>Model_MKT</th>                                                     <td> 3.666e+04</td> <td> 3655.399</td> <td>   10.030</td> <td> 0.000</td> <td> 2.95e+04</td> <td> 4.38e+04</td>
</tr>
<tr>
  <th>Model_MKX</th>                                                     <td> 4.037e+04</td> <td> 2659.838</td> <td>   15.176</td> <td> 0.000</td> <td> 3.52e+04</td> <td> 4.56e+04</td>
</tr>
<tr>
  <th>Model_MKZ</th>                                                     <td> 4.545e+04</td> <td> 2644.995</td> <td>   17.183</td> <td> 0.000</td> <td> 4.03e+04</td> <td> 5.06e+04</td>
</tr>
<tr>
  <th>Model_MKZ Hybrid</th>                                              <td> 4.595e+04</td> <td> 7367.178</td> <td>    6.237</td> <td> 0.000</td> <td> 3.15e+04</td> <td> 6.04e+04</td>
</tr>
<tr>
  <th>Model_ML55 AMG</th>                                                <td>-6.109e+04</td> <td> 7034.609</td> <td>   -8.684</td> <td> 0.000</td> <td>-7.49e+04</td> <td>-4.73e+04</td>
</tr>
<tr>
  <th>Model_MP4-12C</th>                                                 <td> 3729.8231</td> <td> 4860.062</td> <td>    0.767</td> <td> 0.443</td> <td>-5796.851</td> <td> 1.33e+04</td>
</tr>
<tr>
  <th>Model_MPV</th>                                                     <td> -840.0421</td> <td> 4638.817</td> <td>   -0.181</td> <td> 0.856</td> <td>-9933.031</td> <td> 8252.947</td>
</tr>
<tr>
  <th>Model_MR2</th>                                                     <td>-6326.5774</td> <td> 3236.439</td> <td>   -1.955</td> <td> 0.051</td> <td>-1.27e+04</td> <td>   17.477</td>
</tr>
<tr>
  <th>Model_MR2 Spyder</th>                                              <td>-1745.7910</td> <td> 3030.804</td> <td>   -0.576</td> <td> 0.565</td> <td>-7686.760</td> <td> 4195.178</td>
</tr>
<tr>
  <th>Model_MX-3</th>                                                    <td>-1.639e+04</td> <td> 3737.544</td> <td>   -4.385</td> <td> 0.000</td> <td>-2.37e+04</td> <td>-9064.461</td>
</tr>
<tr>
  <th>Model_MX-5 Miata</th>                                              <td>-1.016e+04</td> <td> 2119.902</td> <td>   -4.793</td> <td> 0.000</td> <td>-1.43e+04</td> <td>-6005.283</td>
</tr>
<tr>
  <th>Model_MX-6</th>                                                    <td>-1.517e+04</td> <td> 3381.938</td> <td>   -4.484</td> <td> 0.000</td> <td>-2.18e+04</td> <td>-8536.525</td>
</tr>
<tr>
  <th>Model_Macan</th>                                                   <td>-3.893e+04</td> <td> 2689.996</td> <td>  -14.471</td> <td> 0.000</td> <td>-4.42e+04</td> <td>-3.37e+04</td>
</tr>
<tr>
  <th>Model_Magnum</th>                                                  <td>-4824.3304</td> <td> 4796.725</td> <td>   -1.006</td> <td> 0.315</td> <td>-1.42e+04</td> <td> 4578.191</td>
</tr>
<tr>
  <th>Model_Malibu</th>                                                  <td>-9407.7336</td> <td> 2268.534</td> <td>   -4.147</td> <td> 0.000</td> <td>-1.39e+04</td> <td>-4960.963</td>
</tr>
<tr>
  <th>Model_Malibu Classic</th>                                          <td>-1.144e+04</td> <td> 4316.345</td> <td>   -2.651</td> <td> 0.008</td> <td>-1.99e+04</td> <td>-2983.193</td>
</tr>
<tr>
  <th>Model_Malibu Hybrid</th>                                           <td>-4229.3867</td> <td> 4309.215</td> <td>   -0.981</td> <td> 0.326</td> <td>-1.27e+04</td> <td> 4217.519</td>
</tr>
<tr>
  <th>Model_Malibu Limited</th>                                          <td>-9467.5757</td> <td> 4293.193</td> <td>   -2.205</td> <td> 0.027</td> <td>-1.79e+04</td> <td>-1052.076</td>
</tr>
<tr>
  <th>Model_Malibu Maxx</th>                                             <td>-9560.3987</td> <td> 2673.641</td> <td>   -3.576</td> <td> 0.000</td> <td>-1.48e+04</td> <td>-4319.539</td>
</tr>
<tr>
  <th>Model_Mark LT</th>                                                 <td> 3.156e+04</td> <td> 3096.891</td> <td>   10.191</td> <td> 0.000</td> <td> 2.55e+04</td> <td> 3.76e+04</td>
</tr>
<tr>
  <th>Model_Mark VII</th>                                                <td> 2.019e+04</td> <td> 4863.469</td> <td>    4.150</td> <td> 0.000</td> <td> 1.07e+04</td> <td> 2.97e+04</td>
</tr>
<tr>
  <th>Model_Mark VIII</th>                                               <td>  1.33e+04</td> <td> 3987.982</td> <td>    3.334</td> <td> 0.001</td> <td> 5480.674</td> <td> 2.11e+04</td>
</tr>
<tr>
  <th>Model_Matrix</th>                                                  <td>-5093.3753</td> <td> 2027.899</td> <td>   -2.512</td> <td> 0.012</td> <td>-9068.454</td> <td>-1118.297</td>
</tr>
<tr>
  <th>Model_Maxima</th>                                                  <td> 6828.9394</td> <td> 2773.623</td> <td>    2.462</td> <td> 0.014</td> <td> 1392.095</td> <td> 1.23e+04</td>
</tr>
<tr>
  <th>Model_Maybach</th>                                                 <td> 7.901e+04</td> <td> 5129.745</td> <td>   15.402</td> <td> 0.000</td> <td>  6.9e+04</td> <td> 8.91e+04</td>
</tr>
<tr>
  <th>Model_Mazdaspeed 3</th>                                            <td>-7452.7927</td> <td> 4392.794</td> <td>   -1.697</td> <td> 0.090</td> <td>-1.61e+04</td> <td> 1157.944</td>
</tr>
<tr>
  <th>Model_Mazdaspeed 6</th>                                            <td>  898.2330</td> <td> 3890.465</td> <td>    0.231</td> <td> 0.817</td> <td>-6727.840</td> <td> 8524.306</td>
</tr>
<tr>
  <th>Model_Mazdaspeed MX-5 Miata</th>                                   <td>-3747.2574</td> <td> 4374.400</td> <td>   -0.857</td> <td> 0.392</td> <td>-1.23e+04</td> <td> 4827.423</td>
</tr>
<tr>
  <th>Model_Mazdaspeed Protege</th>                                      <td> -965.0405</td> <td> 5219.163</td> <td>   -0.185</td> <td> 0.853</td> <td>-1.12e+04</td> <td> 9265.540</td>
</tr>
<tr>
  <th>Model_Metris</th>                                                  <td>-2.897e+04</td> <td> 6385.695</td> <td>   -4.537</td> <td> 0.000</td> <td>-4.15e+04</td> <td>-1.65e+04</td>
</tr>
<tr>
  <th>Model_Metro</th>                                                   <td>-2.245e+04</td> <td> 3275.606</td> <td>   -6.853</td> <td> 0.000</td> <td>-2.89e+04</td> <td> -1.6e+04</td>
</tr>
<tr>
  <th>Model_Mighty Max Pickup</th>                                       <td>-1.378e+04</td> <td> 3192.122</td> <td>   -4.316</td> <td> 0.000</td> <td>   -2e+04</td> <td>-7521.028</td>
</tr>
<tr>
  <th>Model_Millenia</th>                                                <td>-2545.4407</td> <td> 2872.889</td> <td>   -0.886</td> <td> 0.376</td> <td>-8176.865</td> <td> 3085.984</td>
</tr>
<tr>
  <th>Model_Mirage</th>                                                  <td>-1.484e+04</td> <td> 2393.763</td> <td>   -6.197</td> <td> 0.000</td> <td>-1.95e+04</td> <td>-1.01e+04</td>
</tr>
<tr>
  <th>Model_Mirage G4</th>                                               <td>-1.366e+04</td> <td> 4235.426</td> <td>   -3.225</td> <td> 0.001</td> <td> -2.2e+04</td> <td>-5357.613</td>
</tr>
<tr>
  <th>Model_Model D</th>                                                 <td>-3726.3687</td> <td> 2508.038</td> <td>   -1.486</td> <td> 0.137</td> <td>-8642.615</td> <td> 1189.877</td>
</tr>
<tr>
  <th>Model_Model S</th>                                                 <td>-2.336e+04</td> <td> 2699.272</td> <td>   -8.652</td> <td> 0.000</td> <td>-2.86e+04</td> <td>-1.81e+04</td>
</tr>
<tr>
  <th>Model_Monaco</th>                                                  <td> -1.08e+04</td> <td> 6319.182</td> <td>   -1.710</td> <td> 0.087</td> <td>-2.32e+04</td> <td> 1583.093</td>
</tr>
<tr>
  <th>Model_Montana</th>                                                 <td> 4302.0981</td> <td> 3844.360</td> <td>    1.119</td> <td> 0.263</td> <td>-3233.600</td> <td> 1.18e+04</td>
</tr>
<tr>
  <th>Model_Montana SV6</th>                                             <td>  411.2000</td> <td> 5933.381</td> <td>    0.069</td> <td> 0.945</td> <td>-1.12e+04</td> <td>  1.2e+04</td>
</tr>
<tr>
  <th>Model_Monte Carlo</th>                                             <td>-7170.7976</td> <td> 2630.461</td> <td>   -2.726</td> <td> 0.006</td> <td>-1.23e+04</td> <td>-2014.580</td>
</tr>
<tr>
  <th>Model_Montero</th>                                                 <td> 9739.9702</td> <td> 4045.074</td> <td>    2.408</td> <td> 0.016</td> <td> 1810.832</td> <td> 1.77e+04</td>
</tr>
<tr>
  <th>Model_Montero Sport</th>                                           <td> 4788.0098</td> <td> 1806.451</td> <td>    2.651</td> <td> 0.008</td> <td> 1247.012</td> <td> 8329.008</td>
</tr>
<tr>
  <th>Model_Mulsanne</th>                                                <td> 1.706e+04</td> <td> 4201.410</td> <td>    4.060</td> <td> 0.000</td> <td> 8822.090</td> <td> 2.53e+04</td>
</tr>
<tr>
  <th>Model_Murano</th>                                                  <td> 7795.7258</td> <td> 2075.863</td> <td>    3.755</td> <td> 0.000</td> <td> 3726.628</td> <td> 1.19e+04</td>
</tr>
<tr>
  <th>Model_Murano CrossCabriolet</th>                                   <td> 1.396e+04</td> <td> 4693.098</td> <td>    2.975</td> <td> 0.003</td> <td> 4763.293</td> <td> 2.32e+04</td>
</tr>
<tr>
  <th>Model_Murcielago</th>                                              <td>-9.985e+04</td> <td> 2629.525</td> <td>  -37.971</td> <td> 0.000</td> <td>-1.05e+05</td> <td>-9.47e+04</td>
</tr>
<tr>
  <th>Model_Mustang</th>                                                 <td>-3058.8967</td> <td> 1868.180</td> <td>   -1.637</td> <td> 0.102</td> <td>-6720.896</td> <td>  603.102</td>
</tr>
<tr>
  <th>Model_Mustang SVT Cobra</th>                                       <td>-1.511e+04</td> <td> 3210.533</td> <td>   -4.708</td> <td> 0.000</td> <td>-2.14e+04</td> <td>-8821.441</td>
</tr>
<tr>
  <th>Model_NSX</th>                                                     <td> 5.312e+04</td> <td> 3202.387</td> <td>   16.588</td> <td> 0.000</td> <td> 4.68e+04</td> <td> 5.94e+04</td>
</tr>
<tr>
  <th>Model_NV200</th>                                                   <td> 2320.8189</td> <td> 4990.618</td> <td>    0.465</td> <td> 0.642</td> <td>-7461.769</td> <td> 1.21e+04</td>
</tr>
<tr>
  <th>Model_NX</th>                                                      <td>-3028.7121</td> <td> 3372.106</td> <td>   -0.898</td> <td> 0.369</td> <td>-9638.700</td> <td> 3581.276</td>
</tr>
<tr>
  <th>Model_NX 200t</th>                                                 <td>-1.573e+04</td> <td> 2206.290</td> <td>   -7.130</td> <td> 0.000</td> <td>-2.01e+04</td> <td>-1.14e+04</td>
</tr>
<tr>
  <th>Model_NX 300h</th>                                                 <td>-9709.9497</td> <td> 3315.209</td> <td>   -2.929</td> <td> 0.003</td> <td>-1.62e+04</td> <td>-3211.490</td>
</tr>
<tr>
  <th>Model_Navajo</th>                                                  <td> -1.54e+04</td> <td> 3778.941</td> <td>   -4.076</td> <td> 0.000</td> <td>-2.28e+04</td> <td>-7997.336</td>
</tr>
<tr>
  <th>Model_Navigator</th>                                               <td> 5.484e+04</td> <td> 2695.557</td> <td>   20.344</td> <td> 0.000</td> <td> 4.96e+04</td> <td> 6.01e+04</td>
</tr>
<tr>
  <th>Model_Neon</th>                                                    <td>-4362.1242</td> <td> 4369.753</td> <td>   -0.998</td> <td> 0.318</td> <td>-1.29e+04</td> <td> 4203.448</td>
</tr>
<tr>
  <th>Model_New Beetle</th>                                              <td>-1.231e+04</td> <td> 5733.366</td> <td>   -2.147</td> <td> 0.032</td> <td>-2.35e+04</td> <td>-1071.747</td>
</tr>
<tr>
  <th>Model_New Yorker</th>                                              <td> -2.26e+04</td> <td> 5613.552</td> <td>   -4.026</td> <td> 0.000</td> <td>-3.36e+04</td> <td>-1.16e+04</td>
</tr>
<tr>
  <th>Model_Ninety-Eight</th>                                            <td>-1.385e+04</td> <td> 3160.792</td> <td>   -4.382</td> <td> 0.000</td> <td>   -2e+04</td> <td>-7654.103</td>
</tr>
<tr>
  <th>Model_Nitro</th>                                                   <td>-5735.3082</td> <td> 4469.337</td> <td>   -1.283</td> <td> 0.199</td> <td>-1.45e+04</td> <td> 3025.467</td>
</tr>
<tr>
  <th>Model_Odyssey</th>                                                 <td> 4066.6707</td> <td> 4478.069</td> <td>    0.908</td> <td> 0.364</td> <td>-4711.221</td> <td> 1.28e+04</td>
</tr>
<tr>
  <th>Model_Omni</th>                                                    <td>-1.005e+04</td> <td> 8604.851</td> <td>   -1.168</td> <td> 0.243</td> <td>-2.69e+04</td> <td> 6818.987</td>
</tr>
<tr>
  <th>Model_Optima</th>                                                  <td>-1371.1320</td> <td> 1879.143</td> <td>   -0.730</td> <td> 0.466</td> <td>-5054.621</td> <td> 2312.357</td>
</tr>
<tr>
  <th>Model_Optima Hybrid</th>                                           <td> 1139.1152</td> <td> 2998.690</td> <td>    0.380</td> <td> 0.704</td> <td>-4738.905</td> <td> 7017.135</td>
</tr>
<tr>
  <th>Model_Outback</th>                                                 <td>-4343.8414</td> <td> 1782.717</td> <td>   -2.437</td> <td> 0.015</td> <td>-7838.317</td> <td> -849.366</td>
</tr>
<tr>
  <th>Model_Outlander</th>                                               <td>-5050.7123</td> <td> 1971.623</td> <td>   -2.562</td> <td> 0.010</td> <td>-8915.480</td> <td>-1185.945</td>
</tr>
<tr>
  <th>Model_Outlander Sport</th>                                         <td>-7290.9516</td> <td> 1751.847</td> <td>   -4.162</td> <td> 0.000</td> <td>-1.07e+04</td> <td>-3856.989</td>
</tr>
<tr>
  <th>Model_PT Cruiser</th>                                              <td>-1.179e+04</td> <td> 4160.059</td> <td>   -2.835</td> <td> 0.005</td> <td>-1.99e+04</td> <td>-3638.648</td>
</tr>
<tr>
  <th>Model_Pacifica</th>                                                <td>-6666.4631</td> <td> 3947.429</td> <td>   -1.689</td> <td> 0.091</td> <td>-1.44e+04</td> <td> 1071.270</td>
</tr>
<tr>
  <th>Model_Panamera</th>                                                <td> 9011.5754</td> <td> 2152.165</td> <td>    4.187</td> <td> 0.000</td> <td> 4792.910</td> <td> 1.32e+04</td>
</tr>
<tr>
  <th>Model_Park Avenue</th>                                             <td> 9386.2134</td> <td> 2957.505</td> <td>    3.174</td> <td> 0.002</td> <td> 3588.924</td> <td> 1.52e+04</td>
</tr>
<tr>
  <th>Model_Park Ward</th>                                               <td>-3.565e+04</td> <td> 4821.699</td> <td>   -7.394</td> <td> 0.000</td> <td>-4.51e+04</td> <td>-2.62e+04</td>
</tr>
<tr>
  <th>Model_Paseo</th>                                                   <td>-1.286e+04</td> <td> 3735.632</td> <td>   -3.443</td> <td> 0.001</td> <td>-2.02e+04</td> <td>-5538.139</td>
</tr>
<tr>
  <th>Model_Passat</th>                                                  <td>-7372.0319</td> <td> 5765.820</td> <td>   -1.279</td> <td> 0.201</td> <td>-1.87e+04</td> <td> 3930.105</td>
</tr>
<tr>
  <th>Model_Passport</th>                                                <td> -426.0252</td> <td> 2001.632</td> <td>   -0.213</td> <td> 0.831</td> <td>-4349.616</td> <td> 3497.566</td>
</tr>
<tr>
  <th>Model_Pathfinder</th>                                              <td> 4248.8570</td> <td> 2211.292</td> <td>    1.921</td> <td> 0.055</td> <td>  -85.709</td> <td> 8583.423</td>
</tr>
<tr>
  <th>Model_Phaeton</th>                                                 <td> 3.075e+04</td> <td> 5854.820</td> <td>    5.252</td> <td> 0.000</td> <td> 1.93e+04</td> <td> 4.22e+04</td>
</tr>
<tr>
  <th>Model_Phantom</th>                                                 <td> 1.209e+05</td> <td> 3150.830</td> <td>   38.375</td> <td> 0.000</td> <td> 1.15e+05</td> <td> 1.27e+05</td>
</tr>
<tr>
  <th>Model_Phantom Coupe</th>                                           <td> 1.155e+05</td> <td> 4116.543</td> <td>   28.061</td> <td> 0.000</td> <td> 1.07e+05</td> <td> 1.24e+05</td>
</tr>
<tr>
  <th>Model_Phantom Drophead Coupe</th>                                  <td> 1.491e+05</td> <td> 4117.446</td> <td>   36.206</td> <td> 0.000</td> <td> 1.41e+05</td> <td> 1.57e+05</td>
</tr>
<tr>
  <th>Model_Pickup</th>                                                  <td>-1.448e+04</td> <td> 1853.402</td> <td>   -7.812</td> <td> 0.000</td> <td>-1.81e+04</td> <td>-1.08e+04</td>
</tr>
<tr>
  <th>Model_Pilot</th>                                                   <td>   51.6121</td> <td> 1317.505</td> <td>    0.039</td> <td> 0.969</td> <td>-2530.956</td> <td> 2634.180</td>
</tr>
<tr>
  <th>Model_Precis</th>                                                  <td>-1.062e+04</td> <td> 6983.092</td> <td>   -1.520</td> <td> 0.129</td> <td>-2.43e+04</td> <td> 3073.057</td>
</tr>
<tr>
  <th>Model_Prelude</th>                                                 <td>-3053.2410</td> <td> 2858.075</td> <td>   -1.068</td> <td> 0.285</td> <td>-8655.628</td> <td> 2549.146</td>
</tr>
<tr>
  <th>Model_Previa</th>                                                  <td>-9489.0721</td> <td> 4639.740</td> <td>   -2.045</td> <td> 0.041</td> <td>-1.86e+04</td> <td> -394.273</td>
</tr>
<tr>
  <th>Model_Prius</th>                                                   <td> -462.8847</td> <td> 2483.306</td> <td>   -0.186</td> <td> 0.852</td> <td>-5330.650</td> <td> 4404.881</td>
</tr>
<tr>
  <th>Model_Prius Prime</th>                                             <td> 1488.5464</td> <td> 4473.467</td> <td>    0.333</td> <td> 0.739</td> <td>-7280.325</td> <td> 1.03e+04</td>
</tr>
<tr>
  <th>Model_Prius c</th>                                                 <td>-6688.2681</td> <td> 2664.700</td> <td>   -2.510</td> <td> 0.012</td> <td>-1.19e+04</td> <td>-1464.935</td>
</tr>
<tr>
  <th>Model_Prius v</th>                                                 <td> 2385.6769</td> <td> 2498.000</td> <td>    0.955</td> <td> 0.340</td> <td>-2510.892</td> <td> 7282.246</td>
</tr>
<tr>
  <th>Model_Prizm</th>                                                   <td>-1.225e+04</td> <td> 3355.982</td> <td>   -3.651</td> <td> 0.000</td> <td>-1.88e+04</td> <td>-5672.737</td>
</tr>
<tr>
  <th>Model_Probe</th>                                                   <td>-5976.1594</td> <td> 3298.962</td> <td>   -1.812</td> <td> 0.070</td> <td>-1.24e+04</td> <td>  490.453</td>
</tr>
<tr>
  <th>Model_Protege</th>                                                 <td>-5851.7125</td> <td> 2628.688</td> <td>   -2.226</td> <td> 0.026</td> <td> -1.1e+04</td> <td> -698.969</td>
</tr>
<tr>
  <th>Model_Protege5</th>                                                <td>-4983.8724</td> <td> 5236.561</td> <td>   -0.952</td> <td> 0.341</td> <td>-1.52e+04</td> <td> 5280.813</td>
</tr>
<tr>
  <th>Model_Prowler</th>                                                 <td> 5712.3133</td> <td> 4778.000</td> <td>    1.196</td> <td> 0.232</td> <td>-3653.503</td> <td> 1.51e+04</td>
</tr>
<tr>
  <th>Model_Pulsar</th>                                                  <td>-1575.8871</td> <td> 7249.402</td> <td>   -0.217</td> <td> 0.828</td> <td>-1.58e+04</td> <td> 1.26e+04</td>
</tr>
<tr>
  <th>Model_Q3</th>                                                      <td> 1544.2599</td> <td> 3545.293</td> <td>    0.436</td> <td> 0.663</td> <td>-5405.209</td> <td> 8493.729</td>
</tr>
<tr>
  <th>Model_Q40</th>                                                     <td>-1.489e+04</td> <td> 5453.553</td> <td>   -2.730</td> <td> 0.006</td> <td>-2.56e+04</td> <td>-4196.094</td>
</tr>
<tr>
  <th>Model_Q45</th>                                                     <td> 1.134e+04</td> <td> 4189.839</td> <td>    2.708</td> <td> 0.007</td> <td> 3131.096</td> <td> 1.96e+04</td>
</tr>
<tr>
  <th>Model_Q5</th>                                                      <td> 1.004e+04</td> <td> 3357.402</td> <td>    2.990</td> <td> 0.003</td> <td> 3457.062</td> <td> 1.66e+04</td>
</tr>
<tr>
  <th>Model_Q50</th>                                                     <td>-5942.4857</td> <td> 2488.843</td> <td>   -2.388</td> <td> 0.017</td> <td>-1.08e+04</td> <td>-1063.866</td>
</tr>
<tr>
  <th>Model_Q60 Convertible</th>                                         <td>-1301.1088</td> <td> 3687.387</td> <td>   -0.353</td> <td> 0.724</td> <td>-8529.108</td> <td> 5926.891</td>
</tr>
<tr>
  <th>Model_Q60 Coupe</th>                                               <td>-2820.6768</td> <td> 2827.677</td> <td>   -0.998</td> <td> 0.319</td> <td>-8363.478</td> <td> 2722.125</td>
</tr>
<tr>
  <th>Model_Q7</th>                                                      <td>  1.39e+04</td> <td> 3492.407</td> <td>    3.980</td> <td> 0.000</td> <td> 7054.181</td> <td> 2.07e+04</td>
</tr>
<tr>
  <th>Model_Q70</th>                                                     <td> -447.3480</td> <td> 2541.894</td> <td>   -0.176</td> <td> 0.860</td> <td>-5429.959</td> <td> 4535.263</td>
</tr>
<tr>
  <th>Model_QX</th>                                                      <td>-1864.0524</td> <td> 4108.921</td> <td>   -0.454</td> <td> 0.650</td> <td>-9918.342</td> <td> 6190.238</td>
</tr>
<tr>
  <th>Model_QX4</th>                                                     <td>-2027.6247</td> <td> 3794.643</td> <td>   -0.534</td> <td> 0.593</td> <td>-9465.868</td> <td> 5410.618</td>
</tr>
<tr>
  <th>Model_QX50</th>                                                    <td>-1.707e+04</td> <td> 3282.311</td> <td>   -5.200</td> <td> 0.000</td> <td>-2.35e+04</td> <td>-1.06e+04</td>
</tr>
<tr>
  <th>Model_QX56</th>                                                    <td> 2126.9897</td> <td> 3609.277</td> <td>    0.589</td> <td> 0.556</td> <td>-4947.900</td> <td> 9201.880</td>
</tr>
<tr>
  <th>Model_QX60</th>                                                    <td>-3824.2082</td> <td> 3057.790</td> <td>   -1.251</td> <td> 0.211</td> <td>-9818.075</td> <td> 2169.658</td>
</tr>
<tr>
  <th>Model_QX70</th>                                                    <td>-6321.2750</td> <td> 3575.790</td> <td>   -1.768</td> <td> 0.077</td> <td>-1.33e+04</td> <td>  687.974</td>
</tr>
<tr>
  <th>Model_QX80</th>                                                    <td> 1824.9559</td> <td> 3385.079</td> <td>    0.539</td> <td> 0.590</td> <td>-4810.462</td> <td> 8460.374</td>
</tr>
<tr>
  <th>Model_Quattroporte</th>                                            <td> 4941.4690</td> <td> 5921.859</td> <td>    0.834</td> <td> 0.404</td> <td>-6666.534</td> <td> 1.65e+04</td>
</tr>
<tr>
  <th>Model_Quest</th>                                                   <td> 8889.1278</td> <td> 4265.761</td> <td>    2.084</td> <td> 0.037</td> <td>  527.401</td> <td> 1.73e+04</td>
</tr>
<tr>
  <th>Model_R-Class</th>                                                 <td>-1.659e+04</td> <td> 3134.073</td> <td>   -5.294</td> <td> 0.000</td> <td>-2.27e+04</td> <td>-1.04e+04</td>
</tr>
<tr>
  <th>Model_R32</th>                                                     <td>-5058.8065</td> <td> 7395.470</td> <td>   -0.684</td> <td> 0.494</td> <td>-1.96e+04</td> <td> 9437.764</td>
</tr>
<tr>
  <th>Model_R8</th>                                                      <td> 7.984e+04</td> <td> 3443.482</td> <td>   23.185</td> <td> 0.000</td> <td> 7.31e+04</td> <td> 8.66e+04</td>
</tr>
<tr>
  <th>Model_RAM 150</th>                                                 <td> -1.86e+04</td> <td> 4531.071</td> <td>   -4.106</td> <td> 0.000</td> <td>-2.75e+04</td> <td>-9720.899</td>
</tr>
<tr>
  <th>Model_RAM 250</th>                                                 <td>-1.959e+04</td> <td> 4440.781</td> <td>   -4.411</td> <td> 0.000</td> <td>-2.83e+04</td> <td>-1.09e+04</td>
</tr>
<tr>
  <th>Model_RAV4</th>                                                    <td>-1222.3593</td> <td> 1557.203</td> <td>   -0.785</td> <td> 0.432</td> <td>-4274.783</td> <td> 1830.064</td>
</tr>
<tr>
  <th>Model_RAV4 EV</th>                                                 <td>  1.29e+04</td> <td> 6313.465</td> <td>    2.043</td> <td> 0.041</td> <td>  520.769</td> <td> 2.53e+04</td>
</tr>
<tr>
  <th>Model_RAV4 Hybrid</th>                                             <td>  855.1799</td> <td> 3229.060</td> <td>    0.265</td> <td> 0.791</td> <td>-5474.409</td> <td> 7184.769</td>
</tr>
<tr>
  <th>Model_RC 200t</th>                                                 <td>-8219.5141</td> <td> 5069.598</td> <td>   -1.621</td> <td> 0.105</td> <td>-1.82e+04</td> <td> 1717.890</td>
</tr>
<tr>
  <th>Model_RC 300</th>                                                  <td>-1.153e+04</td> <td> 5065.455</td> <td>   -2.276</td> <td> 0.023</td> <td>-2.15e+04</td> <td>-1601.995</td>
</tr>
<tr>
  <th>Model_RC 350</th>                                                  <td>-1.176e+04</td> <td> 3036.022</td> <td>   -3.874</td> <td> 0.000</td> <td>-1.77e+04</td> <td>-5811.516</td>
</tr>
<tr>
  <th>Model_RC F</th>                                                    <td>-1.574e+04</td> <td> 4184.607</td> <td>   -3.762</td> <td> 0.000</td> <td>-2.39e+04</td> <td>-7541.489</td>
</tr>
<tr>
  <th>Model_RDX</th>                                                     <td>-1.182e+04</td> <td> 1710.819</td> <td>   -6.909</td> <td> 0.000</td> <td>-1.52e+04</td> <td>-8466.346</td>
</tr>
<tr>
  <th>Model_RL</th>                                                      <td> 4561.7908</td> <td> 2457.248</td> <td>    1.856</td> <td> 0.063</td> <td> -254.897</td> <td> 9378.479</td>
</tr>
<tr>
  <th>Model_RLX</th>                                                     <td> 5113.0781</td> <td> 2263.681</td> <td>    2.259</td> <td> 0.024</td> <td>  675.821</td> <td> 9550.336</td>
</tr>
<tr>
  <th>Model_RS 4</th>                                                    <td>  2.05e+04</td> <td> 4994.657</td> <td>    4.105</td> <td> 0.000</td> <td> 1.07e+04</td> <td> 3.03e+04</td>
</tr>
<tr>
  <th>Model_RS 5</th>                                                    <td> 6877.1486</td> <td> 4260.223</td> <td>    1.614</td> <td> 0.106</td> <td>-1473.723</td> <td> 1.52e+04</td>
</tr>
<tr>
  <th>Model_RS 6</th>                                                    <td> 3.154e+04</td> <td> 7624.190</td> <td>    4.136</td> <td> 0.000</td> <td> 1.66e+04</td> <td> 4.65e+04</td>
</tr>
<tr>
  <th>Model_RS 7</th>                                                    <td> 2.215e+04</td> <td> 5205.010</td> <td>    4.256</td> <td> 0.000</td> <td> 1.19e+04</td> <td> 3.24e+04</td>
</tr>
<tr>
  <th>Model_RSX</th>                                                     <td>-1.223e+04</td> <td> 2075.918</td> <td>   -5.894</td> <td> 0.000</td> <td>-1.63e+04</td> <td>-8165.318</td>
</tr>
<tr>
  <th>Model_RX 300</th>                                                  <td>-8457.1160</td> <td> 3031.784</td> <td>   -2.789</td> <td> 0.005</td> <td>-1.44e+04</td> <td>-2514.225</td>
</tr>
<tr>
  <th>Model_RX 330</th>                                                  <td>-9289.7511</td> <td> 3001.332</td> <td>   -3.095</td> <td> 0.002</td> <td>-1.52e+04</td> <td>-3406.553</td>
</tr>
<tr>
  <th>Model_RX 350</th>                                                  <td>-1.123e+04</td> <td> 2274.406</td> <td>   -4.937</td> <td> 0.000</td> <td>-1.57e+04</td> <td>-6770.389</td>
</tr>
<tr>
  <th>Model_RX 400h</th>                                                 <td>-5295.0538</td> <td> 3003.875</td> <td>   -1.763</td> <td> 0.078</td> <td>-1.12e+04</td> <td>  593.129</td>
</tr>
<tr>
  <th>Model_RX 450h</th>                                                 <td>-6036.3818</td> <td> 2854.888</td> <td>   -2.114</td> <td> 0.035</td> <td>-1.16e+04</td> <td> -440.242</td>
</tr>
<tr>
  <th>Model_RX-7</th>                                                    <td>-1741.4610</td> <td> 4692.174</td> <td>   -0.371</td> <td> 0.711</td> <td>-1.09e+04</td> <td> 7456.119</td>
</tr>
<tr>
  <th>Model_RX-8</th>                                                    <td> 5787.5040</td> <td> 2696.665</td> <td>    2.146</td> <td> 0.032</td> <td>  501.513</td> <td> 1.11e+04</td>
</tr>
<tr>
  <th>Model_Rabbit</th>                                                  <td>-1.319e+04</td> <td> 5838.965</td> <td>   -2.259</td> <td> 0.024</td> <td>-2.46e+04</td> <td>-1745.935</td>
</tr>
<tr>
  <th>Model_Raider</th>                                                  <td>-7074.9759</td> <td> 2311.784</td> <td>   -3.060</td> <td> 0.002</td> <td>-1.16e+04</td> <td>-2543.426</td>
</tr>
<tr>
  <th>Model_Rainier</th>                                                 <td> -854.0842</td> <td> 2904.857</td> <td>   -0.294</td> <td> 0.769</td> <td>-6548.173</td> <td> 4840.005</td>
</tr>
<tr>
  <th>Model_Rally Wagon</th>                                             <td> 2929.2805</td> <td> 4942.627</td> <td>    0.593</td> <td> 0.553</td> <td>-6759.236</td> <td> 1.26e+04</td>
</tr>
<tr>
  <th>Model_Ram 50 Pickup</th>                                           <td>-1.422e+04</td> <td> 4585.638</td> <td>   -3.102</td> <td> 0.002</td> <td>-2.32e+04</td> <td>-5233.783</td>
</tr>
<tr>
  <th>Model_Ram Cargo</th>                                               <td> -1.12e+04</td> <td> 4456.287</td> <td>   -2.514</td> <td> 0.012</td> <td>-1.99e+04</td> <td>-2469.363</td>
</tr>
<tr>
  <th>Model_Ram Pickup 1500</th>                                         <td>-1.239e+04</td> <td> 3792.530</td> <td>   -3.266</td> <td> 0.001</td> <td>-1.98e+04</td> <td>-4952.867</td>
</tr>
<tr>
  <th>Model_Ram Van</th>                                                 <td> -2.76e+04</td> <td> 4541.865</td> <td>   -6.078</td> <td> 0.000</td> <td>-3.65e+04</td> <td>-1.87e+04</td>
</tr>
<tr>
  <th>Model_Ram Wagon</th>                                               <td>-1.645e+04</td> <td> 5171.735</td> <td>   -3.181</td> <td> 0.001</td> <td>-2.66e+04</td> <td>-6313.249</td>
</tr>
<tr>
  <th>Model_Ramcharger</th>                                              <td>-1.518e+04</td> <td> 5847.886</td> <td>   -2.596</td> <td> 0.009</td> <td>-2.66e+04</td> <td>-3718.005</td>
</tr>
<tr>
  <th>Model_Range Rover</th>                                             <td>  3.16e+04</td> <td> 1974.910</td> <td>   16.003</td> <td> 0.000</td> <td> 2.77e+04</td> <td> 3.55e+04</td>
</tr>
<tr>
  <th>Model_Range Rover Evoque</th>                                      <td>-7069.1429</td> <td> 1919.384</td> <td>   -3.683</td> <td> 0.000</td> <td>-1.08e+04</td> <td>-3306.774</td>
</tr>
<tr>
  <th>Model_Range Rover Sport</th>                                       <td>-1904.5533</td> <td> 1939.169</td> <td>   -0.982</td> <td> 0.326</td> <td>-5705.705</td> <td> 1896.599</td>
</tr>
<tr>
  <th>Model_Ranger</th>                                                  <td> -311.4846</td> <td> 1685.089</td> <td>   -0.185</td> <td> 0.853</td> <td>-3614.589</td> <td> 2991.620</td>
</tr>
<tr>
  <th>Model_Rapide</th>                                                  <td> 1.027e+04</td> <td> 3554.144</td> <td>    2.889</td> <td> 0.004</td> <td> 3302.327</td> <td> 1.72e+04</td>
</tr>
<tr>
  <th>Model_Rapide S</th>                                                <td>-1.781e+04</td> <td> 4066.298</td> <td>   -4.381</td> <td> 0.000</td> <td>-2.58e+04</td> <td>-9842.198</td>
</tr>
<tr>
  <th>Model_Reatta</th>                                                  <td>  -1.8e+04</td> <td> 3663.000</td> <td>   -4.914</td> <td> 0.000</td> <td>-2.52e+04</td> <td>-1.08e+04</td>
</tr>
<tr>
  <th>Model_Regal</th>                                                   <td>-3517.9792</td> <td> 2279.214</td> <td>   -1.544</td> <td> 0.123</td> <td>-7985.686</td> <td>  949.727</td>
</tr>
<tr>
  <th>Model_Regency</th>                                                 <td>-1.564e+04</td> <td> 4881.554</td> <td>   -3.203</td> <td> 0.001</td> <td>-2.52e+04</td> <td>-6067.976</td>
</tr>
<tr>
  <th>Model_Rendezvous</th>                                              <td>-3242.3175</td> <td> 2899.424</td> <td>   -1.118</td> <td> 0.263</td> <td>-8925.757</td> <td> 2441.122</td>
</tr>
<tr>
  <th>Model_Reno</th>                                                    <td>-8240.2347</td> <td> 1996.303</td> <td>   -4.128</td> <td> 0.000</td> <td>-1.22e+04</td> <td>-4327.089</td>
</tr>
<tr>
  <th>Model_Reventon</th>                                                <td> 1.023e+06</td> <td> 6277.591</td> <td>  162.915</td> <td> 0.000</td> <td> 1.01e+06</td> <td> 1.04e+06</td>
</tr>
<tr>
  <th>Model_Ridgeline</th>                                               <td>-1.102e+04</td> <td> 2181.819</td> <td>   -5.052</td> <td> 0.000</td> <td>-1.53e+04</td> <td>-6744.891</td>
</tr>
<tr>
  <th>Model_Rio</th>                                                     <td>-1.309e+04</td> <td> 1787.698</td> <td>   -7.322</td> <td> 0.000</td> <td>-1.66e+04</td> <td>-9584.748</td>
</tr>
<tr>
  <th>Model_Riviera</th>                                                 <td>-2.132e+04</td> <td> 3578.051</td> <td>   -5.959</td> <td> 0.000</td> <td>-2.83e+04</td> <td>-1.43e+04</td>
</tr>
<tr>
  <th>Model_Roadmaster</th>                                              <td> -2.26e+04</td> <td> 2837.458</td> <td>   -7.965</td> <td> 0.000</td> <td>-2.82e+04</td> <td> -1.7e+04</td>
</tr>
<tr>
  <th>Model_Rogue</th>                                                   <td> 1554.7892</td> <td> 2229.232</td> <td>    0.697</td> <td> 0.486</td> <td>-2814.941</td> <td> 5924.520</td>
</tr>
<tr>
  <th>Model_Rogue Select</th>                                            <td>-2910.9433</td> <td> 3816.582</td> <td>   -0.763</td> <td> 0.446</td> <td>-1.04e+04</td> <td> 4570.306</td>
</tr>
<tr>
  <th>Model_Rondo</th>                                                   <td>-6158.9060</td> <td> 1997.234</td> <td>   -3.084</td> <td> 0.002</td> <td>-1.01e+04</td> <td>-2243.937</td>
</tr>
<tr>
  <th>Model_Routan</th>                                                  <td> 1105.6252</td> <td> 6623.958</td> <td>    0.167</td> <td> 0.867</td> <td>-1.19e+04</td> <td> 1.41e+04</td>
</tr>
<tr>
  <th>Model_S-10</th>                                                    <td>-1.197e+04</td> <td> 2001.302</td> <td>   -5.979</td> <td> 0.000</td> <td>-1.59e+04</td> <td>-8043.581</td>
</tr>
<tr>
  <th>Model_S-10 Blazer</th>                                             <td>-1.935e+04</td> <td> 3129.798</td> <td>   -6.181</td> <td> 0.000</td> <td>-2.55e+04</td> <td>-1.32e+04</td>
</tr>
<tr>
  <th>Model_S-15</th>                                                    <td> 1.387e+04</td> <td> 4229.162</td> <td>    3.279</td> <td> 0.001</td> <td> 5575.353</td> <td> 2.22e+04</td>
</tr>
<tr>
  <th>Model_S-15 Jimmy</th>                                              <td> 1.598e+04</td> <td> 3889.237</td> <td>    4.108</td> <td> 0.000</td> <td> 8354.973</td> <td> 2.36e+04</td>
</tr>
<tr>
  <th>Model_S-Class</th>                                                 <td> 4.323e+04</td> <td> 1894.892</td> <td>   22.816</td> <td> 0.000</td> <td> 3.95e+04</td> <td> 4.69e+04</td>
</tr>
<tr>
  <th>Model_S2000</th>                                                   <td> 5904.5165</td> <td> 2845.525</td> <td>    2.075</td> <td> 0.038</td> <td>  326.729</td> <td> 1.15e+04</td>
</tr>
<tr>
  <th>Model_S3</th>                                                      <td> 4987.6116</td> <td> 4306.613</td> <td>    1.158</td> <td> 0.247</td> <td>-3454.193</td> <td> 1.34e+04</td>
</tr>
<tr>
  <th>Model_S4</th>                                                      <td> 9840.5744</td> <td> 3707.649</td> <td>    2.654</td> <td> 0.008</td> <td> 2572.856</td> <td> 1.71e+04</td>
</tr>
<tr>
  <th>Model_S40</th>                                                     <td>-1809.8580</td> <td> 8054.125</td> <td>   -0.225</td> <td> 0.822</td> <td>-1.76e+04</td> <td>  1.4e+04</td>
</tr>
<tr>
  <th>Model_S5</th>                                                      <td> 1.134e+04</td> <td> 3970.103</td> <td>    2.857</td> <td> 0.004</td> <td> 3558.621</td> <td> 1.91e+04</td>
</tr>
<tr>
  <th>Model_S6</th>                                                      <td> 9617.8437</td> <td> 4491.461</td> <td>    2.141</td> <td> 0.032</td> <td>  813.700</td> <td> 1.84e+04</td>
</tr>
<tr>
  <th>Model_S60</th>                                                     <td> 2548.4420</td> <td> 7767.467</td> <td>    0.328</td> <td> 0.743</td> <td>-1.27e+04</td> <td> 1.78e+04</td>
</tr>
<tr>
  <th>Model_S60 Cross Country</th>                                       <td> 7025.9078</td> <td> 1.05e+04</td> <td>    0.672</td> <td> 0.502</td> <td>-1.35e+04</td> <td> 2.75e+04</td>
</tr>
<tr>
  <th>Model_S7</th>                                                      <td> 1.655e+04</td> <td> 4778.958</td> <td>    3.463</td> <td> 0.001</td> <td> 7181.699</td> <td> 2.59e+04</td>
</tr>
<tr>
  <th>Model_S70</th>                                                     <td>-1.986e+04</td> <td> 7743.799</td> <td>   -2.564</td> <td> 0.010</td> <td> -3.5e+04</td> <td>-4676.059</td>
</tr>
<tr>
  <th>Model_S8</th>                                                      <td> 2.728e+04</td> <td> 4777.932</td> <td>    5.710</td> <td> 0.000</td> <td> 1.79e+04</td> <td> 3.66e+04</td>
</tr>
<tr>
  <th>Model_S80</th>                                                     <td> 6456.4802</td> <td> 8040.393</td> <td>    0.803</td> <td> 0.422</td> <td>-9304.264</td> <td> 2.22e+04</td>
</tr>
<tr>
  <th>Model_S90</th>                                                     <td> 3894.0124</td> <td> 8249.331</td> <td>    0.472</td> <td> 0.637</td> <td>-1.23e+04</td> <td> 2.01e+04</td>
</tr>
<tr>
  <th>Model_SC 300</th>                                                  <td> -3.31e+04</td> <td> 4200.621</td> <td>   -7.880</td> <td> 0.000</td> <td>-4.13e+04</td> <td>-2.49e+04</td>
</tr>
<tr>
  <th>Model_SC 400</th>                                                  <td>-3.833e+04</td> <td> 4199.452</td> <td>   -9.127</td> <td> 0.000</td> <td>-4.66e+04</td> <td>-3.01e+04</td>
</tr>
<tr>
  <th>Model_SC 430</th>                                                  <td> 8262.4433</td> <td> 4167.462</td> <td>    1.983</td> <td> 0.047</td> <td>   93.403</td> <td> 1.64e+04</td>
</tr>
<tr>
  <th>Model_SL-Class</th>                                                <td> 2.948e+04</td> <td> 2375.544</td> <td>   12.409</td> <td> 0.000</td> <td> 2.48e+04</td> <td> 3.41e+04</td>
</tr>
<tr>
  <th>Model_SLC-Class</th>                                               <td>-2.224e+04</td> <td> 5109.162</td> <td>   -4.354</td> <td> 0.000</td> <td>-3.23e+04</td> <td>-1.22e+04</td>
</tr>
<tr>
  <th>Model_SLK-Class</th>                                               <td>-1.988e+04</td> <td> 2621.874</td> <td>   -7.581</td> <td> 0.000</td> <td> -2.5e+04</td> <td>-1.47e+04</td>
</tr>
<tr>
  <th>Model_SLR McLaren</th>                                             <td> 3.589e+05</td> <td> 4195.964</td> <td>   85.535</td> <td> 0.000</td> <td> 3.51e+05</td> <td> 3.67e+05</td>
</tr>
<tr>
  <th>Model_SLS AMG</th>                                                 <td> 7.185e+04</td> <td> 4191.832</td> <td>   17.141</td> <td> 0.000</td> <td> 6.36e+04</td> <td> 8.01e+04</td>
</tr>
<tr>
  <th>Model_SLS AMG GT</th>                                              <td> 9.076e+04</td> <td> 3372.872</td> <td>   26.907</td> <td> 0.000</td> <td> 8.41e+04</td> <td> 9.74e+04</td>
</tr>
<tr>
  <th>Model_SLS AMG GT Final Edition</th>                                <td> 9.795e+04</td> <td> 5106.940</td> <td>   19.179</td> <td> 0.000</td> <td> 8.79e+04</td> <td> 1.08e+05</td>
</tr>
<tr>
  <th>Model_SLX</th>                                                     <td>-3.199e+04</td> <td> 3558.009</td> <td>   -8.990</td> <td> 0.000</td> <td> -3.9e+04</td> <td> -2.5e+04</td>
</tr>
<tr>
  <th>Model_SQ5</th>                                                     <td> 9976.0264</td> <td> 4133.944</td> <td>    2.413</td> <td> 0.016</td> <td> 1872.687</td> <td> 1.81e+04</td>
</tr>
<tr>
  <th>Model_SRT Viper</th>                                               <td> -319.8561</td> <td> 5603.341</td> <td>   -0.057</td> <td> 0.954</td> <td>-1.13e+04</td> <td> 1.07e+04</td>
</tr>
<tr>
  <th>Model_SRX</th>                                                     <td>-3932.5893</td> <td> 1751.713</td> <td>   -2.245</td> <td> 0.025</td> <td>-7366.290</td> <td> -498.889</td>
</tr>
<tr>
  <th>Model_SS</th>                                                      <td>-1.388e+04</td> <td> 4328.610</td> <td>   -3.206</td> <td> 0.001</td> <td>-2.24e+04</td> <td>-5391.483</td>
</tr>
<tr>
  <th>Model_SSR</th>                                                     <td>-6005.8802</td> <td> 4362.049</td> <td>   -1.377</td> <td> 0.169</td> <td>-1.46e+04</td> <td> 2544.590</td>
</tr>
<tr>
  <th>Model_STS</th>                                                     <td> 8844.8959</td> <td> 2322.667</td> <td>    3.808</td> <td> 0.000</td> <td> 4292.013</td> <td> 1.34e+04</td>
</tr>
<tr>
  <th>Model_STS-V</th>                                                   <td> 1.071e+04</td> <td> 4086.957</td> <td>    2.621</td> <td> 0.009</td> <td> 2700.470</td> <td> 1.87e+04</td>
</tr>
<tr>
  <th>Model_SVX</th>                                                     <td>-1.681e+04</td> <td> 2755.824</td> <td>   -6.101</td> <td> 0.000</td> <td>-2.22e+04</td> <td>-1.14e+04</td>
</tr>
<tr>
  <th>Model_SX4</th>                                                     <td>-8137.7835</td> <td> 1621.759</td> <td>   -5.018</td> <td> 0.000</td> <td>-1.13e+04</td> <td>-4958.818</td>
</tr>
<tr>
  <th>Model_Safari</th>                                                  <td> 2.912e+04</td> <td> 5306.020</td> <td>    5.487</td> <td> 0.000</td> <td> 1.87e+04</td> <td> 3.95e+04</td>
</tr>
<tr>
  <th>Model_Safari Cargo</th>                                            <td> 3.068e+04</td> <td> 5517.567</td> <td>    5.560</td> <td> 0.000</td> <td> 1.99e+04</td> <td> 4.15e+04</td>
</tr>
<tr>
  <th>Model_Samurai</th>                                                 <td>-1.634e+04</td> <td> 3895.736</td> <td>   -4.194</td> <td> 0.000</td> <td> -2.4e+04</td> <td>-8703.192</td>
</tr>
<tr>
  <th>Model_Santa Fe</th>                                                <td>-2916.6799</td> <td> 1854.053</td> <td>   -1.573</td> <td> 0.116</td> <td>-6550.987</td> <td>  717.628</td>
</tr>
<tr>
  <th>Model_Santa Fe Sport</th>                                          <td>-2611.7869</td> <td> 1963.983</td> <td>   -1.330</td> <td> 0.184</td> <td>-6461.577</td> <td> 1238.004</td>
</tr>
<tr>
  <th>Model_Savana</th>                                                  <td> 1.646e+04</td> <td> 4815.000</td> <td>    3.419</td> <td> 0.001</td> <td> 7022.706</td> <td> 2.59e+04</td>
</tr>
<tr>
  <th>Model_Savana Cargo</th>                                            <td> 1.379e+04</td> <td> 4788.430</td> <td>    2.880</td> <td> 0.004</td> <td> 4406.014</td> <td> 2.32e+04</td>
</tr>
<tr>
  <th>Model_Scoupe</th>                                                  <td>-1.043e+04</td> <td> 3143.813</td> <td>   -3.317</td> <td> 0.001</td> <td>-1.66e+04</td> <td>-4264.800</td>
</tr>
<tr>
  <th>Model_Sebring</th>                                                 <td>-8766.3300</td> <td> 3647.681</td> <td>   -2.403</td> <td> 0.016</td> <td>-1.59e+04</td> <td>-1616.160</td>
</tr>
<tr>
  <th>Model_Sedona</th>                                                  <td> -502.0923</td> <td> 4143.195</td> <td>   -0.121</td> <td> 0.904</td> <td>-8623.567</td> <td> 7619.382</td>
</tr>
<tr>
  <th>Model_Sentra</th>                                                  <td>-2336.3333</td> <td> 2446.492</td> <td>   -0.955</td> <td> 0.340</td> <td>-7131.937</td> <td> 2459.270</td>
</tr>
<tr>
  <th>Model_Sephia</th>                                                  <td>-9144.6454</td> <td> 2980.684</td> <td>   -3.068</td> <td> 0.002</td> <td> -1.5e+04</td> <td>-3301.921</td>
</tr>
<tr>
  <th>Model_Sequoia</th>                                                 <td> 5962.6236</td> <td> 1791.976</td> <td>    3.327</td> <td> 0.001</td> <td> 2449.999</td> <td> 9475.248</td>
</tr>
<tr>
  <th>Model_Seville</th>                                                 <td> 7551.0906</td> <td> 3237.314</td> <td>    2.333</td> <td> 0.020</td> <td> 1205.322</td> <td> 1.39e+04</td>
</tr>
<tr>
  <th>Model_Shadow</th>                                                  <td>-1.274e+04</td> <td> 5014.385</td> <td>   -2.540</td> <td> 0.011</td> <td>-2.26e+04</td> <td>-2908.801</td>
</tr>
<tr>
  <th>Model_Shelby GT350</th>                                            <td>-7641.9247</td> <td> 3156.927</td> <td>   -2.421</td> <td> 0.016</td> <td>-1.38e+04</td> <td>-1453.730</td>
</tr>
<tr>
  <th>Model_Shelby GT500</th>                                            <td>-3.998e+04</td> <td> 3483.549</td> <td>  -11.477</td> <td> 0.000</td> <td>-4.68e+04</td> <td>-3.32e+04</td>
</tr>
<tr>
  <th>Model_Sidekick</th>                                                <td>-1.606e+04</td> <td> 1732.997</td> <td>   -9.266</td> <td> 0.000</td> <td>-1.95e+04</td> <td>-1.27e+04</td>
</tr>
<tr>
  <th>Model_Sienna</th>                                                  <td> 4094.2276</td> <td> 4159.211</td> <td>    0.984</td> <td> 0.325</td> <td>-4058.641</td> <td> 1.22e+04</td>
</tr>
<tr>
  <th>Model_Sierra 1500</th>                                             <td> 2.027e+04</td> <td> 3325.383</td> <td>    6.097</td> <td> 0.000</td> <td> 1.38e+04</td> <td> 2.68e+04</td>
</tr>
<tr>
  <th>Model_Sierra 1500 Classic</th>                                     <td> 1.677e+04</td> <td> 2936.011</td> <td>    5.711</td> <td> 0.000</td> <td>  1.1e+04</td> <td> 2.25e+04</td>
</tr>
<tr>
  <th>Model_Sierra 1500 Hybrid</th>                                      <td> 2.343e+04</td> <td> 3644.550</td> <td>    6.430</td> <td> 0.000</td> <td> 1.63e+04</td> <td> 3.06e+04</td>
</tr>
<tr>
  <th>Model_Sierra 1500HD</th>                                           <td> 1.819e+04</td> <td> 3931.248</td> <td>    4.626</td> <td> 0.000</td> <td> 1.05e+04</td> <td> 2.59e+04</td>
</tr>
<tr>
  <th>Model_Sierra C3</th>                                               <td> 2.615e+04</td> <td> 7429.015</td> <td>    3.520</td> <td> 0.000</td> <td> 1.16e+04</td> <td> 4.07e+04</td>
</tr>
<tr>
  <th>Model_Sierra Classic 1500</th>                                     <td> 2873.2521</td> <td> 4270.203</td> <td>    0.673</td> <td> 0.501</td> <td>-5497.182</td> <td> 1.12e+04</td>
</tr>
<tr>
  <th>Model_Sigma</th>                                                   <td>-8541.2755</td> <td> 7001.327</td> <td>   -1.220</td> <td> 0.223</td> <td>-2.23e+04</td> <td> 5182.697</td>
</tr>
<tr>
  <th>Model_Silhouette</th>                                              <td> 1.284e+04</td> <td> 4143.129</td> <td>    3.100</td> <td> 0.002</td> <td> 4722.274</td> <td>  2.1e+04</td>
</tr>
<tr>
  <th>Model_Silver Seraph</th>                                           <td>-7.215e+04</td> <td> 4821.699</td> <td>  -14.964</td> <td> 0.000</td> <td>-8.16e+04</td> <td>-6.27e+04</td>
</tr>
<tr>
  <th>Model_Silverado 1500</th>                                          <td>-1.411e+04</td> <td> 1782.018</td> <td>   -7.920</td> <td> 0.000</td> <td>-1.76e+04</td> <td>-1.06e+04</td>
</tr>
<tr>
  <th>Model_Silverado 1500 Classic</th>                                  <td>-1.688e+04</td> <td> 1784.263</td> <td>   -9.463</td> <td> 0.000</td> <td>-2.04e+04</td> <td>-1.34e+04</td>
</tr>
<tr>
  <th>Model_Silverado 1500 Hybrid</th>                                   <td>-1.058e+04</td> <td> 2584.297</td> <td>   -4.092</td> <td> 0.000</td> <td>-1.56e+04</td> <td>-5509.714</td>
</tr>
<tr>
  <th>Model_Sixty Special</th>                                           <td>-2.741e+04</td> <td> 7085.070</td> <td>   -3.869</td> <td> 0.000</td> <td>-4.13e+04</td> <td>-1.35e+04</td>
</tr>
<tr>
  <th>Model_Skylark</th>                                                 <td> -1.67e+04</td> <td> 2752.404</td> <td>   -6.066</td> <td> 0.000</td> <td>-2.21e+04</td> <td>-1.13e+04</td>
</tr>
<tr>
  <th>Model_Solstice</th>                                                <td>-3207.2485</td> <td> 2502.009</td> <td>   -1.282</td> <td> 0.200</td> <td>-8111.677</td> <td> 1697.180</td>
</tr>
<tr>
  <th>Model_Sonata</th>                                                  <td>-2631.7164</td> <td> 1824.247</td> <td>   -1.443</td> <td> 0.149</td> <td>-6207.598</td> <td>  944.166</td>
</tr>
<tr>
  <th>Model_Sonata Hybrid</th>                                           <td> 1230.6205</td> <td> 3062.960</td> <td>    0.402</td> <td> 0.688</td> <td>-4773.380</td> <td> 7234.621</td>
</tr>
<tr>
  <th>Model_Sonic</th>                                                   <td>-1.714e+04</td> <td> 1791.524</td> <td>   -9.566</td> <td> 0.000</td> <td>-2.06e+04</td> <td>-1.36e+04</td>
</tr>
<tr>
  <th>Model_Sonoma</th>                                                  <td> 2.029e+04</td> <td> 3154.740</td> <td>    6.431</td> <td> 0.000</td> <td> 1.41e+04</td> <td> 2.65e+04</td>
</tr>
<tr>
  <th>Model_Sorento</th>                                                 <td>-2785.7455</td> <td> 1326.785</td> <td>   -2.100</td> <td> 0.036</td> <td>-5386.503</td> <td> -184.988</td>
</tr>
<tr>
  <th>Model_Soul</th>                                                    <td>-1.113e+04</td> <td> 2204.278</td> <td>   -5.048</td> <td> 0.000</td> <td>-1.54e+04</td> <td>-6805.778</td>
</tr>
<tr>
  <th>Model_Soul EV</th>                                                 <td>-4489.2907</td> <td> 5363.485</td> <td>   -0.837</td> <td> 0.403</td> <td> -1.5e+04</td> <td> 6024.190</td>
</tr>
<tr>
  <th>Model_Spark</th>                                                   <td>-2.312e+04</td> <td> 2295.416</td> <td>  -10.071</td> <td> 0.000</td> <td>-2.76e+04</td> <td>-1.86e+04</td>
</tr>
<tr>
  <th>Model_Spark EV</th>                                                <td>-1.717e+04</td> <td> 5523.384</td> <td>   -3.109</td> <td> 0.002</td> <td> -2.8e+04</td> <td>-6344.650</td>
</tr>
<tr>
  <th>Model_Spectra</th>                                                 <td>-7514.7144</td> <td> 1633.146</td> <td>   -4.601</td> <td> 0.000</td> <td>-1.07e+04</td> <td>-4313.429</td>
</tr>
<tr>
  <th>Model_Spirit</th>                                                  <td>-1.228e+04</td> <td> 5871.612</td> <td>   -2.091</td> <td> 0.037</td> <td>-2.38e+04</td> <td> -765.650</td>
</tr>
<tr>
  <th>Model_Sportage</th>                                                <td>-6712.4776</td> <td> 1820.623</td> <td>   -3.687</td> <td> 0.000</td> <td>-1.03e+04</td> <td>-3143.699</td>
</tr>
<tr>
  <th>Model_Sportvan</th>                                                <td> -3.05e+04</td> <td> 4502.495</td> <td>   -6.774</td> <td> 0.000</td> <td>-3.93e+04</td> <td>-2.17e+04</td>
</tr>
<tr>
  <th>Model_Spyder</th>                                                  <td>-1.549e+04</td> <td> 6153.534</td> <td>   -2.517</td> <td> 0.012</td> <td>-2.75e+04</td> <td>-3425.288</td>
</tr>
<tr>
  <th>Model_Stanza</th>                                                  <td> -904.8963</td> <td> 3886.231</td> <td>   -0.233</td> <td> 0.816</td> <td>-8522.670</td> <td> 6712.877</td>
</tr>
<tr>
  <th>Model_Stealth</th>                                                 <td>-1.387e+04</td> <td> 5162.647</td> <td>   -2.687</td> <td> 0.007</td> <td> -2.4e+04</td> <td>-3754.707</td>
</tr>
<tr>
  <th>Model_Stratus</th>                                                 <td>  485.3919</td> <td> 4685.037</td> <td>    0.104</td> <td> 0.917</td> <td>-8698.198</td> <td> 9668.981</td>
</tr>
<tr>
  <th>Model_Suburban</th>                                                <td> 3966.9170</td> <td> 1704.826</td> <td>    2.327</td> <td> 0.020</td> <td>  625.124</td> <td> 7308.710</td>
</tr>
<tr>
  <th>Model_Sunbird</th>                                                 <td>-1.375e+04</td> <td> 2272.058</td> <td>   -6.053</td> <td> 0.000</td> <td>-1.82e+04</td> <td>-9298.968</td>
</tr>
<tr>
  <th>Model_Sundance</th>                                                <td>-7654.4349</td> <td> 4724.893</td> <td>   -1.620</td> <td> 0.105</td> <td>-1.69e+04</td> <td> 1607.281</td>
</tr>
<tr>
  <th>Model_Sunfire</th>                                                 <td>-6995.1226</td> <td> 3269.093</td> <td>   -2.140</td> <td> 0.032</td> <td>-1.34e+04</td> <td> -587.060</td>
</tr>
<tr>
  <th>Model_Superamerica</th>                                            <td> 2.422e+04</td> <td> 4866.398</td> <td>    4.976</td> <td> 0.000</td> <td> 1.47e+04</td> <td> 3.38e+04</td>
</tr>
<tr>
  <th>Model_Supersports Convertible ISR</th>                             <td>-4.176e+04</td> <td> 7442.525</td> <td>   -5.612</td> <td> 0.000</td> <td>-5.64e+04</td> <td>-2.72e+04</td>
</tr>
<tr>
  <th>Model_Supra</th>                                                   <td> 3463.6133</td> <td> 3167.607</td> <td>    1.093</td> <td> 0.274</td> <td>-2745.516</td> <td> 9672.743</td>
</tr>
<tr>
  <th>Model_Swift</th>                                                   <td>-1.266e+04</td> <td> 2940.974</td> <td>   -4.304</td> <td> 0.000</td> <td>-1.84e+04</td> <td>-6892.867</td>
</tr>
<tr>
  <th>Model_Syclone</th>                                                 <td> 1.048e+04</td> <td> 7389.569</td> <td>    1.418</td> <td> 0.156</td> <td>-4003.726</td> <td>  2.5e+04</td>
</tr>
<tr>
  <th>Model_T100</th>                                                    <td>-1.677e+04</td> <td> 2170.160</td> <td>   -7.726</td> <td> 0.000</td> <td> -2.1e+04</td> <td>-1.25e+04</td>
</tr>
<tr>
  <th>Model_TC</th>                                                      <td>-2.456e+04</td> <td> 5721.628</td> <td>   -4.292</td> <td> 0.000</td> <td>-3.58e+04</td> <td>-1.33e+04</td>
</tr>
<tr>
  <th>Model_TL</th>                                                      <td>-5893.7952</td> <td> 1714.588</td> <td>   -3.437</td> <td> 0.001</td> <td>-9254.723</td> <td>-2532.867</td>
</tr>
<tr>
  <th>Model_TLX</th>                                                     <td>-9083.9467</td> <td> 1871.007</td> <td>   -4.855</td> <td> 0.000</td> <td>-1.28e+04</td> <td>-5416.407</td>
</tr>
<tr>
  <th>Model_TSX</th>                                                     <td>-7318.5437</td> <td> 1948.468</td> <td>   -3.756</td> <td> 0.000</td> <td>-1.11e+04</td> <td>-3499.165</td>
</tr>
<tr>
  <th>Model_TSX Sport Wagon</th>                                         <td>-6745.2858</td> <td> 3014.661</td> <td>   -2.237</td> <td> 0.025</td> <td>-1.27e+04</td> <td> -835.959</td>
</tr>
<tr>
  <th>Model_TT</th>                                                      <td> 2963.5479</td> <td> 4287.950</td> <td>    0.691</td> <td> 0.489</td> <td>-5441.674</td> <td> 1.14e+04</td>
</tr>
<tr>
  <th>Model_TT RS</th>                                                   <td> 1.735e+04</td> <td> 5880.296</td> <td>    2.950</td> <td> 0.003</td> <td> 5819.930</td> <td> 2.89e+04</td>
</tr>
<tr>
  <th>Model_TTS</th>                                                     <td>  1.02e+04</td> <td> 4774.494</td> <td>    2.136</td> <td> 0.033</td> <td>  840.047</td> <td> 1.96e+04</td>
</tr>
<tr>
  <th>Model_Tacoma</th>                                                  <td>-8871.6472</td> <td> 1786.167</td> <td>   -4.967</td> <td> 0.000</td> <td>-1.24e+04</td> <td>-5370.410</td>
</tr>
<tr>
  <th>Model_Tahoe</th>                                                   <td> 1297.5840</td> <td> 2176.609</td> <td>    0.596</td> <td> 0.551</td> <td>-2968.995</td> <td> 5564.163</td>
</tr>
<tr>
  <th>Model_Tahoe Hybrid</th>                                            <td> 4860.8629</td> <td> 3236.552</td> <td>    1.502</td> <td> 0.133</td> <td>-1483.412</td> <td> 1.12e+04</td>
</tr>
<tr>
  <th>Model_Tahoe Limited/Z71</th>                                       <td>-3.139e+04</td> <td> 5355.067</td> <td>   -5.861</td> <td> 0.000</td> <td>-4.19e+04</td> <td>-2.09e+04</td>
</tr>
<tr>
  <th>Model_Taurus</th>                                                  <td> 3552.5668</td> <td> 2009.682</td> <td>    1.768</td> <td> 0.077</td> <td> -386.804</td> <td> 7491.938</td>
</tr>
<tr>
  <th>Model_Taurus X</th>                                                <td> 7902.9242</td> <td> 2353.857</td> <td>    3.357</td> <td> 0.001</td> <td> 3288.903</td> <td> 1.25e+04</td>
</tr>
<tr>
  <th>Model_Tempo</th>                                                   <td>-2663.5800</td> <td> 3006.554</td> <td>   -0.886</td> <td> 0.376</td> <td>-8557.014</td> <td> 3229.854</td>
</tr>
<tr>
  <th>Model_Tercel</th>                                                  <td>-1.118e+04</td> <td> 2939.075</td> <td>   -3.805</td> <td> 0.000</td> <td>-1.69e+04</td> <td>-5422.791</td>
</tr>
<tr>
  <th>Model_Terrain</th>                                                 <td> 2.549e+04</td> <td> 3114.212</td> <td>    8.187</td> <td> 0.000</td> <td> 1.94e+04</td> <td> 3.16e+04</td>
</tr>
<tr>
  <th>Model_Terraza</th>                                                 <td> 2596.5818</td> <td> 4288.599</td> <td>    0.605</td> <td> 0.545</td> <td>-5809.913</td> <td>  1.1e+04</td>
</tr>
<tr>
  <th>Model_Thunderbird</th>                                             <td> 1.125e+04</td> <td> 2813.556</td> <td>    3.997</td> <td> 0.000</td> <td> 5730.050</td> <td> 1.68e+04</td>
</tr>
<tr>
  <th>Model_Tiburon</th>                                                 <td>-4526.8179</td> <td> 1853.820</td> <td>   -2.442</td> <td> 0.015</td> <td>-8160.668</td> <td> -892.968</td>
</tr>
<tr>
  <th>Model_Tiguan</th>                                                  <td>-7255.2987</td> <td> 5737.694</td> <td>   -1.264</td> <td> 0.206</td> <td>-1.85e+04</td> <td> 3991.706</td>
</tr>
<tr>
  <th>Model_Titan</th>                                                   <td>-6838.4376</td> <td> 2645.306</td> <td>   -2.585</td> <td> 0.010</td> <td> -1.2e+04</td> <td>-1653.120</td>
</tr>
<tr>
  <th>Model_Toronado</th>                                                <td>-1.084e+04</td> <td> 4037.385</td> <td>   -2.684</td> <td> 0.007</td> <td>-1.88e+04</td> <td>-2922.016</td>
</tr>
<tr>
  <th>Model_Torrent</th>                                                 <td>-5093.0362</td> <td> 2437.502</td> <td>   -2.089</td> <td> 0.037</td> <td>-9871.017</td> <td> -315.055</td>
</tr>
<tr>
  <th>Model_Touareg</th>                                                 <td> 8560.5889</td> <td> 5787.862</td> <td>    1.479</td> <td> 0.139</td> <td>-2784.754</td> <td> 1.99e+04</td>
</tr>
<tr>
  <th>Model_Touareg 2</th>                                               <td> 3747.5116</td> <td> 6124.145</td> <td>    0.612</td> <td> 0.541</td> <td>-8257.013</td> <td> 1.58e+04</td>
</tr>
<tr>
  <th>Model_Town Car</th>                                                <td> 5.392e+04</td> <td> 3317.034</td> <td>   16.257</td> <td> 0.000</td> <td> 4.74e+04</td> <td> 6.04e+04</td>
</tr>
<tr>
  <th>Model_Town and Country</th>                                        <td>-3991.5621</td> <td> 5271.696</td> <td>   -0.757</td> <td> 0.449</td> <td>-1.43e+04</td> <td> 6341.994</td>
</tr>
<tr>
  <th>Model_Tracker</th>                                                 <td>-1.043e+04</td> <td> 2413.813</td> <td>   -4.323</td> <td> 0.000</td> <td>-1.52e+04</td> <td>-5702.525</td>
</tr>
<tr>
  <th>Model_TrailBlazer</th>                                             <td>-1.103e+04</td> <td> 1968.636</td> <td>   -5.602</td> <td> 0.000</td> <td>-1.49e+04</td> <td>-7168.998</td>
</tr>
<tr>
  <th>Model_TrailBlazer EXT</th>                                         <td>-7273.8578</td> <td> 2659.639</td> <td>   -2.735</td> <td> 0.006</td> <td>-1.25e+04</td> <td>-2060.445</td>
</tr>
<tr>
  <th>Model_Trans Sport</th>                                             <td>-1.585e+04</td> <td> 4397.099</td> <td>   -3.605</td> <td> 0.000</td> <td>-2.45e+04</td> <td>-7230.582</td>
</tr>
<tr>
  <th>Model_Transit Connect</th>                                         <td> 6729.0682</td> <td> 3859.820</td> <td>    1.743</td> <td> 0.081</td> <td> -836.935</td> <td> 1.43e+04</td>
</tr>
<tr>
  <th>Model_Transit Wagon</th>                                           <td> 1956.6327</td> <td> 3402.060</td> <td>    0.575</td> <td> 0.565</td> <td>-4712.071</td> <td> 8625.336</td>
</tr>
<tr>
  <th>Model_Traverse</th>                                                <td>-7836.3791</td> <td> 1907.681</td> <td>   -4.108</td> <td> 0.000</td> <td>-1.16e+04</td> <td>-4096.951</td>
</tr>
<tr>
  <th>Model_Trax</th>                                                    <td>-1.588e+04</td> <td> 2158.291</td> <td>   -7.355</td> <td> 0.000</td> <td>-2.01e+04</td> <td>-1.16e+04</td>
</tr>
<tr>
  <th>Model_Tribeca</th>                                                 <td>-2037.5493</td> <td> 3113.260</td> <td>   -0.654</td> <td> 0.513</td> <td>-8140.149</td> <td> 4065.050</td>
</tr>
<tr>
  <th>Model_Tribute</th>                                                 <td>-6904.8873</td> <td> 1882.421</td> <td>   -3.668</td> <td> 0.000</td> <td>-1.06e+04</td> <td>-3214.973</td>
</tr>
<tr>
  <th>Model_Tribute Hybrid</th>                                          <td> -136.6361</td> <td> 2928.529</td> <td>   -0.047</td> <td> 0.963</td> <td>-5877.126</td> <td> 5603.854</td>
</tr>
<tr>
  <th>Model_Truck</th>                                                   <td>-9119.4585</td> <td> 1516.283</td> <td>   -6.014</td> <td> 0.000</td> <td>-1.21e+04</td> <td>-6147.247</td>
</tr>
<tr>
  <th>Model_Tucson</th>                                                  <td>-8014.5433</td> <td> 1695.435</td> <td>   -4.727</td> <td> 0.000</td> <td>-1.13e+04</td> <td>-4691.158</td>
</tr>
<tr>
  <th>Model_Tundra</th>                                                  <td>-1.602e+04</td> <td> 1741.079</td> <td>   -9.203</td> <td> 0.000</td> <td>-1.94e+04</td> <td>-1.26e+04</td>
</tr>
<tr>
  <th>Model_Typhoon</th>                                                 <td> 1.335e+04</td> <td> 5729.522</td> <td>    2.330</td> <td> 0.020</td> <td> 2117.196</td> <td> 2.46e+04</td>
</tr>
<tr>
  <th>Model_Uplander</th>                                                <td>-4627.6946</td> <td> 4183.113</td> <td>   -1.106</td> <td> 0.269</td> <td>-1.28e+04</td> <td> 3572.026</td>
</tr>
<tr>
  <th>Model_V12 Vanquish</th>                                            <td> 4.478e+04</td> <td> 3471.997</td> <td>   12.897</td> <td> 0.000</td> <td>  3.8e+04</td> <td> 5.16e+04</td>
</tr>
<tr>
  <th>Model_V12 Vantage</th>                                             <td>-1.796e+04</td> <td> 3492.722</td> <td>   -5.141</td> <td> 0.000</td> <td>-2.48e+04</td> <td>-1.11e+04</td>
</tr>
<tr>
  <th>Model_V12 Vantage S</th>                                           <td>-4.166e+04</td> <td> 4045.337</td> <td>  -10.298</td> <td> 0.000</td> <td>-4.96e+04</td> <td>-3.37e+04</td>
</tr>
<tr>
  <th>Model_V40</th>                                                     <td>   58.6781</td> <td> 8387.084</td> <td>    0.007</td> <td> 0.994</td> <td>-1.64e+04</td> <td> 1.65e+04</td>
</tr>
<tr>
  <th>Model_V50</th>                                                     <td>-1551.8789</td> <td> 8181.495</td> <td>   -0.190</td> <td> 0.850</td> <td>-1.76e+04</td> <td> 1.45e+04</td>
</tr>
<tr>
  <th>Model_V60</th>                                                     <td> 3231.6013</td> <td> 7809.118</td> <td>    0.414</td> <td> 0.679</td> <td>-1.21e+04</td> <td> 1.85e+04</td>
</tr>
<tr>
  <th>Model_V60 Cross Country</th>                                       <td> 4866.2712</td> <td> 8187.169</td> <td>    0.594</td> <td> 0.552</td> <td>-1.12e+04</td> <td> 2.09e+04</td>
</tr>
<tr>
  <th>Model_V70</th>                                                     <td>  388.5196</td> <td> 8368.821</td> <td>    0.046</td> <td> 0.963</td> <td> -1.6e+04</td> <td> 1.68e+04</td>
</tr>
<tr>
  <th>Model_V8</th>                                                      <td>-1.998e+04</td> <td> 4887.987</td> <td>   -4.087</td> <td> 0.000</td> <td>-2.96e+04</td> <td>-1.04e+04</td>
</tr>
<tr>
  <th>Model_V8 Vantage</th>                                              <td>-6.463e+04</td> <td> 1810.445</td> <td>  -35.699</td> <td> 0.000</td> <td>-6.82e+04</td> <td>-6.11e+04</td>
</tr>
<tr>
  <th>Model_V90</th>                                                     <td>-2.063e+04</td> <td> 1.03e+04</td> <td>   -1.999</td> <td> 0.046</td> <td>-4.09e+04</td> <td> -401.551</td>
</tr>
<tr>
  <th>Model_Van</th>                                                     <td>  270.3994</td> <td> 6496.831</td> <td>    0.042</td> <td> 0.967</td> <td>-1.25e+04</td> <td>  1.3e+04</td>
</tr>
<tr>
  <th>Model_Vanagon</th>                                                 <td>-1.508e+04</td> <td> 7516.603</td> <td>   -2.006</td> <td> 0.045</td> <td>-2.98e+04</td> <td> -341.199</td>
</tr>
<tr>
  <th>Model_Vandura</th>                                                 <td> 4129.5787</td> <td> 4342.870</td> <td>    0.951</td> <td> 0.342</td> <td>-4383.298</td> <td> 1.26e+04</td>
</tr>
<tr>
  <th>Model_Vanquish</th>                                                <td> 7.024e+04</td> <td> 2650.993</td> <td>   26.495</td> <td> 0.000</td> <td>  6.5e+04</td> <td> 7.54e+04</td>
</tr>
<tr>
  <th>Model_Vanwagon</th>                                                <td>-4315.3589</td> <td> 6515.683</td> <td>   -0.662</td> <td> 0.508</td> <td>-1.71e+04</td> <td> 8456.655</td>
</tr>
<tr>
  <th>Model_Veloster</th>                                                <td>-9483.5280</td> <td> 2037.625</td> <td>   -4.654</td> <td> 0.000</td> <td>-1.35e+04</td> <td>-5489.383</td>
</tr>
<tr>
  <th>Model_Venture</th>                                                 <td> -702.6295</td> <td> 4190.675</td> <td>   -0.168</td> <td> 0.867</td> <td>-8917.174</td> <td> 7511.914</td>
</tr>
<tr>
  <th>Model_Venza</th>                                                   <td> 3942.6831</td> <td> 1735.417</td> <td>    2.272</td> <td> 0.023</td> <td>  540.926</td> <td> 7344.440</td>
</tr>
<tr>
  <th>Model_Veracruz</th>                                                <td>  -87.8289</td> <td> 2068.162</td> <td>   -0.042</td> <td> 0.966</td> <td>-4141.831</td> <td> 3966.173</td>
</tr>
<tr>
  <th>Model_Verano</th>                                                  <td>-7289.8133</td> <td> 2358.712</td> <td>   -3.091</td> <td> 0.002</td> <td>-1.19e+04</td> <td>-2666.276</td>
</tr>
<tr>
  <th>Model_Verona</th>                                                  <td>-2698.1187</td> <td> 2434.873</td> <td>   -1.108</td> <td> 0.268</td> <td>-7470.946</td> <td> 2074.709</td>
</tr>
<tr>
  <th>Model_Versa</th>                                                   <td>-8079.5541</td> <td> 2546.041</td> <td>   -3.173</td> <td> 0.002</td> <td>-1.31e+04</td> <td>-3088.814</td>
</tr>
<tr>
  <th>Model_Versa Note</th>                                              <td>-7301.2561</td> <td> 2598.885</td> <td>   -2.809</td> <td> 0.005</td> <td>-1.24e+04</td> <td>-2206.933</td>
</tr>
<tr>
  <th>Model_Veyron 16.4</th>                                             <td> 6.922e+05</td> <td> 3637.595</td> <td>  190.291</td> <td> 0.000</td> <td> 6.85e+05</td> <td> 6.99e+05</td>
</tr>
<tr>
  <th>Model_Vibe</th>                                                    <td>-7911.7489</td> <td> 2688.313</td> <td>   -2.943</td> <td> 0.003</td> <td>-1.32e+04</td> <td>-2642.129</td>
</tr>
<tr>
  <th>Model_Vigor</th>                                                   <td> -2.14e+04</td> <td> 4082.171</td> <td>   -5.242</td> <td> 0.000</td> <td>-2.94e+04</td> <td>-1.34e+04</td>
</tr>
<tr>
  <th>Model_Viper</th>                                                   <td>-5478.9419</td> <td> 4818.341</td> <td>   -1.137</td> <td> 0.256</td> <td>-1.49e+04</td> <td> 3965.950</td>
</tr>
<tr>
  <th>Model_Virage</th>                                                  <td> 9369.8515</td> <td> 4787.657</td> <td>    1.957</td> <td> 0.050</td> <td>  -14.894</td> <td> 1.88e+04</td>
</tr>
<tr>
  <th>Model_Vitara</th>                                                  <td>-4263.7013</td> <td> 1875.648</td> <td>   -2.273</td> <td> 0.023</td> <td>-7940.339</td> <td> -587.064</td>
</tr>
<tr>
  <th>Model_Voyager</th>                                                 <td>-7322.7481</td> <td> 5714.577</td> <td>   -1.281</td> <td> 0.200</td> <td>-1.85e+04</td> <td> 3878.942</td>
</tr>
<tr>
  <th>Model_WRX</th>                                                     <td>-1881.8088</td> <td> 1870.918</td> <td>   -1.006</td> <td> 0.315</td> <td>-5549.174</td> <td> 1785.556</td>
</tr>
<tr>
  <th>Model_Windstar</th>                                                <td> 1.642e+04</td> <td> 4133.749</td> <td>    3.972</td> <td> 0.000</td> <td> 8314.357</td> <td> 2.45e+04</td>
</tr>
<tr>
  <th>Model_Windstar Cargo</th>                                          <td> 1.228e+04</td> <td> 5765.948</td> <td>    2.129</td> <td> 0.033</td> <td>  974.800</td> <td> 2.36e+04</td>
</tr>
<tr>
  <th>Model_Wraith</th>                                                  <td>-7.142e+04</td> <td> 4148.676</td> <td>  -17.216</td> <td> 0.000</td> <td>-7.96e+04</td> <td>-6.33e+04</td>
</tr>
<tr>
  <th>Model_X-90</th>                                                    <td>-1.416e+04</td> <td> 3179.579</td> <td>   -4.454</td> <td> 0.000</td> <td>-2.04e+04</td> <td>-7930.280</td>
</tr>
<tr>
  <th>Model_X1</th>                                                      <td>-1.882e+04</td> <td> 3641.154</td> <td>   -5.168</td> <td> 0.000</td> <td> -2.6e+04</td> <td>-1.17e+04</td>
</tr>
<tr>
  <th>Model_X3</th>                                                      <td>-1.226e+04</td> <td> 3023.142</td> <td>   -4.054</td> <td> 0.000</td> <td>-1.82e+04</td> <td>-6329.807</td>
</tr>
<tr>
  <th>Model_X4</th>                                                      <td>-8311.1132</td> <td> 3477.253</td> <td>   -2.390</td> <td> 0.017</td> <td>-1.51e+04</td> <td>-1495.017</td>
</tr>
<tr>
  <th>Model_X5</th>                                                      <td>-3476.6013</td> <td> 3014.214</td> <td>   -1.153</td> <td> 0.249</td> <td>-9385.051</td> <td> 2431.849</td>
</tr>
<tr>
  <th>Model_X5 M</th>                                                    <td>-7987.7850</td> <td> 4673.345</td> <td>   -1.709</td> <td> 0.087</td> <td>-1.71e+04</td> <td> 1172.887</td>
</tr>
<tr>
  <th>Model_X6</th>                                                      <td>  817.6930</td> <td> 3231.291</td> <td>    0.253</td> <td> 0.800</td> <td>-5516.270</td> <td> 7151.656</td>
</tr>
<tr>
  <th>Model_X6 M</th>                                                    <td>-4587.7850</td> <td> 4673.345</td> <td>   -0.982</td> <td> 0.326</td> <td>-1.37e+04</td> <td> 4572.887</td>
</tr>
<tr>
  <th>Model_XC</th>                                                      <td> 8400.7091</td> <td> 1.03e+04</td> <td>    0.812</td> <td> 0.417</td> <td>-1.19e+04</td> <td> 2.87e+04</td>
</tr>
<tr>
  <th>Model_XC60</th>                                                    <td> 2038.5189</td> <td> 7723.952</td> <td>    0.264</td> <td> 0.792</td> <td>-1.31e+04</td> <td> 1.72e+04</td>
</tr>
<tr>
  <th>Model_XC70</th>                                                    <td> 3116.4594</td> <td> 7825.969</td> <td>    0.398</td> <td> 0.690</td> <td>-1.22e+04</td> <td> 1.85e+04</td>
</tr>
<tr>
  <th>Model_XC90</th>                                                    <td> 7137.6710</td> <td> 7883.778</td> <td>    0.905</td> <td> 0.365</td> <td>-8316.078</td> <td> 2.26e+04</td>
</tr>
<tr>
  <th>Model_XG300</th>                                                   <td> 4931.3553</td> <td> 4934.727</td> <td>    0.999</td> <td> 0.318</td> <td>-4741.676</td> <td> 1.46e+04</td>
</tr>
<tr>
  <th>Model_XG350</th>                                                   <td> 3433.5025</td> <td> 2933.017</td> <td>    1.171</td> <td> 0.242</td> <td>-2315.785</td> <td> 9182.790</td>
</tr>
<tr>
  <th>Model_XL-7</th>                                                    <td> -780.9299</td> <td> 1469.636</td> <td>   -0.531</td> <td> 0.595</td> <td>-3661.704</td> <td> 2099.844</td>
</tr>
<tr>
  <th>Model_XL7</th>                                                     <td>-1952.5671</td> <td> 1394.892</td> <td>   -1.400</td> <td> 0.162</td> <td>-4686.828</td> <td>  781.694</td>
</tr>
<tr>
  <th>Model_XLR</th>                                                     <td> 3.031e+04</td> <td> 2820.946</td> <td>   10.744</td> <td> 0.000</td> <td> 2.48e+04</td> <td> 3.58e+04</td>
</tr>
<tr>
  <th>Model_XLR-V</th>                                                   <td> 3.092e+04</td> <td> 3631.263</td> <td>    8.516</td> <td> 0.000</td> <td> 2.38e+04</td> <td>  3.8e+04</td>
</tr>
<tr>
  <th>Model_XT</th>                                                      <td>-1.106e+04</td> <td> 4093.781</td> <td>   -2.702</td> <td> 0.007</td> <td>-1.91e+04</td> <td>-3037.430</td>
</tr>
<tr>
  <th>Model_XT5</th>                                                     <td>-2273.7220</td> <td> 3004.936</td> <td>   -0.757</td> <td> 0.449</td> <td>-8163.985</td> <td> 3616.541</td>
</tr>
<tr>
  <th>Model_XTS</th>                                                     <td> 5534.1500</td> <td> 1657.966</td> <td>    3.338</td> <td> 0.001</td> <td> 2284.212</td> <td> 8784.088</td>
</tr>
<tr>
  <th>Model_XV Crosstrek</th>                                            <td>-7724.6162</td> <td> 2031.091</td> <td>   -3.803</td> <td> 0.000</td> <td>-1.17e+04</td> <td>-3743.279</td>
</tr>
<tr>
  <th>Model_Xterra</th>                                                  <td> -567.9202</td> <td> 2178.321</td> <td>   -0.261</td> <td> 0.794</td> <td>-4837.856</td> <td> 3702.016</td>
</tr>
<tr>
  <th>Model_Yaris</th>                                                   <td>-1.296e+04</td> <td> 1850.016</td> <td>   -7.005</td> <td> 0.000</td> <td>-1.66e+04</td> <td>-9332.543</td>
</tr>
<tr>
  <th>Model_Yaris iA</th>                                                <td>-1.169e+04</td> <td> 5034.211</td> <td>   -2.322</td> <td> 0.020</td> <td>-2.16e+04</td> <td>-1823.275</td>
</tr>
<tr>
  <th>Model_Yukon</th>                                                   <td> 3.457e+04</td> <td> 3295.285</td> <td>   10.492</td> <td> 0.000</td> <td> 2.81e+04</td> <td>  4.1e+04</td>
</tr>
<tr>
  <th>Model_Yukon Denali</th>                                            <td> 1454.0864</td> <td> 7457.491</td> <td>    0.195</td> <td> 0.845</td> <td>-1.32e+04</td> <td> 1.61e+04</td>
</tr>
<tr>
  <th>Model_Yukon Hybrid</th>                                            <td> 4.281e+04</td> <td> 3375.931</td> <td>   12.681</td> <td> 0.000</td> <td> 3.62e+04</td> <td> 4.94e+04</td>
</tr>
<tr>
  <th>Model_Yukon XL</th>                                                <td> 3.725e+04</td> <td> 3294.053</td> <td>   11.307</td> <td> 0.000</td> <td> 3.08e+04</td> <td> 4.37e+04</td>
</tr>
<tr>
  <th>Model_Z3</th>                                                      <td>-1.747e+04</td> <td> 3133.168</td> <td>   -5.575</td> <td> 0.000</td> <td>-2.36e+04</td> <td>-1.13e+04</td>
</tr>
<tr>
  <th>Model_Z4</th>                                                      <td>-4503.2799</td> <td> 3193.990</td> <td>   -1.410</td> <td> 0.159</td> <td>-1.08e+04</td> <td> 1757.567</td>
</tr>
<tr>
  <th>Model_Z4 M</th>                                                    <td>-1692.8037</td> <td> 4056.365</td> <td>   -0.417</td> <td> 0.676</td> <td>-9644.074</td> <td> 6258.467</td>
</tr>
<tr>
  <th>Model_Z8</th>                                                      <td>  6.67e+04</td> <td> 4558.143</td> <td>   14.633</td> <td> 0.000</td> <td> 5.78e+04</td> <td> 7.56e+04</td>
</tr>
<tr>
  <th>Model_ZDX</th>                                                     <td> 2208.2458</td> <td> 2807.982</td> <td>    0.786</td> <td> 0.432</td> <td>-3295.950</td> <td> 7712.441</td>
</tr>
<tr>
  <th>Model_Zephyr</th>                                                  <td> 3.972e+04</td> <td> 7287.768</td> <td>    5.450</td> <td> 0.000</td> <td> 2.54e+04</td> <td>  5.4e+04</td>
</tr>
<tr>
  <th>Model_allroad</th>                                                 <td> 1.209e+04</td> <td> 3974.600</td> <td>    3.042</td> <td> 0.002</td> <td> 4300.671</td> <td> 1.99e+04</td>
</tr>
<tr>
  <th>Model_allroad quattro</th>                                         <td> 1.398e+04</td> <td> 3716.910</td> <td>    3.762</td> <td> 0.000</td> <td> 6696.257</td> <td> 2.13e+04</td>
</tr>
<tr>
  <th>Model_e-Golf</th>                                                  <td>-1.016e+04</td> <td> 8074.348</td> <td>   -1.258</td> <td> 0.208</td> <td> -2.6e+04</td> <td> 5668.882</td>
</tr>
<tr>
  <th>Model_i-MiEV</th>                                                  <td>-1.446e+04</td> <td> 6069.106</td> <td>   -2.383</td> <td> 0.017</td> <td>-2.64e+04</td> <td>-2565.293</td>
</tr>
<tr>
  <th>Model_i3</th>                                                      <td>-1.153e+04</td> <td> 6359.136</td> <td>   -1.814</td> <td> 0.070</td> <td> -2.4e+04</td> <td>  930.301</td>
</tr>
<tr>
  <th>Model_iA</th>                                                      <td>-1.921e+04</td> <td> 4713.862</td> <td>   -4.074</td> <td> 0.000</td> <td>-2.84e+04</td> <td>-9964.967</td>
</tr>
<tr>
  <th>Model_iM</th>                                                      <td>-1.701e+04</td> <td> 4699.879</td> <td>   -3.618</td> <td> 0.000</td> <td>-2.62e+04</td> <td>-7792.638</td>
</tr>
<tr>
  <th>Model_iQ</th>                                                      <td>-2.076e+04</td> <td> 3519.450</td> <td>   -5.898</td> <td> 0.000</td> <td>-2.77e+04</td> <td>-1.39e+04</td>
</tr>
<tr>
  <th>Model_tC</th>                                                      <td>-1.309e+04</td> <td> 2280.205</td> <td>   -5.742</td> <td> 0.000</td> <td>-1.76e+04</td> <td>-8623.135</td>
</tr>
<tr>
  <th>Model_xA</th>                                                      <td>-1.607e+04</td> <td> 2826.893</td> <td>   -5.684</td> <td> 0.000</td> <td>-2.16e+04</td> <td>-1.05e+04</td>
</tr>
<tr>
  <th>Model_xB</th>                                                      <td>-1.502e+04</td> <td> 2533.439</td> <td>   -5.930</td> <td> 0.000</td> <td>   -2e+04</td> <td>-1.01e+04</td>
</tr>
<tr>
  <th>Model_xD</th>                                                      <td>-1.716e+04</td> <td> 2403.513</td> <td>   -7.138</td> <td> 0.000</td> <td>-2.19e+04</td> <td>-1.24e+04</td>
</tr>
<tr>
  <th>Engine Fuel Type_diesel</th>                                       <td>-1.311e+05</td> <td> 1.58e+04</td> <td>   -8.307</td> <td> 0.000</td> <td>-1.62e+05</td> <td>   -1e+05</td>
</tr>
<tr>
  <th>Engine Fuel Type_electric</th>                                     <td>-1.211e+05</td> <td> 1.69e+04</td> <td>   -7.172</td> <td> 0.000</td> <td>-1.54e+05</td> <td> -8.8e+04</td>
</tr>
<tr>
  <th>Engine Fuel Type_flex-fuel (premium unleaded recommended/E85)</th> <td>-1.354e+05</td> <td>  1.6e+04</td> <td>   -8.448</td> <td> 0.000</td> <td>-1.67e+05</td> <td>-1.04e+05</td>
</tr>
<tr>
  <th>Engine Fuel Type_flex-fuel (premium unleaded required/E85)</th>    <td>-1.375e+05</td> <td> 1.58e+04</td> <td>   -8.686</td> <td> 0.000</td> <td>-1.69e+05</td> <td>-1.06e+05</td>
</tr>
<tr>
  <th>Engine Fuel Type_flex-fuel (unleaded/E85)</th>                     <td>-1.361e+05</td> <td> 1.58e+04</td> <td>   -8.608</td> <td> 0.000</td> <td>-1.67e+05</td> <td>-1.05e+05</td>
</tr>
<tr>
  <th>Engine Fuel Type_flex-fuel (unleaded/natural gas)</th>             <td>-1.321e+05</td> <td> 1.62e+04</td> <td>   -8.155</td> <td> 0.000</td> <td>-1.64e+05</td> <td>   -1e+05</td>
</tr>
<tr>
  <th>Engine Fuel Type_natural gas</th>                                  <td>-1.311e+05</td> <td> 1.65e+04</td> <td>   -7.954</td> <td> 0.000</td> <td>-1.63e+05</td> <td>-9.88e+04</td>
</tr>
<tr>
  <th>Engine Fuel Type_premium unleaded (recommended)</th>               <td>-1.342e+05</td> <td> 1.58e+04</td> <td>   -8.481</td> <td> 0.000</td> <td>-1.65e+05</td> <td>-1.03e+05</td>
</tr>
<tr>
  <th>Engine Fuel Type_premium unleaded (required)</th>                  <td>-1.346e+05</td> <td> 1.58e+04</td> <td>   -8.523</td> <td> 0.000</td> <td>-1.66e+05</td> <td>-1.04e+05</td>
</tr>
<tr>
  <th>Engine Fuel Type_regular unleaded</th>                             <td>-1.353e+05</td> <td> 1.58e+04</td> <td>   -8.561</td> <td> 0.000</td> <td>-1.66e+05</td> <td>-1.04e+05</td>
</tr>
<tr>
  <th>Transmission Type_AUTOMATED_MANUAL</th>                            <td>-3.297e+05</td> <td> 3.94e+04</td> <td>   -8.378</td> <td> 0.000</td> <td>-4.07e+05</td> <td>-2.53e+05</td>
</tr>
<tr>
  <th>Transmission Type_AUTOMATIC</th>                                   <td>-3.337e+05</td> <td> 3.94e+04</td> <td>   -8.480</td> <td> 0.000</td> <td>-4.11e+05</td> <td>-2.57e+05</td>
</tr>
<tr>
  <th>Transmission Type_DIRECT_DRIVE</th>                                <td>-3.297e+05</td> <td> 3.95e+04</td> <td>   -8.343</td> <td> 0.000</td> <td>-4.07e+05</td> <td>-2.52e+05</td>
</tr>
<tr>
  <th>Transmission Type_MANUAL</th>                                      <td>-3.354e+05</td> <td> 3.93e+04</td> <td>   -8.524</td> <td> 0.000</td> <td>-4.12e+05</td> <td>-2.58e+05</td>
</tr>
<tr>
  <th>Driven_Wheels_all wheel drive</th>                                 <td>-3.309e+05</td> <td> 3.93e+04</td> <td>   -8.414</td> <td> 0.000</td> <td>-4.08e+05</td> <td>-2.54e+05</td>
</tr>
<tr>
  <th>Driven_Wheels_four wheel drive</th>                                <td>-3.311e+05</td> <td> 3.93e+04</td> <td>   -8.423</td> <td> 0.000</td> <td>-4.08e+05</td> <td>-2.54e+05</td>
</tr>
<tr>
  <th>Driven_Wheels_front wheel drive</th>                               <td>-3.329e+05</td> <td> 3.93e+04</td> <td>   -8.465</td> <td> 0.000</td> <td> -4.1e+05</td> <td>-2.56e+05</td>
</tr>
<tr>
  <th>Driven_Wheels_rear wheel drive</th>                                <td>-3.336e+05</td> <td> 3.93e+04</td> <td>   -8.487</td> <td> 0.000</td> <td>-4.11e+05</td> <td>-2.57e+05</td>
</tr>
<tr>
  <th>Vehicle Size_Compact</th>                                          <td>-4.432e+05</td> <td> 5.24e+04</td> <td>   -8.461</td> <td> 0.000</td> <td>-5.46e+05</td> <td>-3.41e+05</td>
</tr>
<tr>
  <th>Vehicle Size_Large</th>                                            <td>-4.417e+05</td> <td> 5.25e+04</td> <td>   -8.419</td> <td> 0.000</td> <td>-5.45e+05</td> <td>-3.39e+05</td>
</tr>
<tr>
  <th>Vehicle Size_Midsize</th>                                          <td>-4.436e+05</td> <td> 5.24e+04</td> <td>   -8.461</td> <td> 0.000</td> <td>-5.46e+05</td> <td>-3.41e+05</td>
</tr>
<tr>
  <th>Vehicle Style_2dr Hatchback</th>                                   <td>-8.375e+04</td> <td> 1.01e+04</td> <td>   -8.310</td> <td> 0.000</td> <td>-1.04e+05</td> <td> -6.4e+04</td>
</tr>
<tr>
  <th>Vehicle Style_2dr SUV</th>                                         <td>-8.441e+04</td> <td> 1.04e+04</td> <td>   -8.128</td> <td> 0.000</td> <td>-1.05e+05</td> <td>-6.41e+04</td>
</tr>
<tr>
  <th>Vehicle Style_4dr Hatchback</th>                                   <td>-8.582e+04</td> <td> 1.01e+04</td> <td>   -8.527</td> <td> 0.000</td> <td>-1.06e+05</td> <td>-6.61e+04</td>
</tr>
<tr>
  <th>Vehicle Style_4dr SUV</th>                                         <td>-8.456e+04</td> <td> 1.03e+04</td> <td>   -8.249</td> <td> 0.000</td> <td>-1.05e+05</td> <td>-6.45e+04</td>
</tr>
<tr>
  <th>Vehicle Style_Cargo Minivan</th>                                   <td>-9.065e+04</td> <td> 1.14e+04</td> <td>   -7.965</td> <td> 0.000</td> <td>-1.13e+05</td> <td>-6.83e+04</td>
</tr>
<tr>
  <th>Vehicle Style_Cargo Van</th>                                       <td>-7.634e+04</td> <td> 8939.359</td> <td>   -8.539</td> <td> 0.000</td> <td>-9.39e+04</td> <td>-5.88e+04</td>
</tr>
<tr>
  <th>Vehicle Style_Convertible</th>                                     <td>-7.739e+04</td> <td> 1.01e+04</td> <td>   -7.683</td> <td> 0.000</td> <td>-9.71e+04</td> <td>-5.76e+04</td>
</tr>
<tr>
  <th>Vehicle Style_Convertible SUV</th>                                 <td>-8.242e+04</td> <td> 1.04e+04</td> <td>   -7.931</td> <td> 0.000</td> <td>-1.03e+05</td> <td>-6.21e+04</td>
</tr>
<tr>
  <th>Vehicle Style_Coupe</th>                                           <td>-8.549e+04</td> <td> 1.01e+04</td> <td>   -8.496</td> <td> 0.000</td> <td>-1.05e+05</td> <td>-6.58e+04</td>
</tr>
<tr>
  <th>Vehicle Style_Crew Cab Pickup</th>                                 <td>-7.797e+04</td> <td> 9324.862</td> <td>   -8.362</td> <td> 0.000</td> <td>-9.63e+04</td> <td>-5.97e+04</td>
</tr>
<tr>
  <th>Vehicle Style_Extended Cab Pickup</th>                             <td> -8.08e+04</td> <td> 9318.610</td> <td>   -8.671</td> <td> 0.000</td> <td>-9.91e+04</td> <td>-6.25e+04</td>
</tr>
<tr>
  <th>Vehicle Style_Passenger Minivan</th>                               <td>-8.786e+04</td> <td> 1.13e+04</td> <td>   -7.789</td> <td> 0.000</td> <td> -1.1e+05</td> <td>-6.58e+04</td>
</tr>
<tr>
  <th>Vehicle Style_Passenger Van</th>                                   <td> -7.63e+04</td> <td> 8989.535</td> <td>   -8.488</td> <td> 0.000</td> <td>-9.39e+04</td> <td>-5.87e+04</td>
</tr>
<tr>
  <th>Vehicle Style_Regular Cab Pickup</th>                              <td>-8.086e+04</td> <td> 9333.654</td> <td>   -8.663</td> <td> 0.000</td> <td>-9.92e+04</td> <td>-6.26e+04</td>
</tr>
<tr>
  <th>Vehicle Style_Sedan</th>                                           <td>-8.728e+04</td> <td>    1e+04</td> <td>   -8.697</td> <td> 0.000</td> <td>-1.07e+05</td> <td>-6.76e+04</td>
</tr>
<tr>
  <th>Vehicle Style_Wagon</th>                                           <td>-8.661e+04</td> <td>    1e+04</td> <td>   -8.619</td> <td> 0.000</td> <td>-1.06e+05</td> <td>-6.69e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>12341.470</td> <th>  Durbin-Watson:     </th>   <td>   2.356</td>   
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>148226190.597</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 4.098</td>   <th>  Prob(JB):          </th>   <td>    0.00</td>   
</tr>
<tr>
  <th>Kurtosis:</th>       <td>566.550</td>  <th>  Cond. No.          </th>   <td>1.26e+22</td>   
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 7.9e-31. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.



# Plotting true msrp vs the predicted msrp to evalate 


```python
X = df[['Driven_Wheels_all wheel drive','Driven_Wheels_front wheel drive',
        'Driven_Wheels_rear wheel drive','Engine Cylinders']].values
y = df['MSRP'].values

model =RandomForestRegressor()
model.fit(X, y)

y_pred = model.predict(X)
```


```python
df['y_pred'] = y_pred
sns.lmplot(x='MSRP', y='y_pred', data=df)
```




    <seaborn.axisgrid.FacetGrid at 0x1a19232b70>




![png](testcarprediction_files/testcarprediction_202_1.png)


# Future interests
Other info


```python
#Get examples from ebay and see if I can predict price or correct price. 
# Against my price? 
# Create another model for outliers 
# Classification for normal cars vs outliers(determine higher prices)
```
