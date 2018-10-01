<br>

```python
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

%matplotlib inline
```


```python
df = pd.read_csv('../Customer_segment/int_online_tx.csv')
```

# EXAMINE DATA


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
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>536370</td>
      <td>22728</td>
      <td>ALARM CLOCK BAKELIKE PINK</td>
      <td>24</td>
      <td>12/1/10 8:45</td>
      <td>3.75</td>
      <td>12583.0</td>
      <td>France</td>
    </tr>
    <tr>
      <th>1</th>
      <td>536370</td>
      <td>22727</td>
      <td>ALARM CLOCK BAKELIKE RED</td>
      <td>24</td>
      <td>12/1/10 8:45</td>
      <td>3.75</td>
      <td>12583.0</td>
      <td>France</td>
    </tr>
    <tr>
      <th>2</th>
      <td>536370</td>
      <td>22726</td>
      <td>ALARM CLOCK BAKELIKE GREEN</td>
      <td>12</td>
      <td>12/1/10 8:45</td>
      <td>3.75</td>
      <td>12583.0</td>
      <td>France</td>
    </tr>
    <tr>
      <th>3</th>
      <td>536370</td>
      <td>21724</td>
      <td>PANDA AND BUNNIES STICKER SHEET</td>
      <td>12</td>
      <td>12/1/10 8:45</td>
      <td>0.85</td>
      <td>12583.0</td>
      <td>France</td>
    </tr>
    <tr>
      <th>4</th>
      <td>536370</td>
      <td>21883</td>
      <td>STARS GIFT TAPE</td>
      <td>24</td>
      <td>12/1/10 8:45</td>
      <td>0.65</td>
      <td>12583.0</td>
      <td>France</td>
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
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35111</th>
      <td>581587</td>
      <td>22613</td>
      <td>PACK OF 20 SPACEBOY NAPKINS</td>
      <td>12</td>
      <td>12/9/11 12:50</td>
      <td>0.85</td>
      <td>12680.0</td>
      <td>France</td>
    </tr>
    <tr>
      <th>35112</th>
      <td>581587</td>
      <td>22899</td>
      <td>CHILDREN'S APRON DOLLY GIRL</td>
      <td>6</td>
      <td>12/9/11 12:50</td>
      <td>2.10</td>
      <td>12680.0</td>
      <td>France</td>
    </tr>
    <tr>
      <th>35113</th>
      <td>581587</td>
      <td>23254</td>
      <td>CHILDRENS CUTLERY DOLLY GIRL</td>
      <td>4</td>
      <td>12/9/11 12:50</td>
      <td>4.15</td>
      <td>12680.0</td>
      <td>France</td>
    </tr>
    <tr>
      <th>35114</th>
      <td>581587</td>
      <td>23255</td>
      <td>CHILDRENS CUTLERY CIRCUS PARADE</td>
      <td>4</td>
      <td>12/9/11 12:50</td>
      <td>4.15</td>
      <td>12680.0</td>
      <td>France</td>
    </tr>
    <tr>
      <th>35115</th>
      <td>581587</td>
      <td>22138</td>
      <td>BAKING SET 9 PIECE RETROSPOT</td>
      <td>3</td>
      <td>12/9/11 12:50</td>
      <td>4.95</td>
      <td>12680.0</td>
      <td>France</td>
    </tr>
  </tbody>
</table>
</div>




```python
list(df.columns)
```




    ['InvoiceNo',
     'StockCode',
     'Description',
     'Quantity',
     'InvoiceDate',
     'UnitPrice',
     'CustomerID',
     'Country',
     'Sales']




```python
list(df['Country'].unique())
```




    ['France',
     'Australia',
     'Netherlands',
     'Germany',
     'Norway',
     'Switzerland',
     'EIRE',
     'Spain',
     'Poland',
     'Portugal',
     'Italy',
     'Belgium',
     'Lithuania',
     'Japan',
     'Iceland',
     'Channel Islands',
     'Denmark',
     'Cyprus',
     'Sweden',
     'Finland',
     'Austria',
     'Bahrain',
     'Israel',
     'Greece',
     'Hong Kong',
     'Singapore',
     'Lebanon',
     'United Arab Emirates',
     'Saudi Arabia',
     'Czech Republic',
     'Canada',
     'Unspecified',
     'Brazil',
     'USA',
     'European Community',
     'Malta',
     'RSA']




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 35116 entries, 0 to 35115
    Data columns (total 8 columns):
    InvoiceNo      35116 non-null int64
    StockCode      35116 non-null object
    Description    35116 non-null object
    Quantity       35116 non-null int64
    InvoiceDate    35116 non-null object
    UnitPrice      35116 non-null float64
    CustomerID     33698 non-null float64
    Country        35116 non-null object
    dtypes: float64(2), int64(2), object(4)
    memory usage: 2.1+ MB



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
      <th>InvoiceNo</th>
      <th>Quantity</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>35116.000000</td>
      <td>35116.000000</td>
      <td>35116.000000</td>
      <td>33698.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>559940.650273</td>
      <td>14.624302</td>
      <td>4.700512</td>
      <td>12793.819188</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12645.318619</td>
      <td>31.144229</td>
      <td>51.807988</td>
      <td>828.171434</td>
    </tr>
    <tr>
      <th>min</th>
      <td>536370.000000</td>
      <td>1.000000</td>
      <td>0.040000</td>
      <td>12347.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>548737.000000</td>
      <td>5.000000</td>
      <td>1.250000</td>
      <td>12473.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>561037.000000</td>
      <td>10.000000</td>
      <td>1.950000</td>
      <td>12597.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>570672.000000</td>
      <td>12.000000</td>
      <td>3.750000</td>
      <td>12708.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>581587.000000</td>
      <td>2040.000000</td>
      <td>4161.060000</td>
      <td>17844.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes
```




    InvoiceNo        int64
    StockCode       object
    Description     object
    Quantity         int64
    InvoiceDate     object
    UnitPrice      float64
    CustomerID     float64
    Country         object
    dtype: object




```python
df.shape
```




    (35116, 8)




```python
df.isnull().sum()
```




    InvoiceNo         0
    StockCode         0
    Description       0
    Quantity          0
    InvoiceDate       0
    UnitPrice         0
    CustomerID     1418
    Country           0
    dtype: int64




```python
nullvalues = df.isnull().sum()
nullvalues.plot.barh()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x105316d30>




![png](/images/Customer_segments_files/Customer_segments_12_1.png)



```python
# Display distribution of transactions by country
plt.figure(figsize=(7,8))

sns.countplot(y ='Country', data=df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10542b5f8>




![png](/images/Customer_segments_files/Customer_segments_13_1.png)


# CLEANING DATA


```python
df.isnull().sum()
```




    InvoiceNo         0
    StockCode         0
    Description       0
    Quantity          0
    InvoiceDate       0
    UnitPrice         0
    CustomerID     1418
    Country           0
    dtype: int64




```python
# I will just keep transactions with customer ID's
df = df[df.CustomerID.notnull()]
```


```python
# Also will change CustomerID from floats into integers
df['CustomerID'] = df.CustomerID.astype(int)
```


```python
df.isnull().sum()
```




    InvoiceNo      0
    StockCode      0
    Description    0
    Quantity       0
    InvoiceDate    0
    UnitPrice      0
    CustomerID     0
    Country        0
    dtype: int64




```python
# check work
df.CustomerID.head()
```




    0    12583
    1    12583
    2    12583
    3    12583
    4    12583
    Name: CustomerID, dtype: int64




```python
# Create new feature for Sales
df['Sales'] = (df['Quantity'] * df['UnitPrice'])
```


```python
df.Sales.head()
```




    0    90.0
    1    90.0
    2    45.0
    3    10.2
    4    15.6
    Name: Sales, dtype: float64




```python
#Save Clean data
df.to_csv('Cleaned_transaction.csv', index=None)
```


```python
# Aggregrate invoice data
invoice_data = df.groupby('CustomerID').InvoiceNo.agg({'total_transactions': 'nunique'})

invoice_data.head()
```

    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:2: FutureWarning: using a dict on a Series for aggregation
    is deprecated and will be removed in a future version
      from ipykernel import kernelapp as app





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
      <th>total_transactions</th>
    </tr>
    <tr>
      <th>CustomerID</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12347</th>
      <td>7</td>
    </tr>
    <tr>
      <th>12348</th>
      <td>4</td>
    </tr>
    <tr>
      <th>12349</th>
      <td>1</td>
    </tr>
    <tr>
      <th>12350</th>
      <td>1</td>
    </tr>
    <tr>
      <th>12352</th>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>


