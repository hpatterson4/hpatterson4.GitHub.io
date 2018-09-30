<br>

<img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 20px; height: 55px">

# Simple Linear Regression with Sacramento Real Estate Data

_Authors: Matt Brems, Sam Stack, Justin Pounders_

---

In this lab you will hone your exploratory data analysis (EDA) skills and practice constructing simple linear regressions using a data set on Sacramento real estate sales.  The data set contains information on qualities of the property, location of the property, and time of sale.

### 1. Read in the Sacramento housing data set.


```python
sac_csv = './datasets/sacramento_real_estate_transactions.csv'
```


```python
type(sac_csv)
```




    str




```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model
import statsmodels.api as sm


% matplotlib inline 
#shows plotting in notebook
```


```python
# A: 
sc = pd.read_csv(sac_csv)
```


```python
type(sc)
```




    pandas.core.frame.DataFrame



### 2. Conduct exploratory data analysis on this data set. 

**Report any notable findings here and any steps you take to clean/process data.**

> **Note:** These EDA checks should be done on every data set you handle. If you find yourself checking repeatedly for missing/corrupted data, it might be beneficial to have a function that you can reuse every time you're given new data.


```python
# A:
sc.head(5)
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
      <th>street</th>
      <th>city</th>
      <th>zip</th>
      <th>state</th>
      <th>beds</th>
      <th>baths</th>
      <th>sq__ft</th>
      <th>type</th>
      <th>sale_date</th>
      <th>price</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3526 HIGH ST</td>
      <td>SACRAMENTO</td>
      <td>95838</td>
      <td>CA</td>
      <td>2</td>
      <td>1</td>
      <td>836</td>
      <td>Residential</td>
      <td>Wed May 21 00:00:00 EDT 2008</td>
      <td>59222</td>
      <td>38.631913</td>
      <td>-121.434879</td>
    </tr>
    <tr>
      <th>1</th>
      <td>51 OMAHA CT</td>
      <td>SACRAMENTO</td>
      <td>95823</td>
      <td>CA</td>
      <td>3</td>
      <td>1</td>
      <td>1167</td>
      <td>Residential</td>
      <td>Wed May 21 00:00:00 EDT 2008</td>
      <td>68212</td>
      <td>38.478902</td>
      <td>-121.431028</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2796 BRANCH ST</td>
      <td>SACRAMENTO</td>
      <td>95815</td>
      <td>CA</td>
      <td>2</td>
      <td>1</td>
      <td>796</td>
      <td>Residential</td>
      <td>Wed May 21 00:00:00 EDT 2008</td>
      <td>68880</td>
      <td>38.618305</td>
      <td>-121.443839</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2805 JANETTE WAY</td>
      <td>SACRAMENTO</td>
      <td>95815</td>
      <td>CA</td>
      <td>2</td>
      <td>1</td>
      <td>852</td>
      <td>Residential</td>
      <td>Wed May 21 00:00:00 EDT 2008</td>
      <td>69307</td>
      <td>38.616835</td>
      <td>-121.439146</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6001 MCMAHON DR</td>
      <td>SACRAMENTO</td>
      <td>95824</td>
      <td>CA</td>
      <td>2</td>
      <td>1</td>
      <td>797</td>
      <td>Residential</td>
      <td>Wed May 21 00:00:00 EDT 2008</td>
      <td>81900</td>
      <td>38.519470</td>
      <td>-121.435768</td>
    </tr>
  </tbody>
</table>
</div>




```python
# A:
type(sc.head(10))
```




    pandas.core.frame.DataFrame




```python
# check null values
sc.isnull().sum()
```




    street       0
    city         0
    zip          0
    state        0
    beds         0
    baths        0
    sq__ft       0
    type         0
    sale_date    0
    price        0
    latitude     0
    longitude    0
    dtype: int64




```python
# checking if the states match to CA
sc['state'].unique()
```




    array(['CA', 'AC'], dtype=object)




```python
# checking the types that are available 
sc['type'].unique()
```




    array(['Residential', 'Condo', 'Multi-Family', 'Unkown'], dtype=object)




```python
# checking datatypes, null values, columns and if all entries are there 
sc.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 985 entries, 0 to 984
    Data columns (total 12 columns):
    street       985 non-null object
    city         985 non-null object
    zip          985 non-null int64
    state        985 non-null object
    beds         985 non-null int64
    baths        985 non-null int64
    sq__ft       985 non-null int64
    type         985 non-null object
    sale_date    985 non-null object
    price        985 non-null int64
    latitude     985 non-null float64
    longitude    985 non-null float64
    dtypes: float64(2), int64(5), object(5)
    memory usage: 92.4+ KB



```python
# checking statistics for columns 
# notice that there is negative values for price, sq_ft
#seen that there are houses that have no bedrooms, or baths
sc.describe()
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
      <th>zip</th>
      <th>beds</th>
      <th>baths</th>
      <th>sq__ft</th>
      <th>price</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>985.000000</td>
      <td>985.000000</td>
      <td>985.000000</td>
      <td>985.000000</td>
      <td>985.000000</td>
      <td>985.000000</td>
      <td>985.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>95750.697462</td>
      <td>2.911675</td>
      <td>1.776650</td>
      <td>1312.918782</td>
      <td>233715.951269</td>
      <td>38.445121</td>
      <td>-121.193371</td>
    </tr>
    <tr>
      <th>std</th>
      <td>85.176072</td>
      <td>1.307932</td>
      <td>0.895371</td>
      <td>856.123224</td>
      <td>139088.818896</td>
      <td>5.103637</td>
      <td>5.100670</td>
    </tr>
    <tr>
      <th>min</th>
      <td>95603.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-984.000000</td>
      <td>-210944.000000</td>
      <td>-121.503471</td>
      <td>-121.551704</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>95660.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>950.000000</td>
      <td>145000.000000</td>
      <td>38.482704</td>
      <td>-121.446119</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>95762.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>1304.000000</td>
      <td>213750.000000</td>
      <td>38.625932</td>
      <td>-121.375799</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>95828.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>1718.000000</td>
      <td>300000.000000</td>
      <td>38.695589</td>
      <td>-121.294893</td>
    </tr>
    <tr>
      <th>max</th>
      <td>95864.000000</td>
      <td>8.000000</td>
      <td>5.000000</td>
      <td>5822.000000</td>
      <td>884790.000000</td>
      <td>39.020808</td>
      <td>38.668433</td>
    </tr>
  </tbody>
</table>
</div>




```python
# found row that have negative price, which also inclued negaitive Sq_ft and state is wrong 
sc[sc['price'] <0]
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
      <th>street</th>
      <th>city</th>
      <th>zip</th>
      <th>state</th>
      <th>beds</th>
      <th>baths</th>
      <th>sq__ft</th>
      <th>type</th>
      <th>sale_date</th>
      <th>price</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>703</th>
      <td>1900 DANBROOK DR</td>
      <td>SACRAMENTO</td>
      <td>95835</td>
      <td>AC</td>
      <td>1</td>
      <td>1</td>
      <td>-984</td>
      <td>Condo</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>-210944</td>
      <td>-121.503471</td>
      <td>38.668433</td>
    </tr>
  </tbody>
</table>
</div>




```python
# seeing how many values are negative 
(sc['price'] <0).value_counts()
```




    False    984
    True       1
    Name: price, dtype: int64




```python
# Decided to correct state, and take the negaive off te sq-ft and price 
# keeping the row
sc.loc[703,'state'] = 'CA'
```


```python
sc.loc[703,'sq__ft'] = 984
```


```python
sc.loc[703,'price'] = 210944
```


```python
# checking value to make sure changes was made
sc.loc[703]
```




    street                   1900 DANBROOK DR
    city                           SACRAMENTO
    zip                                 95835
    state                                  CA
    beds                                    1
    baths                                   1
    sq__ft                                984
    type                                Condo
    sale_date    Fri May 16 00:00:00 EDT 2008
    price                              210944
    latitude                         -121.503
    longitude                         38.6684
    Name: 703, dtype: object




```python
# checked to see how mand houses with no bedrooms
#looks like the houses with no bedrooms also have no baths and no sq_ft
#kept rows because they must be land available to build 
sc[sc['beds'] == 0]
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
      <th>street</th>
      <th>city</th>
      <th>zip</th>
      <th>state</th>
      <th>beds</th>
      <th>baths</th>
      <th>sq__ft</th>
      <th>type</th>
      <th>sale_date</th>
      <th>price</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>73</th>
      <td>17 SERASPI CT</td>
      <td>SACRAMENTO</td>
      <td>95834</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Wed May 21 00:00:00 EDT 2008</td>
      <td>206000</td>
      <td>38.631481</td>
      <td>-121.501880</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2866 KARITSA AVE</td>
      <td>SACRAMENTO</td>
      <td>95833</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Wed May 21 00:00:00 EDT 2008</td>
      <td>244500</td>
      <td>38.626671</td>
      <td>-121.525970</td>
    </tr>
    <tr>
      <th>100</th>
      <td>12209 CONSERVANCY WAY</td>
      <td>RANCHO CORDOVA</td>
      <td>95742</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Wed May 21 00:00:00 EDT 2008</td>
      <td>263500</td>
      <td>38.553867</td>
      <td>-121.219141</td>
    </tr>
    <tr>
      <th>121</th>
      <td>5337 DUSTY ROSE WAY</td>
      <td>RANCHO CORDOVA</td>
      <td>95742</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Wed May 21 00:00:00 EDT 2008</td>
      <td>320000</td>
      <td>38.528575</td>
      <td>-121.228600</td>
    </tr>
    <tr>
      <th>126</th>
      <td>2115 SMOKESTACK WAY</td>
      <td>SACRAMENTO</td>
      <td>95833</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Wed May 21 00:00:00 EDT 2008</td>
      <td>339500</td>
      <td>38.602416</td>
      <td>-121.542965</td>
    </tr>
    <tr>
      <th>133</th>
      <td>8082 LINDA ISLE LN</td>
      <td>SACRAMENTO</td>
      <td>95831</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Wed May 21 00:00:00 EDT 2008</td>
      <td>370000</td>
      <td>38.477200</td>
      <td>-121.521500</td>
    </tr>
    <tr>
      <th>147</th>
      <td>9278 DAIRY CT</td>
      <td>ELK GROVE</td>
      <td>95624</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Wed May 21 00:00:00 EDT 2008</td>
      <td>445000</td>
      <td>38.420338</td>
      <td>-121.363757</td>
    </tr>
    <tr>
      <th>153</th>
      <td>868 HILDEBRAND CIR</td>
      <td>FOLSOM</td>
      <td>95630</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Wed May 21 00:00:00 EDT 2008</td>
      <td>585000</td>
      <td>38.670947</td>
      <td>-121.097727</td>
    </tr>
    <tr>
      <th>169</th>
      <td>14788 NATCHEZ CT</td>
      <td>RANCHO MURIETA</td>
      <td>95683</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Tue May 20 00:00:00 EDT 2008</td>
      <td>97750</td>
      <td>38.492287</td>
      <td>-121.100032</td>
    </tr>
    <tr>
      <th>192</th>
      <td>5201 LAGUNA OAKS DR Unit 126</td>
      <td>ELK GROVE</td>
      <td>95758</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Condo</td>
      <td>Tue May 20 00:00:00 EDT 2008</td>
      <td>145000</td>
      <td>38.423251</td>
      <td>-121.444489</td>
    </tr>
    <tr>
      <th>234</th>
      <td>3139 SPOONWOOD WAY Unit 1</td>
      <td>SACRAMENTO</td>
      <td>95833</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Tue May 20 00:00:00 EDT 2008</td>
      <td>215500</td>
      <td>38.626582</td>
      <td>-121.521510</td>
    </tr>
    <tr>
      <th>236</th>
      <td>2340 HURLEY WAY</td>
      <td>SACRAMENTO</td>
      <td>95825</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Condo</td>
      <td>Tue May 20 00:00:00 EDT 2008</td>
      <td>225000</td>
      <td>38.588816</td>
      <td>-121.408549</td>
    </tr>
    <tr>
      <th>248</th>
      <td>611 BLOSSOM ROCK LN</td>
      <td>FOLSOM</td>
      <td>95630</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Condo</td>
      <td>Tue May 20 00:00:00 EDT 2008</td>
      <td>240000</td>
      <td>38.645700</td>
      <td>-121.119700</td>
    </tr>
    <tr>
      <th>249</th>
      <td>8830 ADUR RD</td>
      <td>ELK GROVE</td>
      <td>95624</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Tue May 20 00:00:00 EDT 2008</td>
      <td>242000</td>
      <td>38.437420</td>
      <td>-121.372876</td>
    </tr>
    <tr>
      <th>253</th>
      <td>221 PICASSO CIR</td>
      <td>SACRAMENTO</td>
      <td>95835</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Tue May 20 00:00:00 EDT 2008</td>
      <td>250000</td>
      <td>38.676658</td>
      <td>-121.528128</td>
    </tr>
    <tr>
      <th>265</th>
      <td>230 BANKSIDE WAY</td>
      <td>SACRAMENTO</td>
      <td>95835</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Tue May 20 00:00:00 EDT 2008</td>
      <td>270000</td>
      <td>38.676937</td>
      <td>-121.529244</td>
    </tr>
    <tr>
      <th>268</th>
      <td>4236 ADRIATIC SEA WAY</td>
      <td>SACRAMENTO</td>
      <td>95834</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Tue May 20 00:00:00 EDT 2008</td>
      <td>270000</td>
      <td>38.647961</td>
      <td>-121.543162</td>
    </tr>
    <tr>
      <th>279</th>
      <td>11281 STANFORD COURT LN Unit 604</td>
      <td>GOLD RIVER</td>
      <td>95670</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Condo</td>
      <td>Tue May 20 00:00:00 EDT 2008</td>
      <td>300000</td>
      <td>38.625289</td>
      <td>-121.260286</td>
    </tr>
    <tr>
      <th>285</th>
      <td>3224 PARKHAM DR</td>
      <td>ROSEVILLE</td>
      <td>95747</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Tue May 20 00:00:00 EDT 2008</td>
      <td>306500</td>
      <td>38.772771</td>
      <td>-121.364877</td>
    </tr>
    <tr>
      <th>286</th>
      <td>15 VANESSA PL</td>
      <td>SACRAMENTO</td>
      <td>95835</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Tue May 20 00:00:00 EDT 2008</td>
      <td>312500</td>
      <td>38.668692</td>
      <td>-121.545490</td>
    </tr>
    <tr>
      <th>308</th>
      <td>5404 ALMOND FALLS WAY</td>
      <td>RANCHO CORDOVA</td>
      <td>95742</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Tue May 20 00:00:00 EDT 2008</td>
      <td>425000</td>
      <td>38.527502</td>
      <td>-121.233492</td>
    </tr>
    <tr>
      <th>310</th>
      <td>14 CASA VATONI PL</td>
      <td>SACRAMENTO</td>
      <td>95834</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Tue May 20 00:00:00 EDT 2008</td>
      <td>433500</td>
      <td>38.650221</td>
      <td>-121.551704</td>
    </tr>
    <tr>
      <th>324</th>
      <td>201 FIRESTONE DR</td>
      <td>ROSEVILLE</td>
      <td>95678</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Tue May 20 00:00:00 EDT 2008</td>
      <td>500500</td>
      <td>38.770153</td>
      <td>-121.300039</td>
    </tr>
    <tr>
      <th>326</th>
      <td>2733 DANA LOOP</td>
      <td>EL DORADO HILLS</td>
      <td>95762</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Tue May 20 00:00:00 EDT 2008</td>
      <td>541000</td>
      <td>38.628459</td>
      <td>-121.055078</td>
    </tr>
    <tr>
      <th>327</th>
      <td>9741 SADDLEBRED CT</td>
      <td>WILTON</td>
      <td>95693</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Tue May 20 00:00:00 EDT 2008</td>
      <td>560000</td>
      <td>38.408841</td>
      <td>-121.198039</td>
    </tr>
    <tr>
      <th>469</th>
      <td>8491 CRYSTAL WALK CIR</td>
      <td>ELK GROVE</td>
      <td>95758</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Mon May 19 00:00:00 EDT 2008</td>
      <td>261000</td>
      <td>38.416916</td>
      <td>-121.407554</td>
    </tr>
    <tr>
      <th>477</th>
      <td>6286 LONETREE BLVD</td>
      <td>ROCKLIN</td>
      <td>95765</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Mon May 19 00:00:00 EDT 2008</td>
      <td>274500</td>
      <td>38.805036</td>
      <td>-121.293608</td>
    </tr>
    <tr>
      <th>494</th>
      <td>3072 VILLAGE PLAZA DR</td>
      <td>ROSEVILLE</td>
      <td>95747</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Mon May 19 00:00:00 EDT 2008</td>
      <td>307000</td>
      <td>38.773094</td>
      <td>-121.365905</td>
    </tr>
    <tr>
      <th>503</th>
      <td>12241 CANYONLANDS DR</td>
      <td>RANCHO CORDOVA</td>
      <td>95742</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Mon May 19 00:00:00 EDT 2008</td>
      <td>331500</td>
      <td>38.557293</td>
      <td>-121.217611</td>
    </tr>
    <tr>
      <th>505</th>
      <td>907 RIO ROBLES AVE</td>
      <td>SACRAMENTO</td>
      <td>95838</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Mon May 19 00:00:00 EDT 2008</td>
      <td>344755</td>
      <td>38.664765</td>
      <td>-121.445006</td>
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
    </tr>
    <tr>
      <th>600</th>
      <td>7 CRYSTALWOOD CIR</td>
      <td>LINCOLN</td>
      <td>95648</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Mon May 19 00:00:00 EDT 2008</td>
      <td>4897</td>
      <td>38.885962</td>
      <td>-121.289436</td>
    </tr>
    <tr>
      <th>601</th>
      <td>7 CRYSTALWOOD CIR</td>
      <td>LINCOLN</td>
      <td>95648</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Mon May 19 00:00:00 EDT 2008</td>
      <td>4897</td>
      <td>38.885962</td>
      <td>-121.289436</td>
    </tr>
    <tr>
      <th>602</th>
      <td>3 CRYSTALWOOD CIR</td>
      <td>LINCOLN</td>
      <td>95648</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Mon May 19 00:00:00 EDT 2008</td>
      <td>4897</td>
      <td>38.886093</td>
      <td>-121.289584</td>
    </tr>
    <tr>
      <th>604</th>
      <td>113 RINETTI WAY</td>
      <td>RIO LINDA</td>
      <td>95673</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>30000</td>
      <td>38.687172</td>
      <td>-121.463933</td>
    </tr>
    <tr>
      <th>686</th>
      <td>5890 TT TRAK</td>
      <td>FORESTHILL</td>
      <td>95631</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>194818</td>
      <td>39.020808</td>
      <td>-120.821518</td>
    </tr>
    <tr>
      <th>718</th>
      <td>9967 HATHERTON WAY</td>
      <td>ELK GROVE</td>
      <td>95757</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>222500</td>
      <td>38.305200</td>
      <td>-121.403300</td>
    </tr>
    <tr>
      <th>737</th>
      <td>3569 SODA WAY</td>
      <td>SACRAMENTO</td>
      <td>95834</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>247000</td>
      <td>38.631139</td>
      <td>-121.501879</td>
    </tr>
    <tr>
      <th>743</th>
      <td>6288 LONETREE BLVD</td>
      <td>ROCKLIN</td>
      <td>95765</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>250000</td>
      <td>38.804993</td>
      <td>-121.293609</td>
    </tr>
    <tr>
      <th>754</th>
      <td>6001 SHOO FLY RD</td>
      <td>PLACERVILLE</td>
      <td>95667</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>270000</td>
      <td>38.813546</td>
      <td>-120.809254</td>
    </tr>
    <tr>
      <th>755</th>
      <td>3040 PARKHAM DR</td>
      <td>ROSEVILLE</td>
      <td>95747</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>271000</td>
      <td>38.770835</td>
      <td>-121.366996</td>
    </tr>
    <tr>
      <th>757</th>
      <td>6007 MARYBELLE LN</td>
      <td>SHINGLE SPRINGS</td>
      <td>95682</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Unkown</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>275000</td>
      <td>38.643470</td>
      <td>-120.888183</td>
    </tr>
    <tr>
      <th>774</th>
      <td>8253 KEEGAN WAY</td>
      <td>ELK GROVE</td>
      <td>95624</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>298000</td>
      <td>38.446286</td>
      <td>-121.400817</td>
    </tr>
    <tr>
      <th>789</th>
      <td>5222 COPPER SUNSET WAY</td>
      <td>RANCHO CORDOVA</td>
      <td>95742</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>313000</td>
      <td>38.529181</td>
      <td>-121.224755</td>
    </tr>
    <tr>
      <th>798</th>
      <td>3232 PARKHAM DR</td>
      <td>ROSEVILLE</td>
      <td>95747</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>325500</td>
      <td>38.772821</td>
      <td>-121.364821</td>
    </tr>
    <tr>
      <th>819</th>
      <td>2274 IVY BRIDGE DR</td>
      <td>ROSEVILLE</td>
      <td>95747</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>375000</td>
      <td>38.778561</td>
      <td>-121.362008</td>
    </tr>
    <tr>
      <th>823</th>
      <td>201 KIRKLAND CT</td>
      <td>LINCOLN</td>
      <td>95648</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>389000</td>
      <td>38.867125</td>
      <td>-121.319085</td>
    </tr>
    <tr>
      <th>824</th>
      <td>12075 APPLESBURY CT</td>
      <td>RANCHO CORDOVA</td>
      <td>95742</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>390000</td>
      <td>38.535700</td>
      <td>-121.224900</td>
    </tr>
    <tr>
      <th>826</th>
      <td>5420 ALMOND FALLS WAY</td>
      <td>RANCHO CORDOVA</td>
      <td>95742</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>396000</td>
      <td>38.527384</td>
      <td>-121.233531</td>
    </tr>
    <tr>
      <th>828</th>
      <td>1515 EL CAMINO VERDE DR</td>
      <td>LINCOLN</td>
      <td>95648</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>400000</td>
      <td>38.904869</td>
      <td>-121.320750</td>
    </tr>
    <tr>
      <th>836</th>
      <td>1536 STONEY CROSS LN</td>
      <td>LINCOLN</td>
      <td>95648</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>433500</td>
      <td>38.860007</td>
      <td>-121.310946</td>
    </tr>
    <tr>
      <th>848</th>
      <td>200 HILLSFORD CT</td>
      <td>ROSEVILLE</td>
      <td>95747</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>511000</td>
      <td>38.780051</td>
      <td>-121.378718</td>
    </tr>
    <tr>
      <th>859</th>
      <td>4478 GREENBRAE RD</td>
      <td>ROCKLIN</td>
      <td>95677</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>600000</td>
      <td>38.781134</td>
      <td>-121.222801</td>
    </tr>
    <tr>
      <th>861</th>
      <td>200 CRADLE MOUNTAIN CT</td>
      <td>EL DORADO HILLS</td>
      <td>95762</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>622500</td>
      <td>38.647800</td>
      <td>-121.030900</td>
    </tr>
    <tr>
      <th>862</th>
      <td>2065 IMPRESSIONIST WAY</td>
      <td>EL DORADO HILLS</td>
      <td>95762</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Fri May 16 00:00:00 EDT 2008</td>
      <td>680000</td>
      <td>38.682961</td>
      <td>-121.033253</td>
    </tr>
    <tr>
      <th>888</th>
      <td>3035 ESTEPA DR Unit 5C</td>
      <td>CAMERON PARK</td>
      <td>95682</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Condo</td>
      <td>Thu May 15 00:00:00 EDT 2008</td>
      <td>119000</td>
      <td>38.681393</td>
      <td>-120.996713</td>
    </tr>
    <tr>
      <th>901</th>
      <td>1530 TOPANGA LN Unit 204</td>
      <td>LINCOLN</td>
      <td>95648</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Condo</td>
      <td>Thu May 15 00:00:00 EDT 2008</td>
      <td>138000</td>
      <td>38.884150</td>
      <td>-121.270277</td>
    </tr>
    <tr>
      <th>917</th>
      <td>501 POPLAR AVE</td>
      <td>WEST SACRAMENTO</td>
      <td>95691</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Thu May 15 00:00:00 EDT 2008</td>
      <td>165000</td>
      <td>38.584526</td>
      <td>-121.534609</td>
    </tr>
    <tr>
      <th>934</th>
      <td>1550 TOPANGA LN Unit 207</td>
      <td>LINCOLN</td>
      <td>95648</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Condo</td>
      <td>Thu May 15 00:00:00 EDT 2008</td>
      <td>188000</td>
      <td>38.884170</td>
      <td>-121.270222</td>
    </tr>
    <tr>
      <th>947</th>
      <td>1525 PENNSYLVANIA AVE</td>
      <td>WEST SACRAMENTO</td>
      <td>95691</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Thu May 15 00:00:00 EDT 2008</td>
      <td>200100</td>
      <td>38.569943</td>
      <td>-121.527539</td>
    </tr>
    <tr>
      <th>970</th>
      <td>3557 SODA WAY</td>
      <td>SACRAMENTO</td>
      <td>95834</td>
      <td>CA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Residential</td>
      <td>Thu May 15 00:00:00 EDT 2008</td>
      <td>224000</td>
      <td>38.631026</td>
      <td>-121.501879</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 12 columns</p>
</div>



The data set is complete with correct datatypes with no null values. Use unique to see if all state in the state column are the same. There was one AC. I just corrected AC to CA which will complete the state dataset. When I looked at the data through the describe. I noticed that there was a min value of a negative price in price column and a negative square feet in the sq_ft column which I decided. I would just correct the format of the values. So, it would make sense. 

_**Fun Fact:** Zip codes often have leading zeros — e.g., 02215 = Boston, MA — which will often get knocked off automatically by many software programs like Python or Excel. You can imagine that this could create some issues. _

### 3. Our goal will be to predict price. List variables that you think qualify as predictors of price in an SLR model. 

**For each of the variables you believe to be a valid potential predictor in an SLR model, generate a plot showing the relationship between the independent and dependent variables.**


```python
type(sc.corr())
```




    pandas.core.frame.DataFrame




```python
sns.heatmap(sc.corr(), annot=True, center =0)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x102184198>




![png](/images/linear-regression-sacramento_files/linear-regression-sacramento_26_1.png)



```python
# A: Using the heat map shows that the indpendent variable x Price
# is Strongly correlated with beds, baths, sq_ft.
#Seems like beds, baths, and sq_ft would be for predictors 
```

When you've finished cleaning or have made a good deal of progress cleaning, it's always a good idea to save your work.
```python
shd.to_csv('./datasets/sacramento_real_estate_transactions_Clean.csv')
```

### 4. Which variable would be the best predictor of Y in an SLR model? Why?


```python
# A: Baths have the strongest correlation compared to beds and sq_ft with price with beds being at 0.42. 
```

### 5. Build a function that will take in two lists, `Y` and `X`, and return the intercept and slope coefficients that minimize SSE. 

`Y` is the target variable and `X` is the predictor variable.

- **Test your function on price and the variable you determined was the best predictor in Problem 4.**
- **Report the slope and intercept.**


```python
# x = np.linspace(-5,50,100)
x = sc['baths']
```


```python
# y = 50 + 2 * x + np.random.normal(0, 20, size=len(x))
y = sc['price']
sc['Mean_Yhat'] = y.mean()
```


```python
intercept_slope = np.sum(np.square(sc['price'] - sc['Mean_Yhat']))
```


```python
# A: I know I did this wrong
def si(table):
    intercept_slope = np.sum(np.square(sc['price'] - sc['Mean_Yhat']))
    return intercept_slope
```


```python
print('Intercept is ', intercept_slope)
```

    Intercept is  18838783738865.37



```python
si('baths')
```




    18838783738865.37



### 6. Interpret the intercept. Interpret the slope.


```python
# A: X works with the y value. As the X value increases so should the y value
# The slope will increase by the price variable
```

### 7. Give an example of how this model could be used for prediction and how it could be used for inference. 

**Be sure to make it clear which example is associated with prediction and which is associated with inference.**


```python
# Prediction
# A: As a real estate agent, I may want to predict the pricing of housing
# in different areas along with the amount of space available and bedrooms
# Considering the cusotmer price range. I will best be able to determine
# where to start looking. 
```

### 8: [Bonus] Using the model you came up with in Problem 5, calculate and plot the residuals.


```python
y_bar = sc['price'].mean()
x_bar = sc['baths'].mean()
std_y = np.std(sc['price'], ddof=1)
std_x = np.std(sc['baths'], ddof=1)
r_xy = sc.corr().loc['baths','price']

beta_1_hat = r_xy * std_y / std_x
beta_0_hat = y_bar - beta_1_hat *x_bar

```


```python
print(beta_1_hat,beta_0_hat)
```

    64318.53523673409 119872.75465554858



```python
sc['Linear_Yhat'] = beta_0_hat + beta_1_hat * sc['baths']
```


```python
# A:

fig = plt.figure(figsize=(15,7))
fig.set_figheight(8)
fig.set_figwidth(15)


ax = fig.gca()


ax.scatter(x=sc['baths'], y=sc['price'], c='k')
ax.plot(sc['baths'], sc['Linear_Yhat'], color='k');

for _, row in sc.iterrows():
    plt.plot((row['baths'], row['baths']), (row['price'], row['Linear_Yhat']), 'r-')
```


![png](/images/linear-regression-sacramento_files/linear-regression-sacramento_46_0.png)


---

> The material following this point can be completed after the second lesson on Monday.

---

## Dummy Variables

---

It is important to be cautious with categorical variables, which represent distict groups or categories, when building a regression. If put in a regression "as-is," categorical variables represented as integers will be treated like *continuous* variables.

That is to say, instead of group "3" having a different effect on the estimation than group "1" it will estimate literally 3 times more than group 1. 

For example, if occupation category "1" represents "analyst" and occupation category "3" represents "barista", and our target variable is salary, if we leave this as a column of integers then barista will always have `beta*3` the effect of analyst.

This will almost certainly force the beta coefficient to be something strange and incorrect. Instead, we can re-represent the categories as multiple "dummy coded" columns.

### 9. Use the `pd.get_dummies` function to convert the `type` column into dummy-coded variables.

Print out the header of the dummy-coded variable output.


```python
# A:
sc_new = pd.get_dummies(sc[['type']])

sc_new.head()
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
      <th>type_Condo</th>
      <th>type_Multi-Family</th>
      <th>type_Residential</th>
      <th>type_Unkown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



---

### A Word of Caution When Creating Dummies

Let's touch on precautions we should take when dummy coding.

**If you convert a qualitative variable to dummy variables, you want to turn a variable with N categories into N-1 variables.**

> **Scenario 1:** Suppose we're working with the variable "sex" or "gender" with values "M" and "F". 

You should include in your model only one variable for "sex = F" which takes on 1 if sex is female and 0 if sex is not female! Rather than saying "a one unit change in X," the coefficient associated with "sex = F" is interpreted as the average change in Y when sex = F relative to when sex = M.

| Female | Male | 
|-------|------|
| 0 | 1 | 
| 1 | 0 |
| 0 | 1 |
| 1 | 0 |
| 1 | 0 |
_As we can see a 1 in the female column indicates a 0 in the male column. And so, we have two columns stating the same information in different ways._

> Scenario 2: Suppose we're modeling revenue at a bar for each of the days of the week. We have a column with strings identifying which day of the week this observation occured in.

We might include six of the days as their own variables: "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday". **But not all 7 days.**  

|Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | 
|-------|---------|-----------|----------|--------|----------|
| 1     | 0       |0          |      0   |0       | 0        | 
| 0     | 1       |0          |      0   |0       | 0        | 
| 0     | 0       |1          |      0   |0       | 0        | 
| 0     | 0       |0          |      1   |0       | 0        | 
| 0     | 0       |0          |      0   |1       | 0        | 
| 0     | 0       |0          |      0   |0       | 1        | 
| 0     | 0       |0          |      0   |0       | 0        | 

_As humans we can infer from the last row that if its is not Monday, Tusday, Wednesday, Thursday, Friday or Saturday than it must be Sunday. Models work the same way._

The coefficient for Monday is then interpreted as the average change in revenue when "day = Monday" relative to "day = Sunday." The coefficient for Tuesday is interpreted in the average change in revenue when "day = Tuesday" relative to "day = Sunday" and so on.

The category you leave out, which the other columns are *relative to* is often referred to as the **reference category**.

### 10. Remove "Unkown" from four dummy coded variable dataframe and append the rest to the original data.


```python
# A: 
sc_new.drop('type_Unkown', axis = 1, inplace=True)
```


```python
sc_new.head()
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
      <th>type_Condo</th>
      <th>type_Multi-Family</th>
      <th>type_Residential</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
sc.head(1)
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
      <th>street</th>
      <th>city</th>
      <th>zip</th>
      <th>state</th>
      <th>beds</th>
      <th>baths</th>
      <th>sq__ft</th>
      <th>type</th>
      <th>sale_date</th>
      <th>price</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>Mean_Yhat</th>
      <th>Linear_Yhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3526 HIGH ST</td>
      <td>SACRAMENTO</td>
      <td>95838</td>
      <td>CA</td>
      <td>2</td>
      <td>1</td>
      <td>836</td>
      <td>Residential</td>
      <td>Wed May 21 00:00:00 EDT 2008</td>
      <td>59222</td>
      <td>38.631913</td>
      <td>-121.434879</td>
      <td>234144.263959</td>
      <td>184191.289892</td>
    </tr>
  </tbody>
</table>
</div>




```python
sc = pd.concat([sc, sc_new], axis=1)
sc.head(1)
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
      <th>street</th>
      <th>city</th>
      <th>zip</th>
      <th>state</th>
      <th>beds</th>
      <th>baths</th>
      <th>sq__ft</th>
      <th>type</th>
      <th>sale_date</th>
      <th>price</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>Mean_Yhat</th>
      <th>Linear_Yhat</th>
      <th>type_Condo</th>
      <th>type_Multi-Family</th>
      <th>type_Residential</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3526 HIGH ST</td>
      <td>SACRAMENTO</td>
      <td>95838</td>
      <td>CA</td>
      <td>2</td>
      <td>1</td>
      <td>836</td>
      <td>Residential</td>
      <td>Wed May 21 00:00:00 EDT 2008</td>
      <td>59222</td>
      <td>38.631913</td>
      <td>-121.434879</td>
      <td>234144.263959</td>
      <td>184191.289892</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### 11. Build what you think may be the best MLR model predicting `price`. 

The independent variables are your choice, but *include at least three variables.* At least one of which should be a dummy-coded variable (either one we created before or a new one).

To construct your model don't forget to load in the statsmodels api:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```

_I'm going to engineer a new dummy variable for 'HUGE houses'.  Those whose square footage is 3 (positive) standard deviations away from the mean._
```
Mean = 1315
STD = 853
Huge Houses > 3775 sq ft
```


```python
sc['Huge_homes'] = (sc['sq__ft'] > 3775).astype(int)
```


```python
sc['Huge_homes'].value_counts()
```




    0    975
    1     10
    Name: Huge_homes, dtype: int64




```python
from sklearn.linear_model import LinearRegression

X = sc[['sq__ft', 'beds', 'baths','Huge_homes']].values
y = sc['price'].values

model = LinearRegression()
model.fit(X,y)

y_pred = model.predict(X)
```

### 12. Plot the true price vs the predicted price to evaluate your MLR visually.

> **Tip:** with seaborn's `sns.lmplot` you can set `x`, `y`, and even a `hue` (which will plot regression lines by category in different colors) to easily plot a regression line.


```python
sc.head()
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
      <th>street</th>
      <th>city</th>
      <th>zip</th>
      <th>state</th>
      <th>beds</th>
      <th>baths</th>
      <th>sq__ft</th>
      <th>type</th>
      <th>sale_date</th>
      <th>price</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>Mean_Yhat</th>
      <th>Linear_Yhat</th>
      <th>type_Condo</th>
      <th>type_Multi-Family</th>
      <th>type_Residential</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3526 HIGH ST</td>
      <td>SACRAMENTO</td>
      <td>95838</td>
      <td>CA</td>
      <td>2</td>
      <td>1</td>
      <td>836</td>
      <td>Residential</td>
      <td>Wed May 21 00:00:00 EDT 2008</td>
      <td>59222</td>
      <td>38.631913</td>
      <td>-121.434879</td>
      <td>234144.263959</td>
      <td>184191.289892</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>51 OMAHA CT</td>
      <td>SACRAMENTO</td>
      <td>95823</td>
      <td>CA</td>
      <td>3</td>
      <td>1</td>
      <td>1167</td>
      <td>Residential</td>
      <td>Wed May 21 00:00:00 EDT 2008</td>
      <td>68212</td>
      <td>38.478902</td>
      <td>-121.431028</td>
      <td>234144.263959</td>
      <td>184191.289892</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2796 BRANCH ST</td>
      <td>SACRAMENTO</td>
      <td>95815</td>
      <td>CA</td>
      <td>2</td>
      <td>1</td>
      <td>796</td>
      <td>Residential</td>
      <td>Wed May 21 00:00:00 EDT 2008</td>
      <td>68880</td>
      <td>38.618305</td>
      <td>-121.443839</td>
      <td>234144.263959</td>
      <td>184191.289892</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2805 JANETTE WAY</td>
      <td>SACRAMENTO</td>
      <td>95815</td>
      <td>CA</td>
      <td>2</td>
      <td>1</td>
      <td>852</td>
      <td>Residential</td>
      <td>Wed May 21 00:00:00 EDT 2008</td>
      <td>69307</td>
      <td>38.616835</td>
      <td>-121.439146</td>
      <td>234144.263959</td>
      <td>184191.289892</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6001 MCMAHON DR</td>
      <td>SACRAMENTO</td>
      <td>95824</td>
      <td>CA</td>
      <td>2</td>
      <td>1</td>
      <td>797</td>
      <td>Residential</td>
      <td>Wed May 21 00:00:00 EDT 2008</td>
      <td>81900</td>
      <td>38.519470</td>
      <td>-121.435768</td>
      <td>234144.263959</td>
      <td>184191.289892</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
sc['y_pred'] = y_pred
sns.lmplot(x='price', y='y_pred', data=sc, hue='Huge_homes')
```




    <seaborn.axisgrid.FacetGrid at 0x1c16640f28>




![png](/images/linear-regression-sacramento_files/linear-regression-sacramento_63_1.png)


### 13. List the five assumptions for an MLR model. 

Indicate which ones are the same as the assumptions for an SLR model. 

**SLR AND MLR**:  

- *Linearity: Y must have an approximately linear relationship with each independent X_i.*
- *Independence: Errors (residuals) e_i and e_j must be independent of one another for any i != j.*
- *Normality: The errors (residuals) follow a Normal distribution.*
- *Equality of Variances: The errors (residuals) should have a roughly consistent pattern, regardless of the value of the X_i. (There should be no discernable relationship between X_1 and the residuals.)*

**MLR ONLY**:  
- *Independence Part 2: The independent variables X_i and X_j must be independent of one another for any i != j*





### 14. Pick at least two assumptions and articulate whether or not you believe them to be met  for your model and why.


```python
# A: With the errors looking skewed right it does not show normality 
```


```python
sc['Residuals'] = sc['price'] - sc['y_pred']
sns.distplot(sc['Residuals'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c16531630>




![png](/images/linear-regression-sacramento_files/linear-regression-sacramento_68_1.png)



```python
#A Looks like it does show linearity because the y and x have an approximate
# linear relationship
```


```python
# Plot

x='Residuals'
y='price'

plt.scatter(x, y, s=area, data=sc, alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```


![png](/images/linear-regression-sacramento_files/linear-regression-sacramento_70_0.png)


### 15. [Bonus] Generate a table showing the point estimates, standard errors, t-scores, p-values, and 95% confidence intervals for the model you built. 

**Write a few sentences interpreting some of the output.**

> **Hint:** scikit-learn does not have this functionality built in, but statsmodels does in the `summary` method.  To fit the statsmodels model use something like the following.  There is one big caveat here, however!  `statsmodels.OLS` does _not_ add an intercept to your model, so you will need to do this explicitly by adding a column filled with the number 1 to your X matrix

```python
import statsmodels.api as sm

# The Default here is Linear Regression (ordinary least squares regression OLS)
model = sm.OLS(y,X).fit()
```

---

> The material following this point can be completed after the first lesson on Tuesday.

---


```python
# Standard Errors assume that the covariance matrix of the errors is correctly specified.
# The condition number is large, 1.7e+04. This might indicate that there are
# strong multicollinearity or other numerical problems.
# A "unit" increase in sq_ft is associated with a 9.4538 "unit" increase in prince.

```


```python
# Importing the stats model API
import statsmodels.api as sm


# Setting my X and y for modeling
sc['intercept'] = 1
X = sc[['intercept','sq__ft','beds','baths','Huge_homes']]
y = sc['price']

model = sm.OLS(y,X).fit()
```


```python
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th> <td>   0.194</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.191</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   59.05</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 04 Jul 2018</td> <th>  Prob (F-statistic):</th> <td>1.07e-44</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>13:37:52</td>     <th>  Log-Likelihood:    </th> <td> -12951.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   985</td>      <th>  AIC:               </th> <td>2.591e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   980</td>      <th>  BIC:               </th> <td>2.594e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th>  <td> 1.252e+05</td> <td> 9748.440</td> <td>   12.844</td> <td> 0.000</td> <td> 1.06e+05</td> <td> 1.44e+05</td>
</tr>
<tr>
  <th>sq__ft</th>     <td>    9.4538</td> <td>    6.985</td> <td>    1.353</td> <td> 0.176</td> <td>   -4.254</td> <td>   23.161</td>
</tr>
<tr>
  <th>beds</th>       <td>-3947.2178</td> <td> 5955.987</td> <td>   -0.663</td> <td> 0.508</td> <td>-1.56e+04</td> <td> 7740.738</td>
</tr>
<tr>
  <th>baths</th>      <td> 5.979e+04</td> <td> 8400.448</td> <td>    7.117</td> <td> 0.000</td> <td> 4.33e+04</td> <td> 7.63e+04</td>
</tr>
<tr>
  <th>Huge_homes</th> <td> 1.747e+05</td> <td> 4.29e+04</td> <td>    4.075</td> <td> 0.000</td> <td> 9.06e+04</td> <td> 2.59e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>231.835</td> <th>  Durbin-Watson:     </th> <td>   0.432</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 549.528</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 1.257</td>  <th>  Prob(JB):          </th> <td>4.69e-120</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.658</td>  <th>  Cond. No.          </th> <td>1.70e+04</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.7e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



### 16. Regression Metrics

Implement a function called `r2_adj()` that will calculate $R^2_{adj}$ for a model. 


```python
def r2_adj(y_true, y_preds, p):
    n = len(y_true)
    y_mean = np.mean(y_true)
    numerator = np.sum(np.square(y_true - y_preds)) / (n - p - 1)
    denominator = np.sum(np.square(y_true - y_mean)) / (n - 1)
    return 1 - numerator / denominator
```

### 17. Metrics, metrics, everywhere...

Write a function to calculate and print or return six regression metrics.  Use other functions liberally, including those found in `sklearn.metrics`.


```python
# A:
from sklearn.metrics import *
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
```


```python
y = sc['price']
x = sc['baths']
```


```python
y = sc['price']
X = sc.drop(['street', 'city', 'zip', 'state', 'type', 'sale_date', 'latitude', 'longitude','sq__ft'], axis="columns")


regression = sklearn.linear_model.LinearRegression(
    fit_intercept = True, 
    normalize = False,
    copy_X = True,
    n_jobs = 1
)

model = regression.fit(X, y)
y_hat = model.predict(X)

y_hat
```


```python
def regression_metrics(y, y_hat, p):
    r2 = sklearn.metrics.r2_score(y, y_hat),
    mse = sklearn.metrics.mean_squared_error(y, y_hat),
    #r2_adj = r2_adj(y, y_hat,p),
    msle = sklearn.metrics.mean_squared_log_error(y, y_hat),
    mae = sklearn.metrics.mean_absolute_error(y,y_hat),
    rmse = np.sqrt(mse)
    
    print('r2 = ', r2)
    print('mse = ', mse)
    #print(r2_adj)
    print('msle = ', msle)
    print('mae = ', mae)
    print('rmse = ', rmse)   
```

### 18. Model Iteration

Evaluate your current home price prediction model by calculating all six regression metrics.  Now adjust your model (e.g. add or take away features) and see how to metrics change.


```python
# A:
regression_metrics(sc['price'], y_pred, X.shape[1])

```

    r2 =  (0.19421379933530336,)
    mse =  (15411199973.689539,)
    msle =  (0.8478591994535334,)
    mae =  (93077.08723188947,)
    rmse =  [124141.85423816]



```python
sc.columns
```




    Index(['street', 'city', 'zip', 'state', 'beds', 'baths', 'sq__ft', 'type',
           'sale_date', 'price', 'latitude', 'longitude', 'Mean_Yhat',
           'Linear_Yhat', 'type_Condo', 'type_Multi-Family', 'type_Residential',
           'y_pred', 'Huge_homes', 'Residuals', 'intercept'],
          dtype='object')




```python
features = ['beds', 'baths', 'sq__ft','type_Condo', 'type_Multi-Family', 'type_Residential']
X = sc[features].values
y = sc['price'].values

model = LinearRegression()
model.fit(X, y)

y_hat = model.predict(X)
```


```python
regression_metrics(sc['price'], y_pred, X.shape[1])
```

    r2 =  (0.19421379933530336,)
    mse =  (15411199973.689539,)
    msle =  (0.8478591994535334,)
    mae =  (93077.08723188947,)
    rmse =  [124141.85423816]


### 19. Bias vs. Variance

At this point, do you think your model is high bias, high variance or in the sweet spot?  If you are doing this after Wednesday, can you provide evidence to support your belief?


```python
# A it seems like it will be in the sweet spot. I don't see signs for high bias or high variance 
```


```python
from sklearn.model_selection import cross_val_score


cv_scores = cross_val_score(model, X, y)
print(cv_scores)
print(np.mean(cv_scores))

```

    [ 0.09621108  0.10519894 -0.14432181]
    0.01902940197070409

