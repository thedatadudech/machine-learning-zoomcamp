```python
### Q1. Pandas version
import pandas as pd
print(f"A1: The panda version is {pd.__version__}")
```

    A1: The panda version is 2.2.3



```python
df = pd.read_csv("https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv")
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
      <th>Laptop</th>
      <th>Status</th>
      <th>Brand</th>
      <th>Model</th>
      <th>CPU</th>
      <th>RAM</th>
      <th>Storage</th>
      <th>Storage type</th>
      <th>GPU</th>
      <th>Screen</th>
      <th>Touch</th>
      <th>Final Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ASUS ExpertBook B1 B1502CBA-EJ0436X Intel Core...</td>
      <td>New</td>
      <td>Asus</td>
      <td>ExpertBook</td>
      <td>Intel Core i5</td>
      <td>8</td>
      <td>512</td>
      <td>SSD</td>
      <td>NaN</td>
      <td>15.6</td>
      <td>No</td>
      <td>1009.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alurin Go Start Intel Celeron N4020/8GB/256GB ...</td>
      <td>New</td>
      <td>Alurin</td>
      <td>Go</td>
      <td>Intel Celeron</td>
      <td>8</td>
      <td>256</td>
      <td>SSD</td>
      <td>NaN</td>
      <td>15.6</td>
      <td>No</td>
      <td>299.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ASUS ExpertBook B1 B1502CBA-EJ0424X Intel Core...</td>
      <td>New</td>
      <td>Asus</td>
      <td>ExpertBook</td>
      <td>Intel Core i3</td>
      <td>8</td>
      <td>256</td>
      <td>SSD</td>
      <td>NaN</td>
      <td>15.6</td>
      <td>No</td>
      <td>789.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MSI Katana GF66 12UC-082XES Intel Core i7-1270...</td>
      <td>New</td>
      <td>MSI</td>
      <td>Katana</td>
      <td>Intel Core i7</td>
      <td>16</td>
      <td>1000</td>
      <td>SSD</td>
      <td>RTX 3050</td>
      <td>15.6</td>
      <td>No</td>
      <td>1199.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HP 15S-FQ5085NS Intel Core i5-1235U/16GB/512GB...</td>
      <td>New</td>
      <td>HP</td>
      <td>15S</td>
      <td>Intel Core i5</td>
      <td>16</td>
      <td>512</td>
      <td>SSD</td>
      <td>NaN</td>
      <td>15.6</td>
      <td>No</td>
      <td>669.01</td>
    </tr>
  </tbody>
</table>
</div>




```python
### Q2. Records count
#How many records are in the dataset?

print(f"A2: There are {df.shape[0]} records in the dataset")
```

    A2: There are 2160 records in the dataset



```python
### Q3. Laptop brands
#How many laptop brands are presented in the dataset?

print(f"A3: There are {df.Brand.nunique()} brands in the dataset")

```

    A3: There are 27 brands in the dataset



```python
### Q4. Missing values
#How many columns in the dataset have missing values?
print(f"A4 There are {sum(df.isna().sum()>0)} columns with missing values")
```

    A4 There are 3 columns with missing values



```python
### Q5. Maximum final price
#What's the maximum final price of Dell notebooks in the dataset?

print(f"""A5 The maximum final price of Dell notebooks in the data is {max(df.query("Brand=='Dell'")["Final Price"])}""")
```

    A5 The maximum final price of Dell notebooks in the data is 3936.0



```python
### Q6. Median value of Screen
print(f"Median  of Screen:  {df.Screen.median()}")
print(f"Mode of Screen:  {df.Screen.mode().iloc[0]}")

Screen_filled = df.Screen.fillna(df.Screen.mode().iloc[0])

print(f"Median  of Screen filled:  {Screen_filled.median()}")

print("A6")
print("No it hasn't changed") if df.Screen.median() == Screen_filled.median() else print("Yes it has changed")
```

    Median  of Screen:  15.6
    Mode of Screen:  15.6
    Median  of Screen filled:  15.6
    A6
    No it hasn't changed



```python
import numpy as np
from numpy.linalg import inv
X =df.query("Brand == 'Innjoo'")[["RAM","Storage","Screen"]].values

```


```python
XTXinv = inv(np.dot(X.T,X))
y =np.array([1100, 1300, 800, 900, 1000, 1100])
```


```python
XTXinvXT = np.dot(XTXinv,X.T)
```


```python
w = np.dot(XTXinvXT,y)
```


```python
print(f"A7 The sum of the resulting vector is {round(sum(w),3)}")
```

    A7 The sum of the resulting vector is 91.3

