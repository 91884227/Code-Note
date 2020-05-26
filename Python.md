# Python note
[TOC]
## call variable by string
```python
globals()[string]
```

## change type of list
```python
list(map(int,  list_))
[ list(map(int,  i)) for i in after_encode]  # 2d list
```

## pickle **save class**
```python
import pickle
class Company(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value
# save
with open('company_data.pkl', 'wb') as output:
    company1 = Company('banana', 40)
    pickle.dump(company1, output, pickle.HIGHEST_PROTOCOL)
    
# read
with open('company_data.pkl', 'rb') as input:
    company1 = pickle.load(input)
```

## collection
```python
from collections import Counter
c = Counter('abcasd')
c.most_common()
```

## .py 的開頭
```python
if __name__ == '__main__':
```
## split training and testing
```python
from sklearn.model_selection import train_test_split
X, y = np.arange(10).reshape((5, 2)), range(5)
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.33, random_state=42)
```
## random
```python
import random

number_list = [7, 14, 21, 28, 35, 42, 49, 56, 63, 70]
print("Original list:", number_list)

random.shuffle(number_list)
print("List after first shuffle:", number_list)
```
## json save/read
```python
# write file
import json
with open('savename.txt', 'w') as outfile:
    json.dump(savedata, outfile)
    
# read file
with open('dictonary.json') as json_file:
    data = json.load(json_file)
```
## os 
```python
import os
path = os.getcwd() #取得目前路徑
os.chdir(path) #改變路徑
os.listdir(path) # 列出folder的全部item
# Rmk: 若想回到上一層路徑 可用 os.chdir("..")
# Rmk: 相對路徑可用 "./" 表示
```
## Regular expression
[Online regular expression](https://regex101.com/)
```python
import re
```
## change numpy type
```python
x = np.array(["1", "2"])
x.astype(int)
```
## select list element by bool index
```python
from itertools import compress
list_a = [1, 2, 4, 6]
fil = [True, False, True, False]
list(compress(list_a, fil))
```

## draw correlation map
資料格式:(dataframe)

| A | B | C |
| -------- | -------- | -------- |
|      |      |   |
|      |      |   |
|      |      |   |
|      |      |   |
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {'A': [45,37,42,35,39],
        'B': [38,31,26,28,33],
        'C': [10,15,17,21,12]
        }
        
df = pd.DataFrame(data,columns=['A','B','C'])

corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
# plt.savefig('corrMatrix.png') 輸出 #須放在 plt.show 之前 
# plt.figure(figsize=(20,20)) 調整大小
plt.show()
```
## select list element by index
```python
test_list = [9, 4, 5, 8, 10, 14] 
index_list = [1, 3, 4]
list(map(test_list.__getitem__, index_list)) # [4, 8, 10]
```
## nested list to 1d list
```
from itertools import chain
list(chain.from_iterable(nested_array))
```
## pandas
```
df.iloc[0] # get first row
df[df.columns[0]] # get first column
```
## Sorting list based on values from another list
```
Y = [ 0,   1,   1,    0,   1,   2,   2,   0,   1]
X = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

X.sort(key=dict(zip(X, Y)).get)
```
## tqdm notebook
```
import tqdm
tqdm_notebook = tqdm.notebook.tqdm
[i for i in tqdm_notebook(range(10000))]
```

