import numpy as np
import pandas as pd
import matplotlib as mtl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
mtl.rcParams['figure.figsize'] = [125, 70]

df = pd.read_csv('flights.csv')

def f(row):
 if row['month'] == 'January':
     val = 1
 elif row['month'] == 'February':
     val = 2
 elif row['month'] == 'March':
     val = 3
 elif row['month'] == 'April':
     val = 4
 elif row['month'] == 'May':
     val = 5
 elif row['month'] == 'June':
     val = 6
 elif row['month'] == 'July':
     val = 7
 elif row['month'] == 'August':
     val = 8
 elif row['month'] == 'September':
     val = 9
 elif row['month'] == 'October':
     val = 10
 elif row['month'] == 'November':
     val = 11
 else :
     val = 12
 return val
df['month'] = df.apply(f,axis=1)
print(df.tail(20))

X = df.loc[:20,['passengers','month']]
y = df.loc[:20,['year']]

model = tree.DecisionTreeClassifier(criterion='entropy')
model.fit(X,y)
model.score(X,y)
#model.predict([[548,8]])
tree.plot_tree(model)
