import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('data/train_wkt_v4.csv')
df.head()
diction = {}
for i in range(len(df)):
    if df.iloc[i,2] != 'MULTIPOLYGON EMPTY':
        if df.iloc[i,1] not in diction:
            diction[df.iloc[i,1]] =1
        else:
            diction[df.iloc[i,1]] += 1
y = [float(i)/sum(diction.values()) for i in diction.values() ]
import matplotlib.pyplot as plt
x = [1,2,3,4,5,6,7,8,9,10]
fig = plt.figure(figsize=(10,10))
plt.bar(x, y)
plt.xticks(x,['Buildings','MSC','road','track','tree', 'crops','Waterways','Standing water','Vehicle (L)','Vehicle (S)'])
plt.xticks(rotation=30,fontsize=14)
plt.ylabel('Frequency',fontsize=14)
plt.title('Histogram of classes in train image',fontsize=18)
plt.show()
fig.savefig('histogram.png')


