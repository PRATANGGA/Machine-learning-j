#!/usr/bin/env python
# coding: utf-8

# In[25]:


import os,zipfile


# In[26]:


# os.mkdir('dataset')
dataset_dir = 'dataset'


# In[27]:


target_file = 'indonesian-salary-by-region-19972022.zip'


# In[28]:


extracting = zipfile.ZipFile(target_file, 'r')
extracting.extractall(dataset_dir)
extracting.close()


# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression

import os
for dirname, _, filenames in os.walk(f'{dataset_dir}/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[30]:


ump_data = pd.read_csv(f"{dataset_dir}/Indonesian Salary by Region (1997-2022).csv")
ump_data.head()


# In[31]:


ump_data.info()


# In[32]:


sns.catplot(x='YEAR', y='SALARY', data=ump_data, kind='point', aspect=2.5)


# In[33]:


ump_data['SALARY'] = pd.to_numeric(ump_data['SALARY'], errors='coerce')

avg = ump_data.groupby(["YEAR"])['SALARY'].mean().reset_index()
avg.head()


# In[34]:


growth=[0]
for i in range(1,26):
    growth.append(avg["SALARY"][i]-avg["SALARY"][i-1])

avg["growth"]=growth
avg.head()


# In[35]:


plt.figure(figsize=(19,10))
plt.title("Average salary growth of country by year", size=20, pad=20)

ax = sns.barplot(x='YEAR', y='growth', data=avg, color='green', edgecolor="black")

# loop through each bar and annotate
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',
                xytext = (0, 10),
                textcoords = 'offset points')

plt.show()


# In[36]:


# ambil 5 region dengan salary tertinggi
top_5_regions = ump_data.groupby('REGION')['SALARY'].max().sort_values(ascending=False)[:5]
top_5_regions = top_5_regions.reset_index()

plt.figure(figsize = (20, 10))
graph = sns.barplot(x = 'REGION', y = 'SALARY', data=top_5_regions, order=top_5_regions['REGION'])
graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
plt.xlabel('Region', fontsize=16)
plt.ylabel('Salary', fontsize=16)
plt.title('Top 5 Province with the Highest Salary', fontsize=22)

for index, row in top_5_regions.iterrows():
    graph.annotate(format(int(row['SALARY']), ','),
                   (row.name, row['SALARY']),
                   ha = 'center', va = 'center',
                   xytext = (0, 10),
                   textcoords = 'offset points')


# In[37]:


ump_data = ump_data.reset_index(drop=True).groupby('REGION').apply(lambda x: x.sort_values('YEAR'))

models = {}
for provinsi in ump_data['REGION'].unique():
    X = ump_data.loc[ump_data['REGION'] == provinsi]['YEAR'].values.reshape(-1, 1)
    y = ump_data.loc[ump_data['REGION'] == provinsi]['SALARY'].values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    models[provinsi] = model


# In[38]:


future_years = np.array(range(ump_data['YEAR'].max() + 1, ump_data['YEAR'].max() + 11))
future_ump = []
for provinsi in ump_data['REGION'].unique():
    model = models[provinsi]
    future_ump_provinsi = model.predict(future_years.reshape(-1, 1))
    future_ump.extend(future_ump_provinsi)


# In[39]:


future_df = pd.DataFrame({
    'REGION': np.repeat(ump_data['REGION'].unique(), 10),
    'YEAR': np.tile(range(ump_data['YEAR'].max() + 1, ump_data['YEAR'].max() + 11), ump_data['REGION'].nunique()),
    'SALARY': future_ump
})


# In[40]:


future_df['SALARY'] = future_df['SALARY'].astype(int)
future_df.info()


# In[41]:


future_df.head()


# In[42]:


combined_df = pd.concat([ump_data, future_df],axis=0,ignore_index=True)


# In[43]:


combined_df.info()


# In[44]:


combined_df.head()


# In[45]:


AVG = combined_df.sort_values(by='YEAR', ascending=True)

AVG['Growth'] = [0] + [AVG['SALARY'].iloc[i] - AVG['SALARY'].iloc[i-1] for i in range(1, len(AVG))]

plt.figure(figsize=(15,12))
plt.title("Average salary growth from 2022 - 2032", size=20, pad=20)

AVG_filtered = AVG.query('YEAR >= 2022 and YEAR <= 2032')

ax = sns.barplot(x='YEAR', y='Growth', data=AVG_filtered, color='green', edgecolor="black")

# loop through each bar and annotate
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',
                xytext = (0, 10),
                textcoords = 'offset points')

plt.show()


# In[46]:


sns.catplot(x='YEAR', y='SALARY', data=combined_df, kind='point', aspect=5)


# In[47]:


import plotly.express as px

fig = px.bar(combined_df, x='REGION', y="SALARY",color="REGION",
  animation_frame="YEAR", range_y=[0,7000000])
fig.show()


# In[48]:


# ambil 5 region dengan salary tertinggi
top_5_regions = combined_df.groupby('REGION')['SALARY'].max().sort_values(ascending=False)[:5]
top_5_regions = top_5_regions.reset_index()



plt.figure(figsize = (16, 8))
graph = sns.barplot(x = 'REGION', y = 'SALARY', data=top_5_regions, order=top_5_regions['REGION'])
graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
plt.xlabel('Region', fontsize=15)
plt.ylabel('Salary', fontsize=15)
plt.title('Top 5 Regions with the Highest Salary', fontsize=20)

for index, row in top_5_regions.iterrows():
    graph.annotate(format(int(row['SALARY']), ','),
                   (row.name, row['SALARY']),
                   ha = 'center', va = 'center',
                   xytext = (0, 10),
                   textcoords = 'offset points')


# In[ ]:




