#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from os.path import join
import mysql.connector
import pymysql
from sqlalchemy import create_engine
from apyori import apriori
from mlxtend.frequent_patterns import apriori, association_rules
%matplotlib inline

#%%
import os
os.chdir('C:\\Users\\DELL\\Documents\\mbajupyter')
df = store_data = pd.read_csv('data2.csv')
df.head()

#%%
df['Description'] = df['Description'].str.strip()
df.dropna(axis = 0, subset=['BillNo'], inplace = True)
df['BillNo'] = df['BillNo'].astype('str')
df = df[~df['BillNo'].str.contains('C')]
df.head(10)



#%%

basket2 = df[df['Date'] == "2/5/2074"].groupby(['BillNo', 'Description'])['Qty'].sum().unstack().reset_index().fillna(0).set_index('BillNo')
basket2


#%%
basket4 = df[df['Date'] == "4/5/2074"].groupby(['BillNo', 'Description'])['Qty'].sum().unstack().reset_index().fillna(0).set_index('BillNo')
basket4

#%%
basket12 = df[df['Date'] == "12/5/2074"].groupby(['BillNo', 'Description'])['Qty'].sum().unstack().reset_index().fillna(0).set_index('BillNo')
basket12


#%%
records = []
for i in range(1, 5):
    records.append([str(basket2.values[i, j]) for j in range(0, 380)])

#%%
df = df.groupby(['BillNo','Description']).size().reset_index(name='count')
basket = (df.groupby(['BillNo', 'Description'])['count']
          .sum().unstack().reset_index().fillna(0)
          .set_index('BillNo'))

#%%
#The encoding function
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
basket_sets = basket.applymap(encode_units)

#%%
# Create a table containing top count of each item present in dataset
df1 = df.groupby('Description')['BillNo'].sum().to_frame().reset_index().sort_values(by='BillNo')
df1


#%%
from mlxtend.frequent_patterns import apriori

#Now, let us return the items and itemsets with at least 5% support:
apriori(basket_sets, min_support = 0.0025, use_colnames = True)

#%%
frequent_itemsets = apriori(basket_sets, min_support = 0.0025, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets

#%%
#finding support of length == 2
support1=frequent_itemsets[ (frequent_itemsets['length'] == 2) &
                   (frequent_itemsets['support'] >= 0.0025) ]

support1

#%%
# a function that takes the value and returns
# a series with as many columns as you want
#def split_itemsets(itemsets):
 #   first_name, last_name = itemsets.split(' ')

  #  return pd.Series({
   #     'first_name': first_name,
    #    'last_name': last_name
    #})

# df_new has the new columns
#support1_new = support1['itemsets'].apply(split_itemsets)

# append the columns to the original dataframe
#support1_final = pd.concat([support1,support1_new],axis=1)

#%%
engine = create_engine("mysql+pymysql://root:@localhost/practisedjangodb")
con = engine.connect()

support1.to_sql(name='frequent3', con=engine, if_exists = 'append', index=False)


#%%
#finding support of length == 3
support3 = frequent_itemsets[ (frequent_itemsets['length'] == 3) &
                   (frequent_itemsets['support'] >= 0.0025) ]
support3


#%%
#visualisation of length 3 items
length3 = support3.sum().div(len(support)).sort_values(ascending=False)
ax3 = length3.plot.bar(title='Support of length 3 itemsets')
x.locator_params(nbins=5, axis='x')
plt.xticks(rotation=60)
plt.rcParams['figure.figsize'] = (18, 7)
plt.tight_layout();
         
#%%
#find the min support
frequent_itemsets.sort_values('support', ascending=True).head()

#%%
#find the min support
frequent_itemsets1 = apriori(basket_sets, min_support = 0.0025, use_colnames=True)
frequent_itemsets1['length'] = frequent_itemsets1['itemsets'].apply(lambda x: len(x))
frequent_itemsets1

#%%
frequent_itemsets1=frequent_itemsets.sort_values('length', ascending=False).head() 
frequent_itemsets1['itemsets']= frequent_itemsets1['itemsets'].apply( 
   lambda x: pd.Series(str(x).split("_")))
frequent_itemsets1

#%%
engine = create_engine("mysql+pymysql://root:@localhost/practisedjangodb")
con = engine.connect()

frequent_itemsets1.head(40).to_sql(name='frequent', con=engine, if_exists = 'append', index=False)




#%%
#After choosing optimal threshold support
## Apriori to select the most important itemsets
frequent_itemsets = apriori(basket_sets, min_support=0.0025, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head(20)


#%%
rules

#%%
rules.rename(columns={'antecedent support':'antecedent_support',
                          'consequent support':'consequent_support'}, 
                 inplace=True)


#%%
user_name=os.environ.get('DB_USER')
password=os.environ.get('DB_PASS')

#%%
connection=mysql.connector.connect(host='localhost',
                                  user='root',
                                  password='',
                                  db='practisedjangodb')

#%%
connection

#%%
majorproject_tables=pd.read_sql_query('SHOW TABLES from practisedjangodb',connection)

#%%
majorproject_tables


#%%
engine = create_engine("mysql+pymysql://root:@localhost/practisedjangodb")
con = engine.connect()

rules.head(60).to_sql(name='relation_relation', con=engine, if_exists = 'append', index=False)



#%%
#DATA_VISUALIZATIONS
sns.countplot(x = 'Description', data = df, order = df['Description'].value_counts().iloc[:10].index)
plt.xticks(rotation=90)


#%%
support = basket.sum().div(len(basket)).sort_values(ascending=False)
ax = support.plot.bar(title='Item Support')
ax.locator_params(nbins=20, axis='x')
plt.xticks(rotation=60)
plt.rcParams['figure.figsize'] = (18, 7)
plt.tight_layout();
plt.savefig('/Users/DELL/Downloads/image.jpg')

#%%
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (18, 5)
color = plt.cm.copper(np.linspace(0, 1, 40))
df['Description'].value_counts().head(40).plot.bar(color = color)
plt.title('frequency of most popular items', fontsize = 20)
plt.xticks(rotation = 90 )
plt.grid()
plt.savefig('/Users/DELL/Downloads/countplot1.jpg', bbox_inches="tight")

#%%
# create a figure and axis
plt.figure(figsize=(14, 8))
plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], alpha=0.9, cmap='YlOrRd');
plt.title('Rules distribution color mapped by lift');
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.colorbar();


#%%

sns.distplot(rules['support'], bins=10, kde=True)

#%%


#%%
