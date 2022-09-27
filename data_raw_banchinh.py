import pandas as pd 
import numpy as np
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
df = pd.read_excel('G:/Case Study/data_raw_datach.xlsx')
df.info()
df.isnull().sum()
df['color'] = df['color'].fillna("color")
df['week'] = pd.to_datetime(df['week'])
df['month'] = df['week'].apply(lambda x: x.month) 
df['year'] = df['week'].apply(lambda x: x.year)
df['day'] = df['week'].apply(lambda x: x.day)
df['year'].unique()
df_00 = df[['weekly_sales','color','price','vendor','functionality','month','year']]
# _________________________________ 2016 _____________________________________
df_2016 = df_00[df_00['year'] == 2016]
df_2016.info()
df_2016_bang1 = df_2016.groupby(['functionality']).agg(total_price = ('price','sum'),total_weekly_sales = ('weekly_sales','sum'))
df_2016_bang1_top5 = df_2016_bang1.sort_values('total_price',ascending=False).head(5)
df_2016_bang1_top5.plot.barh(rot=0,figsize = (12,10),title ="Top 5 best selling functional products in 2016",color={"total_price": "orange", "total_weekly_sales":"blue"})
df_01 = df_00[(df_00['functionality'] == '01.Streaming sticks') & (df_00['year'] == 2016)]
df_01 = df_01.drop('year',axis = 1)
df_01_bang1 = df_01.groupby('month').sum()
df_2016_bang1['total_weekly_sales'].sum()
df_2016.vendor.sum()
# df_2016_bang1_top5.to_excel('G:/Case Study/data_raw_1.xlsx')
# df_2016_bang1.to_excel('G:/Case Study/df_2016_bang1.xlsx')
# df_01_bang1.to_excel('G:/Case Study/data_raw_2.xlsx')

#__________________________________ 2017 _____________________________________
df_2017 = df_00[df_00['year'] == 2017]
df_2017.info()
df_2017_bang1 = df_2017.groupby(['functionality']).agg(total_price = ('price','sum'),total_weekly_sales = ('weekly_sales','sum'))
df_2017_bang2 = df_2017.groupby(['functionality']).agg(total_price = ('price','sum'),total_weekly_sales = ('weekly_sales','sum'),total_vendor = ('vendor','sum'))
df_2017_bang1_top5 = df_2017_bang1.sort_values('total_price',ascending=False).head(5)
df_02 = df_00[(df_00['functionality'] == '06.Mobile phone accessories') & (df_00['year'] == 2017)]
df_02 = df_02.drop('year',axis = 1)
df_02.info()
df_02_bang1 = df_02.groupby(['month','color']).sum()[['weekly_sales','price','vendor']].reset_index()
df_02_bang2 = df_02.groupby('month').sum().sort_values('price',ascending = False)
df_02_bang3 = df_02_bang1[df_02_bang1['month'] == 7].sort_values('price',ascending = False)
(df_02_bang3.iloc[0,3] / df_02_bang2.iloc[0,1])*100 
df_2017_bang1['total_weekly_sales'].sum()
df_2017.vendor.sum()
# df_2017_bang1_top5.to_excel('G:/Case Study/data_raw_3.xlsx') 
# df_02_bang2.to_excel('G:/Case Study/data_2017_month.xlsx')
# df_2017_bang2.to_excel('G:/Case Study/df_2017_bang2.xlsx')

#__________________________________ 2018 _____________________________________
df_2018 = df_00[df_00['year'] == 2018]
df_2018.info()
df_2018_bang1 = df_2018.groupby(['functionality']).agg(total_price = ('price','sum'),total_weekly_sales = ('weekly_sales','sum'))
df_2018_bang2 = df_2018.groupby(['functionality']).agg(total_price = ('price','sum'),total_weekly_sales = ('weekly_sales','sum'),total_vendor = ('vendor','sum'))
df_2018_bang1_top5 = df_2018_bang1.sort_values('total_price',ascending=False).head(5)
df_03 = df_00[(df_00['functionality'] == '06.Mobile phone accessories') & (df_00['year'] == 2018)]
df_03 = df_03.drop('year',axis = 1)
df_03_bang1 = df_03.groupby(['month','color']).sum()[['weekly_sales','price','vendor']].reset_index()
df_03_bang2 = df_03.groupby('month').sum().sort_values('price',ascending = False)
df_03_bang3 = df_03_bang1[df_03_bang1['month'] == 1].sort_values('price',ascending = False)
(df_03_bang3.iloc[0,3] / df_03_bang2.iloc[0,1])*100 
df_2018_bang1['total_weekly_sales'].sum()
df_2018.vendor.sum()
# df_2018_bang1_top5.to_excel('G:/Case Study/data_2018_top5.xlsx')
# df_03_bang2.to_excel('G:/Case Study/data_2018_month.xlsx')
# df_2018_bang2.to_excel('G:/Case Study/data_2018_bang2.xlsx')

#__________________________________ Total ____________________________________
df_total = (df_2016_bang1 + df_2017_bang1 + df_2018_bang1).reset_index()
df_total = df_total.sort_values('total_price',ascending=False)
df_total_top3 = df_total.head(3)
df_total_tail3 = df_total.tail(3) 
df.weekly_sales.sum()
df.vendor.sum()
df.info()
df_total_1 = df.groupby(['year']).sum()[['weekly_sales','price','vendor']].reset_index()
# df_total_1.to_excel('G:/Case Study/df_total_1.xlsx')
# df_total.to_excel('G:/Case Study/data_total.xlsx')


#_____________________________  Machine Learning ______________________
df_ML_00 = pd.read_excel('G:/Case Study/data_raw_datach.xlsx')
df_ML_00 = df_ML_00.dropna()
df_ML_00.isnull().sum()
df_ML_01 = df_ML_00[['weekly_sales','feat_main_page','color','price','vendor','functionality']]
df_ML_01['feat_main_page'] = df_ML_01['feat_main_page'].replace(['True','False'],['1','0']).astype('int')
df_ML_01['color'].value_counts()
df_ML_01['color'] = df_ML_01['color'].replace(['none','black','blue','red','green','grey','white','gold','purple','pink'],
                                              ['0','1','2','3','4','5','6','7','8','9']).astype('int')
df_ML_01['functionality'].value_counts()
df_ML_01['functionality'] = df_ML_01['functionality'].apply(lambda x: x.split('.')[1])

"one Hot Encoding"
dum = pd.get_dummies(df_ML_01.functionality)
df_concat = pd.concat([df_ML_01, dum], axis = 'columns')  
df_chinh = df_concat.drop('functionality',axis = 1)  

" Separate out the independent variable and The target variable "
x = df_chinh.drop('price',axis = 1)
y = df_chinh.price
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2 )
from sklearn.linear_model import LinearRegression
lr_cl = LinearRegression() 
lr_cl.fit(x_train, y_train)

" Test " 
lr_cl.score(x_test,y_test) # 82%

# Funtion Enter parameters 
def price_dudoan(functionality,weekly_sales,feat_main_page,color,vendor):
 lo_index = np.where(x.columns == functionality)[0][0]
 z = np.zeros(len(x.columns))
 z[0] = weekly_sales
 z[1] = feat_main_page
 z[2] = color
 z[3] = vendor
 if lo_index > 0:
     z[lo_index] = 1
 return lr_cl.predict([z])[0]

# Enter parameters 
price_dudoan('Selfie sticks',96,0,4,4)












