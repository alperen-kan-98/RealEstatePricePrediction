# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:52:58 2020

@author: alper
"""
import pandas as pd
import numpy as np

# 0.)Read the csv file in to a DataFrame Object
EvAlpha = pd.read_csv("merge.csv", sep = ",")
EvAlpha.dtypes
#---First column is a mistake which comes from datset-creation-automation 
#---program so I remove that column
Ev = EvAlpha.iloc[:,1:]
Ev.dtypes
print(Ev.isnull().sum())
# I need to convert prices to int (they are float for now)
# To do this 1st I need to handle the NaN values
#------------------------------------------------------------------------------
# 1.)Dealing with missing values
#---1st start with "banyosayisi"
banyosayisi = Ev.iloc[:,7:8]
from sklearn.impute import SimpleImputer
impForBanyoSayisi = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
banyosayisi = impForBanyoSayisi.fit_transform(banyosayisi)
banyosayisi = banyosayisi = banyosayisi.astype('int64')
banyosayisidf = pd.DataFrame.from_records(banyosayisi)
banyosayisidf.columns = ['banyosayisi']
#----banyosayisidf --> No nan values and its a DF---#
#------------------------------------------------------------------------------
#---2nd deal with m2
m2 = Ev.iloc[:,1:2]
impForM2 = SimpleImputer(missing_values=np.nan, strategy='mean')
m2 = impForM2.fit_transform(m2)
m2 = m2 = m2.astype('int64')
m2df = pd.DataFrame.from_records(m2)
m2df.columns = ['m2']
# m2 --> No nan values and its a DF
#------------------------------------------------------------------------------
#---3rd now time for "price" but maybe we will need both nan prices 
#---and mean value imputed prices so lets create both of them
price = Ev.iloc[:,-1:]## This will be the NaN values included version
print(price.median())
impForPrice = SimpleImputer(missing_values=np.nan, strategy='median')
price = impForPrice.fit_transform(price)
price = price = price.astype('int64')
pricedf = pd.DataFrame.from_records(price)
pricedf.columns = ['price']
##Mean is giving a bad result
## I think median is better
## But maybe dropping that rows will be more clever
### For that dont combine this DF with whole dataset and the use "dropna"
#---4th deal with esyali
#--- Most of the records are unknown so I fill them with most frequent
esyaliN =  Ev.iloc[:,8:9]
impForEsyali = SimpleImputer(missing_values='Belirtilmemiş', strategy='most_frequent')
esyaliN = impForEsyali.fit_transform(esyaliN)
esyaliNdf = pd.DataFrame.from_records(esyaliN)
esyaliNdf.columns = ['esyali']
# 3.) Now combine these and drop the nan values(price), we will still have
#     more than 1700 records so it will be enough
ilce = Ev.iloc[:,0:1]
two_to_six = Ev.iloc[:,2:7]
esyali = Ev.iloc[:,8:9]
esyali_price = Ev.iloc[:,8:10]
priceUnTouched = Ev.iloc[:,-1:]
EvNew = pd.concat([ilce, m2df, two_to_six, banyosayisidf, esyaliNdf, priceUnTouched], axis =1)
EvNew = EvNew.dropna()
EvNewDrop = pd.concat([ilce, m2df, two_to_six, banyosayisidf, esyali_price], axis =1)
EvNewDrop = EvNewDrop.dropna()
print(EvNewDrop.isnull().sum())
EvNew = EvNew.astype({"price": int})
EvNew.dtypes
EvNew = EvNew.astype({"yas": int})
EvNew.dtypes
EvNew = EvNew.astype({"yas": int})
EvNew.dtypes
print(EvNew.isnull().sum())
## Pre-Processing 1st Step finished
### Now I will look at visualization a bit and then 
####I will scale the numerical values and label the cat values

EvNew = EvNew[EvNew.price<10000000 ]# 10.000.000 UNUTMA
# 4.) Encode the categorical variables
from sklearn import preprocessing 
import matplotlib.pyplot as plt
###################################################○
ohe = preprocessing.OneHotEncoder() 
ilcedfZ = EvNew.iloc[:,0:1]
ilcedfZ2 = ohe.fit_transform(ilcedfZ).toarray()
ilcenewDF = pd.DataFrame(ilcedfZ2)#$$$$$
ilcenewDF.columns = (['arnavutkoy', 'atasehir', 'avcilar', 'bahcelievler', 'bakirkoy',
       'basaksehir', 'besiktas', 'beykoz', 'beylikduzu', 'beyoglu',
       'buyukcekmece', 'catalca', 'cekmekoy', 'esenler', 'esenyurt',
       'eyup', 'fatih', 'gaziosmanpasa', 'gungoren', 'kadikoy',
       'kagithane', 'kartal', 'kucukcekmece', 'maltepe', 'pendik',
       'sariyer', 'sisli', 'tuzla', 'umraniye', 'uskudar'])

#--------------------------------------------
ohe2 = preprocessing.OneHotEncoder() 
isitmadfZ = EvNew.iloc[:,2:3]
isitmadfZ = ohe2.fit_transform(isitmadfZ).toarray()
isitmanewDF = pd.DataFrame(isitmadfZ)#$$$$$
isitmanewDF.columns = (['isitma_Doğalgaz sobalı', 'isitma_Isıtma yok', 'isitma_Jeotermal',
       'isitma_Kat Kaloriferi', 'isitma_Klimalı', 'isitma_Kombi Doğalgaz',
       'isitma_Kombi Fueloil', 'isitma_Kombi Kömür',
       'isitma_Merkezi (Pay Ölçer)', 'isitma_Merkezi Doğalgaz',
       'isitma_Sobalı', 'isitma_Yerden ısıtma'])
#--------------------------------------------
ohe3 = preprocessing.OneHotEncoder() 
sitedfZ = EvNew.iloc[:,3:4]
sitedfZ = ohe3.fit_transform(sitedfZ).toarray()
sitenewDF = pd.DataFrame(sitedfZ)#$$$$$
ohe3.get_feature_names(['site'])
sitenewDF.columns=(['site_Evet', 'site_Hayır'])
#--------------------------------------------
ohe4 = preprocessing.OneHotEncoder() 
katdfZ = EvNew.iloc[:,4:5]
katdfZ = ohe4.fit_transform(katdfZ).toarray()
katnewDF = pd.DataFrame(katdfZ)#$$$$$
ohe4.get_feature_names(['kat'])
katnewDF.columns = (['kat_1', 'kat_10', 'kat_11', 'kat_12', 'kat_13', 'kat_14',
       'kat_15', 'kat_16', 'kat_17', 'kat_18', 'kat_19', 'kat_2',
       'kat_20', 'kat_20-30', 'kat_21', 'kat_22', 'kat_25', 'kat_26',
       'kat_27', 'kat_29', 'kat_3', 'kat_4', 'kat_5', 'kat_6', 'kat_7',
       'kat_8', 'kat_9', 'kat_Bahçe dublex', 'kat_Bahçe katı',
       'kat_Düz giriş', 'kat_Kot 1', 'kat_Kot 2', 'kat_Kot 3',
       'kat_Müstakil', 'kat_Tam bodrum', 'kat_Villa tipi',
       'kat_Yarı bodrum', 'kat_Yüksek giriş', 'kat_Çatı Dubleks',
       'kat_Çatı Katı'])
#--------------------------------------------
ohe5 = preprocessing.OneHotEncoder()
odadfZ = EvNew.iloc[:,5:6]
odadfZ = ohe5.fit_transform(odadfZ).toarray()
odanewDF = pd.DataFrame(odadfZ)#$$$$$
ohe5.get_feature_names(['oda'])
odanewDF.columns =(['oda_1', 'oda_1+1', 'oda_2+1', 'oda_2+2', 'oda_3+1', 'oda_3+2',
       'oda_4+1', 'oda_4+2', 'oda_5', 'oda_5+1', 'oda_5+2', 'oda_5+3',
       'oda_5+4', 'oda_6+1', 'oda_6+2', 'oda_6+3', 'oda_7+1', 'oda_7+2',
       'oda_7+3', 'oda_8+', 'oda_Stüdyo'])
#--------------------------------------------
ohe6 = preprocessing.OneHotEncoder()
esyalidfZ = EvNew.iloc[:,8:9]
esyalidfZ = ohe6.fit_transform(esyalidfZ).toarray()
esyalinewDF = pd.DataFrame(esyalidfZ)#$$$$$
ohe6.get_feature_names(['esyali'])
esyalinewDF.columns= (['esyali_Boş', 'esyali_Eşyalı'])
###################################################

# 5.) Now I will scale the numerical values

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

mmscaler = MinMaxScaler()

scaler = StandardScaler()
m2df2 = EvNew.iloc[:,1:2]
m2scaled = mmscaler.fit_transform(m2df2)
m2scaledfC = pd.DataFrame(m2scaled, index=range(m2scaled.shape[0]))
m2scaledfC.columns = ['m2']

scaler2 = StandardScaler()
yasdf2 = EvNew.iloc[:,6:7]
yasscaled = mmscaler.fit_transform(yasdf2)
yasscaledC = pd.DataFrame(yasscaled, index=range(yasscaled.shape[0]))
yasscaledC.columns = ['yas']

scaler3 = StandardScaler()
banyosayisidf2 = EvNew.iloc[:,7:8]
banyosayisiscaled = mmscaler.fit_transform(banyosayisidf2)
banyosayisiscaledC = pd.DataFrame(banyosayisiscaled, index=range(banyosayisiscaled.shape[0]))
banyosayisiscaledC.columns = ['banyosayisi']

scaler4 = StandardScaler()
pricedf2 = EvNew.iloc[:,9:]
pricescaled = mmscaler.fit_transform(pricedf2)
pricescaledC = pd.DataFrame(pricescaled, index=range(pricescaled.shape[0]))
pricescaledC.columns = ['price']

### Train data

TrainData1 = pd.concat([ilcenewDF,m2scaledfC,isitmanewDF,sitenewDF,katnewDF,odanewDF,yasscaledC,banyosayisiscaledC,esyalinewDF], axis=1)
TrainData2 = pricescaledC

from sklearn.model_selection import train_test_split

mlr1_train, mlr1_test, mlr2_train, mlr2_test = train_test_split(TrainData1, TrainData2, test_size= 0.33, random_state=7)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(mlr1_train,mlr2_train)

mlr_pred = regressor.predict(mlr1_test)
from sklearn.metrics import r2_score
print("R2 = ", r2_score(mlr2_test,regressor.predict(mlr1_test)))

dumdum = np.arange(start=0, stop=109, step=1)


import statsmodels.api as sm
X = np.append(arr = np.ones((1715,1)).astype(int), values = TrainData1, axis=1)
X_list = TrainData1.iloc[:,dumdum].values

r_ols2 = sm.OLS(endog=TrainData2,exog=X_list.astype(float)).fit()

print(r_ols2.summary())

print("R2 = ", r2_score(mlr2_test,regressor.predict(mlr1_test)))


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
A = cross_val_score(regressor, TrainData1, TrainData2, cv=cv)
A = A = A.astype('int64')

TrainData1.describe()
################ Multi Linear Regression with feature selection

from sklearn.feature_selection import SelectKBest, f_regression
Atry = SelectKBest(f_regression, k=40).fit_transform(TrainData1,TrainData2)

##<------>
Atrydf = pd.DataFrame(Atry)
dumdum2 = np.arange(start=0, stop=39, step=1)
X_list2 = Atrydf.iloc[:,dumdum2].values
r_ols3 = sm.OLS(endog=TrainData2,exog=X_list2.astype(float)).fit()
print(r_ols3.summary())
##<------>

s_train, s_test, d_train, d_test = train_test_split(Atry, TrainData2, test_size= 0.33, random_state=7)

regressorNew = LinearRegression()
regressorNew.fit(s_train,d_train)

FinalPred = regressorNew.predict(s_test)

print("R2 = ", r2_score(d_test,regressorNew.predict(s_test)))


cv2 = ShuffleSplit(n_splits=5, test_size=0.33, random_state=7)
CrossValArr = cross_val_score(regressorNew, Atry, TrainData2, cv=cv2)


import seaborn as sns  

sns.heatmap(EvNew.corr(), annot = True)

sns.pairplot(EvNew,kind='reg')
############## Random Forest
from sklearn.ensemble import RandomForestRegressor

reg=RandomForestRegressor(n_estimators=100,random_state=7)
reg.fit(s_train,d_train)
randomForest_predict=reg.predict(s_test)

print("R2 = ", r2_score(d_test,reg.predict(s_test)))
cv3 = ShuffleSplit(n_splits=5, test_size=0.33, random_state=7)
CrossValArr2 = cross_val_score(reg, Atry, TrainData2, cv=cv3)

p = pd.DataFrame(randomForest_predict)


k = d_test

inverseresult = mmscaler.inverse_transform(p)
inverseresult.reshape(-1,1)
inversetest = mmscaler.inverse_transform(k)
inversetest = inversetest = inversetest.astype('int64')
inverseresult = inverseresult = inverseresult.astype('int64')

sns.swarmplot(x=EvNew['ilce'], y= EvNew['price'])
plt.xticks(rotation=90)
plt.ticklabel_format(style='plain', axis='y')
plt.yticks(np.arange(500000,10500000,step=500000))
plt.show()

sns.catplot(x='isitma', y='price', data=EvNew)
plt.xticks(rotation=90)
plt.ticklabel_format(style='plain', axis='y')
plt.yticks(np.arange(250000,10500000,step=500000))
plt.show()

sns.catplot(x='site', y='price', data=EvNew)
plt.xticks(rotation=90)
plt.ticklabel_format(style='plain', axis='y')
plt.yticks(np.arange(250000,10500000,step=500000))
plt.show()

sns.catplot(x='kat', y='price', data=EvNew)
plt.xticks(rotation=90)
plt.ticklabel_format(style='plain', axis='y')
plt.yticks(np.arange(250000,10500000,step=500000))
plt.show()

sns.catplot(x='oda', y='price', data=EvNew)
plt.xticks(rotation=90)
plt.ticklabel_format(style='plain', axis='y')
plt.yticks(np.arange(250000,10500000,step=500000))
plt.show()

sns.catplot(x='esyali', y='price', data=EvNew)
plt.xticks(rotation=90)
plt.ticklabel_format(style='plain', axis='y')
plt.yticks(np.arange(250000,10500000,step=500000))
plt.show()
