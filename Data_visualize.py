# This code describes the dataset and features used in this project

# Written By: Khaula
# Feb 12th, 2020

# Standard imports





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix


dataset = pd.read_csv('~/AllDataPrep_Feb10.csv', header=0) # this is the preprocessed files in which missing values were filled by Linear Interpolation for PM2.5

# print('dataset data types', dataset.dtypes)
print('dataset.shape', dataset.shape)
#print(dataset.head(5))

# Dropping Dummy variables:
dataset.drop([ 'Date_Time', 'LAT','LON'],  axis=1, inplace=True)

#print(dataset.head(10))

#print("Data shape", dataset.shape)
#print(dataset.dtypes)
#print(dataset.describe())

#Using Pearson Correlation

#plt.figure(figsize=(20,12))
#cor = dataset.corr()
#sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
#plt.title("Features Correlation", y=-0.1)
#plt.savefig('/home/khaula/Desktop/PM10_LSTM_From_Jan1/GwangjuStationsData/Data_Prep/FeatureCorrelation2.png')
#plt.show()



dataset.drop([ 'WDIR(deg)', 'CMAQ_H2SO4(pptv)','CMAQ_CO(ppmv)', 'CMAQ_O3(ppbv)','CMAQ_NH3(ppbv)', 'CMAQ_HNO3(pptv)','CMAQ_N2O5(pptv)', 'CMAQ_NO3(pptv)',  'CMAQ_SO2(ppbv)','CMAQ_PM10(ug/m3)', 'CMAQ_PM2.5(ug/m3)','CMAQ_SS(ug/m3)','CMAQ_ASO4(ug/m3)','CMAQ_OA(ug/m3)','CMAQ_DUST(ug/m3)','CMAQ_NO2(ppbv)', 'CMAQ_NMVOC(ppbv)','CMAQ_NO(ppbv)', 'EMIS_PM25(g/s)', 'EMIS_PM10(g/s)', 'EMIS_NOx(mol/s)','EMIS_CO(mol/s)', 'EMIS_NMVOC(mol/s)', 'EMIS_SO2(mol/s)'], axis=1, inplace=True)
#print(dataset.head(5))


# Uncomment these commads to show the data distribution:
#PM25 = dataset['OBS_PM2.5(ug/m3)']
#PM10 = dataset['OBS_PM10(ug/m3)']
#plt.scatter(PM25, PM10 )
#plt.show()

print(dataset.describe())

# Plotting correlation between each feature ScatterMatrixCorrelation, Using pandas library:
plt.figure(figsize=(30,20))
scatter_matrix(dataset)

plt.savefig('~/ScatterMatrixCorrelation3.png')
plt.show()


# Use pandas library to plot each feature distribution: Histogram
#features = dataset.loc[:, 'OBS_PM2.5(ug/m3)': 'CMAQ_OTHR(ug/m3)']
#features.hist(figsize=(16, 10))
#plt.savefig('~/HistogramAll.png')
#plt.show()

# Use pandas library to plot each feature Correlation:

#Using Pearson Correlation

#plt.figure(figsize=(16,10))
#cor = dataset.corr()
#sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
#plt.title("Selected Features Correlation", y=-0.1)
#plt.savefig('~/Selected_FeatureCorrelation.png')
#plt.show()




'''
Results:
Data shape (2609373, 40)
OBS_PM2.5(ug/m3)     float64
OBS_PM10(ug/m3)      float64
OBS_O3(ppbv)         float64
OBS_NO2(ppbv)        float64
OBS_CO(ppmv)         float64
OBS_SO2(ppbv)        float64
T(K)                 float64
RH(%)                float64
WDIR(deg)            float64
WSPD(m/s)            float64
PBL(m)               float64
PRECIP(mm/hr)        float64
ZRUF(m)              float64
CMAQ_ANH4(ug/m3)     float64
CMAQ_ANO3(ug/m3)     float64
CMAQ_ASO4(ug/m3)     float64
CMAQ_CO(ppmv)        float64
CMAQ_DUST(ug/m3)     float64
CMAQ_EC(ug/m3)       float64
CMAQ_H2SO4(pptv)     float64
CMAQ_HNO3(pptv)      float64
CMAQ_N2O5(pptv)      float64
CMAQ_NH3(ppbv)       float64
CMAQ_NMVOC(ppbv)     float64
CMAQ_NO(ppbv)        float64
CMAQ_NO2(ppbv)       float64
CMAQ_NO3(pptv)       float64
CMAQ_O3(ppbv)        float64
CMAQ_OA(ug/m3)       float64
CMAQ_OTHR(ug/m3)     float64
CMAQ_PM10(ug/m3)     float64
CMAQ_PM2.5(ug/m3)    float64
CMAQ_SO2(ppbv)       float64
CMAQ_SS(ug/m3)       float64
EMIS_CO(mol/s)       float64
EMIS_NMVOC(mol/s)    float64
EMIS_NOx(mol/s)      float64
EMIS_PM10(g/s)       float64
EMIS_PM25(g/s)       float64
EMIS_SO2(mol/s)      float64
dtype: object
       OBS_PM2.5(ug/m3)  OBS_PM10(ug/m3)  OBS_O3(ppbv)  OBS_NO2(ppbv)  OBS_CO(ppmv)  OBS_SO2(ppbv)  ...  EMIS_CO(mol/s)  EMIS_NMVOC(mol/s)  EMIS_NOx(mol/s)  EMIS_PM10(g/s)  EMIS_PM25(g/s)  EMIS_SO2(mol/s)
count      2.609373e+06     2.609373e+06  2.609373e+06   2.609373e+06  2.609373e+06   2.609373e+06  ...    2.609373e+06       2.609373e+06     2.609373e+06    2.609373e+06    2.609373e+06     2.609373e+06
mean       2.635704e+01     4.708050e+01  2.669235e+01   2.388639e+01  4.995652e-01   4.464723e+00  ...    1.959283e+01       5.373999e+00     9.307626e+00    5.526676e+01    1.902123e+01     1.313184e+00
std        1.591240e+01     2.902623e+01  1.964593e+01   1.687691e+01  2.479343e-01   3.554822e+00  ...    2.522161e+01       5.784406e+00     6.805147e+00    6.122826e+01    2.649554e+01     3.218474e+00
min        0.000000e+00     0.000000e+00  0.000000e+00   0.000000e+00  0.000000e+00   0.000000e+00  ...    2.218339e-01       2.776177e-02     2.364747e-02    9.257047e-02    9.257047e-02     0.000000e+00
25%        1.500000e+01     2.800000e+01  1.100000e+01   1.100000e+01  3.000000e-01   3.000000e+00  ...    5.764527e+00       1.848659e+00     3.399968e+00    1.607308e+01    6.331139e+00     1.031500e-01
50%        2.328571e+01     4.100000e+01  2.400000e+01   2.000000e+01  5.000000e-01   4.000000e+00  ...    1.221520e+01       4.092754e+00     7.340751e+00    3.942593e+01    1.343157e+01     2.966799e-01
75%        3.400000e+01     6.000000e+01  3.800000e+01   3.300000e+01  6.000000e-01   5.000000e+00  ...    2.379851e+01       6.345867e+00     1.431113e+01    8.202190e+01    2.515547e+01     1.078466e+00
max        2.400000e+02     7.320000e+02  2.060000e+02   2.720000e+02  9.000000e+00   3.290000e+02  ...    1.344164e+02       3.241810e+01     2.401797e+01    4.613462e+02    2.061773e+02     1.704534e+01


'''

