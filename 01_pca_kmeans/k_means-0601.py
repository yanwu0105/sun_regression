
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import datetime as dt

import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import cluster, datasets, metrics
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import time
from datetime import datetime, timedelta
import matplotlib.dates as mdates  


# In[3]:


##3 load sql site data
import pyodbc

server = '168.14.xx.xx'
username = 'username'
password = 'password'
database = 'database name'
driver = '{ODBC Driver 13 for SQL Server}'
connectionString = 'DRIVER={0};PORT=1433;SERVER={1};DATABASE={2};UID={3};PWD={4}'.format(driver, server, database, username, password)
cnxn = pyodbc.connect(connectionString)
cursor = cnxn.cursor()

site="SELECT [CITY_ID],[CITY_NAME],[DISTRICT_ID],[DISTRICT_NAME],[REGION_ID],[CASE_NO],[CASE_NAME],[KWP]FROM [dbname].[dbo].[power]"
df_site=pd.read_sql(site, cnxn)


# In[4]:


# case_no & case_system data
d_m = pd.read_excel("d_m.xls")
d_m = d_m.iloc[:,0:2]
d_m.columns = ["CASE_ID","case_system"] 


# In[5]:


# inverter kwp data
inverterkwp = pd.read_csv("inverterkwp.csv")
dis = inverterkwp["DISTRICT_ID"]
reg = inverterkwp["REGION_ID"]
kwp = inverterkwp["Kwp"]
CASE_ID = []
inverter_Kwp = []
for i in range(len(dis)):
    dd = dis[i].split("D")
    dd = dd[1]
    rr = reg[i].split("R")
    rr = rr[1]
    caseid = dd + rr
    CASE_ID.append(caseid)
CASE_ID = pd.DataFrame(CASE_ID,columns=["CASE_ID"])
for l in range(len(kwp)):
    kk = kwp[l].lower()
    kk = kk.split("k")
    kk = float(kk[0])
    inverter_Kwp.append(kk)
inverter_Kwp = pd.DataFrame(inverter_Kwp,columns=["inverter_Kwp"])

inverterkwp = pd.concat([inverterkwp,CASE_ID,inverter_Kwp],axis=1)
inverterkwp = inverterkwp[["CASE_ID","INVERTER_ID","inverter_Kwp"]]


# In[6]:


# inverter days data
data = pd.read_csv("data_col_day.csv")
data_info = data.iloc[:,0:2]
data_1 = data.iloc[:,-30:364]


# In[7]:


data_info.CASE_ID = data_info.CASE_ID.astype(str)
data_info.INVERTER_ID = data_info.INVERTER_ID.astype(str)
inverterkwp.CASE_ID = inverterkwp.CASE_ID.astype(str)
inverterkwp.INVERTER_ID = inverterkwp.INVERTER_ID.astype(str)
data_2 = pd.merge(data_info,inverterkwp,on=["CASE_ID","INVERTER_ID"])
data_2 = pd.concat([data_2,data_1],axis=1)
data_2_drop = data_2.dropna() 


# In[8]:


data_3_info = data_2_drop.iloc[:,0:3]
data_3_data = data_2_drop.iloc[:,3:33]
data_3 = data_3_data.reset_index(drop=True)
data_3_info = data_3_info.reset_index(drop=True)


# In[9]:


data_3_t = data_3.T
data_3_inverter_kwp = data_3_info["inverter_Kwp"]
data_3_inverter_kwp = pd.DataFrame(data_3_inverter_kwp)
data_3_inverter_kwp = data_3_inverter_kwp.T


# In[10]:


# div kwp
d_4 = data_3_t.div(data_3_inverter_kwp.iloc[0], axis='columns')
data_4 = d_4.T


# In[11]:


plt.plot(d_4)


# In[12]:


# standardization
df_to_pca = (data_4-data_4.mean(axis=0))/data_4.std(axis=0)
#pca 績效
pca = PCA().fit(df_to_pca)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")


# In[13]:


# pca_ration
pca_ration = pd.DataFrame(pca.explained_variance_ratio_,columns=["ration"])
pca_ration.head(10)


# In[14]:


# set pca=2
pca = PCA(2)
da_pca = pca.fit_transform(df_to_pca)
da = pd.DataFrame(da_pca,columns=["a","b"])
data_targer = list(range(len(data_3)))
data_targer = pd.DataFrame(data_targer,columns=["target"])
da = pd.concat([da,data_targer],axis=1)
plt.scatter(da["a"],da["b"],c=da["target"])
plt.colorbar()


# In[15]:


# set k-means data type
a = np.array(da.a)
a = pd.DataFrame(a,columns=["a"])
b = np.array(da.b)
b = pd.DataFrame(b,columns=["b"])
da = pd.concat([a,b],axis=1)
X = da.values

# 迴圈
silhouette_avgs = []
ks = range(2, 15)
for k in ks:
    kmeans_fit = cluster.KMeans(n_clusters = k).fit(X)
    cluster_labels = kmeans_fit.labels_
    silhouette_avg = metrics.silhouette_score(X, cluster_labels)
    silhouette_avgs.append(silhouette_avg)

# 作圖並印出 k = 2 到 10 的績效
plt.bar(ks, silhouette_avgs)
plt.show()
print(silhouette_avgs)


# In[16]:


# kmeans plot
km = KMeans(n_clusters=10,random_state=0 )  #K=10群 隨機數10
y_pred_10 = km.fit_predict(X)
y_pred_10 = pd.DataFrame(y_pred_10,columns=["y_pred"])
da_y = pd.concat([da,y_pred_10],axis=1)

cc = pd.DataFrame(km.cluster_centers_)

c_list = ["#A7B352","#F7B243","#F16378","#6A3BBE","#5CEFED","#F6E39E","#3F89E2","#6ACC45","#DB0000","#00929D"]
plt.figure(figsize=(10, 6))
data_0 = da_y[da_y.y_pred==0]
p_1=plt.scatter(data_0.a,data_0.b, c=c_list[0])
data_1 = da_y[da_y.y_pred==1]
p_2=plt.scatter(data_1.a,data_1.b, c=c_list[1])
data_2 = da_y[da_y.y_pred==2]
p_3=plt.scatter(data_2.a,data_2.b, c=c_list[2])
data_3 = da_y[da_y.y_pred==3]
p_4=plt.scatter(data_3.a,data_3.b, c=c_list[3])
data_4 = da_y[da_y.y_pred==4]
p_5=plt.scatter(data_4.a,data_4.b, c=c_list[4])
data_5 = da_y[da_y.y_pred==5]
p_6=plt.scatter(data_5.a,data_5.b, c=c_list[5])
data_6 = da_y[da_y.y_pred==6]
p_7=plt.scatter(data_6.a,data_6.b, c=c_list[6])
data_7 = da_y[da_y.y_pred==7]
p_8=plt.scatter(data_7.a,data_7.b, c=c_list[7])
data_8 = da_y[da_y.y_pred==8]
p_9=plt.scatter(data_8.a,data_8.b, c=c_list[8])
data_9 = da_y[da_y.y_pred==9]
p_10=plt.scatter(data_9.a,data_9.b, c=c_list[9])

plt.scatter(cc[0],cc[1], c="#000000")
#plt.legend(handles = [p_1,p_2,p_3],labels = [1,2,3], loc = 'upper left')
#plt.legend(handles = [p_1,p_2,p_3,p_4,p_5,p_6],labels = [1,2,3,4,5,6], loc = 'upper left')
plt.legend(handles = [p_1,p_2,p_3,p_4,p_5,p_6,p_7,p_8,p_9,p_10],labels = [1,2,3,4,5,6,7,8,9,10], loc = 'upper left')


# In[17]:


df_site = df_site.iloc[1:,]
df_site = df_site.reset_index(drop=True)
df_site.columns = ['CITY_ID', 'CITY_NAME', 'DISTRICT_ID', 'DISTRICT_NAME', 'REGION_ID','CASE_ID', 'CASE_NAME', 'KWP']


# In[18]:


data_y = pd.concat([data_3_info,y_pred_10],axis=1)


# In[19]:


data_y.CASE_ID = data_y.CASE_ID.astype(str)
df_site.CASE_ID = df_site.CASE_ID.astype(str)
d_m.CASE_ID = d_m.CASE_ID.astype(str)

data_all = pd.merge(data_y,df_site, on='CASE_ID')


# In[20]:


mean_45 = pd.DataFrame(d_4.mean(),columns=["mean"])
mean_45.describe()
mean_45 = mean_45.reset_index(drop=True)

data_all_mean = pd.concat([data_all,mean_45],axis=1)


# In[52]:


invert_view = data_all_mean.groupby(["CITY_ID","CITY_NAME","CASE_ID"])
invert_view = invert_view['mean']
invert_view_before =invert_view.agg(['count','mean', 'std'])
invert_view_before.to_csv("invert_before.csv",encoding="utf_8_sig")


# In[103]:


data_all_sys = pd.merge(data_all_mean,d_m, on='CASE_ID')
data_sys_intkwp = pd.merge(data_all_sys,inverterkwp, on=["CASE_ID","INVERTER_ID"],how='left')
data_sys_intkwp.to_csv("data_sys_intkwp.csv",encoding="utf_8_sig",index=False)

