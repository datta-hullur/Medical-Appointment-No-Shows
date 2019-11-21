# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 15:13:09 2019

@author: dattatreya_rh
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

data=pd.read_csv('kaggle_data.csv')
data.info()
data.columns = ['patient_id', 'appointment_id', 'gender', 'scheduled_day', 
              'appointment_day', 'age', 'neighbourhood', 'scholarship', 'hypertension',
              'diabetes', 'alcoholism', 'handicap', 'sms_received', 'no_show']

data['no_show'] = data['no_show'].replace({'Yes':1,'No':0})
data['gender'] = data['gender'].replace({'M':1,'F':0})

# Removing the record where Age==-1 and age gretare than 110
data= data[(data.age >= 0) & (data.age <= 110)]


# Convert ScheduledDay and AppointmentDay from 'object' type to 'datetime64[ns]'
data['scheduled_day'] = pd.to_datetime(data['scheduled_day']).dt.date.astype('datetime64[ns]')
data['appointment_day'] = pd.to_datetime(data['appointment_day']).dt.date.astype('datetime64[ns]')

data.info()

#Create awaiting_time_days column
data['awaiting_time_days'] = (data.appointment_day - data.scheduled_day).dt.days # and convert timedelta to int


#Creating a new column appointment_dow (day of week appointment)
data['appointment_dow'] = data.scheduled_day.dt.weekday_name
data['appointment_dayofweek'] = data['appointment_day'].map(lambda x: x.dayofweek)

X=data.filter(['gender', 'awaiting_time_days', 'age', 'scholarship', 'hypertension',
              'diabetes', 'alcoholism', 'handicap','appointment_dayofweek', 'sms_received'])
y = data['no_show']

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_sc= sc.fit_transform(X)
#----------------------------------------------------------------------------------------------------
#fitting the Kmeans clustering model
#Elbow method to find otimal number of cluster
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,10):
    kmeans=KMeans(n_clusters=i,init='k-means++', max_iter=300,n_init=10, random_state=0)
    kmeans.fit(X_sc)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 10),wcss)
plt.title("The Elbow Method") 
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


#applying K means alogorithm to dataset for 3 cluaster got silohotte score of 0.25
kmeans=KMeans(n_clusters=2,init='k-means++', max_iter=300,n_init=10, random_state=0)
y_kmeans=kmeans.fit_predict(X_sc)


from sklearn.metrics import silhouette_score
silhouette_score(X_sc, y_kmeans, metric='cosine')

#calculating the dunn index for cluater seperation

def normalize_to_smallest_integers(labels):
    max_v = len(set(labels)) if -1 not in labels else len(set(labels)) - 1
    sorted_labels = np.sort(np.unique(labels))
    unique_labels = range(max_v)
    new_c = np.zeros(len(labels), dtype=np.int32)

    for i, clust in enumerate(sorted_labels):
        new_c[labels == clust] = unique_labels[i]

    return new_c

def dunn(labels, distances):
   
    labels = normalize_to_smallest_integers(labels)

    unique_cluster_distances = np.unique(min_cluster_distances(labels, distances))
    max_diameter = max(diameter(labels, distances))

    if np.size(unique_cluster_distances) > 1:
        return unique_cluster_distances[1] / max_diameter
    else:
        return unique_cluster_distances[0] / max_diameter


def min_cluster_distances(labels, distances):
    labels = normalize_to_smallest_integers(labels)
    n_unique_labels = len(np.unique(labels))

    min_distances = np.zeros((n_unique_labels, n_unique_labels))
    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i + 1, len(labels)):
            if labels[i] != labels[ii] and distances[i, ii] > min_distances[labels[i], labels[ii]]:
                min_distances[labels[i], labels[ii]] = min_distances[labels[ii], labels[i]] = distances[i, ii]
    return min_distances


def diameter(labels, distances):
    labels = normalize_to_smallest_integers(labels)
    n_clusters = len(np.unique(labels))
    diameters = np.zeros(n_clusters)

    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i + 1, len(labels)):
            if labels[i] == labels[ii] and distances[i, ii] > diameters[labels[i]]:
                diameters[labels[i]] = distances[i, ii]
    return diameters

from sklearn.metrics.pairwise import euclidean_distances
dunn_index = dunn(y_kmeans, euclidean_distances(X_sc))
print( dunn_index)  # got dunn index as 0.91123

data['clusters']=y_kmeans

data.to_excel("Data with clusters.xlsx")

cluster_0=data.loc[data['clusters']==0]
cluster_1=data.loc[data['clusters']==1]

cluster_0_desc=cluster_0.describe()
cluster_1_desc=cluster_1.describe()
#---------------------------------------------------------------------------------------------------------------
test_data=pd.read_excel('Medical_Appointment_test_data.xlsx')

test_data_1=test_data.copy()
test_data_1.columns = ['patient_id', 'appointment_id', 'gender', 'scheduled_day', 
              'appointment_day', 'age', 'neighbourhood', 'scholarship', 'hypertension',
              'diabetes', 'alcoholism', 'handicap', 'sms_received']
test_data_1['gender'] = test_data_1['gender'].replace({'M':1,'F':0})
test_data_1['scheduled_day'] = pd.to_datetime(test_data_1['scheduled_day']).dt.date.astype('datetime64[ns]')
test_data_1['appointment_day'] = pd.to_datetime(test_data_1['appointment_day']).dt.date.astype('datetime64[ns]')


#Create awaiting_time_days column
test_data_1['awaiting_time_days'] = (test_data_1.appointment_day - test_data_1.scheduled_day).dt.days # and convert timedelta to int


#Creating a new column appointment_dow (day of week appointment)
test_data_1['appointment_dow'] = test_data_1.scheduled_day.dt.weekday_name
test_data_1['appointment_dayofweek'] = test_data_1['appointment_day'].map(lambda x: x.dayofweek)

test_data_1_1=test_data_1.filter(['gender', 'awaiting_time_days', 'age', 'scholarship', 'hypertension',
              'diabetes', 'alcoholism', 'handicap','appointment_dayofweek', 'sms_received'])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
test_data_1_1 = sc.fit_transform(test_data_1_1)

# Fitting randome forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200)
y_pred_RFC=clf.predict(test_data_1_1)


test_data_1['RF_Predicted']=y_pred_RFC

y_kmeans_cls=kmeans.fit_predict(test_data_1_1)


test_data_1['K_means_predicted']=y_kmeans_cls

# matching the resukts of K means clustreing and Randome Forest
test_data_1['Predicted']=np.where(test_data_1['RF_Predicted']==test_data_1['K_means_predicted'], 
                                           'Show', 'No-show')
test_data["Predicted"]=test_data_1['Predicted']