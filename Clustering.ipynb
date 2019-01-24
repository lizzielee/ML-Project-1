import numpy as np 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import os

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

PROJECT_ROOT_DIR = "."
IMAGES_PATH = PROJECT_ROOT_DIR

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

import pandas as pd 

def load_water_data(filename="data_matched_1.csv"):
  return pd.read_csv(filename)

water = load_water_data()
#print(water.head())
''' DONE
water.hist(bins=50, figsize=(20,15))
save_fig("attr_histogram_plots")
plt.show()
'''

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(water, test_size=0.2, random_state=42)
X_train = train_set.drop('Health_Score', axis=1)
y_train = train_set['Health_Score'].copy()
X_test = test_set.drop('Health_Score', axis=1)
y_test = test_set['Health_Score'].copy()


# Pipeline for numerical attr
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

# Transform numerical data of training
water_num = X_train.drop(['Site_ID', 'Date_Collected', 'Time_Collected'], axis=1)
water_num_tr = num_pipeline.fit_transform(water_num)
water_num_tr = pd.DataFrame(water_num_tr, columns=water_num.columns)
#print(water_num_tr.head())

# Transform numerical data of testing
water_test_num = X_test.drop(['Site_ID', 'Date_Collected', 'Time_Collected'], axis=1)
water_test_num_tr = num_pipeline.fit_transform(water_test_num)
water_test_num_tr = pd.DataFrame(water_test_num_tr, columns=water_test_num.columns)





# Euclidean distace measurement between x and y data points (x and y vector values)
  # Ignore last column label
def euclid_dist(x, y):
  tot_sq = 0
  for i in range(len(x)-1):
    diff = (x[i] - y[i])**2
    tot_sq+=diff
  return tot_sq**(1/2)

  # Spearson rank correlation measurement. Transformed so that it is 1-|abs(s)|
  # So high values indicate dissimilarity, closer to 0 is similarity.
def spearson_rank(x, y):
  mean_x = 0
  mean_y = 0
  for x_i in x[:-1]:
    mean_x += x_i
  mean_x = mean_x / (len(x)-1)
  for y_i in y[:-1]:
    mean_y += y_i
  mean_y = mean_y / (len(y)-1)
  sum_prod = 0
  sum_x = 0
  sum_y = 0
  for i in range(len(x)-1):
    sum_prod+= (x[i]-mean_x)*(y[i]-mean_y)
    sum_x += (x[i]-mean_x)**2
    sum_y += (y[i]-mean_y)**2
  denom = (sum_x*sum_y)**(1/2)
  return 1 - ((sum_prod/denom)**2)**(1/2) # 1-abs(sum_prod/denom)


# Function: K Means
# -------------
# K-Means is an algorithm that takes in a dataset and a constant
# k and returns k centroids (which define clusters of data in the
# dataset which are similar to one another).
class myKMeans:
  def __init__(self):
    self.k = 2 # Default k is 2 here
    self.random = 42 # random number to use for choosing points
    self.t = 1000 # number of iterations to use
    
# K-means algorithm
# make labels column for X in dataset
# choose random k samples to be k cluster ceters
# assign those to have labels matching 
# assign labels based on min distance to cluster centers
# find new centers based on mean of all labelled in cluster
# check if converged, new centers == old centers
# else, assign old = new and continue
# will return the labels of the dataset, cluster means and std devs. 

  def find_k_clusters(self, dataset_copy, k=6, t=50, type_dist="euclid"):
    # choose k points to be cluster centers
    dataset = dataset_copy.copy()
    dataset['labels'] = 0 # make label column for assignment
    clusters = dataset.sample(n=k, random_state=self.random)
    clusters = pd.DataFrame(clusters, columns=water_num_tr.columns)
    clusters = clusters.reset_index(drop=True)
    for i in range(t):
      for index, row in dataset.iterrows():
        min_dist = 1000000000
        for index_c, row_c in clusters.iterrows():
          if type_dist == "euclid":
            #print(row, row_c)
            dist = euclid_dist(row, row_c)
          else: # type = spearson_rank
            dist = spearson_rank(row, row_c)
          if dist < min_dist:
            min_dist = dist
            # set label value of current row to cluster index 
            dataset.set_value(index, 'labels', index_c) 
      # All assigned, so now must find the new cluster means
      new_clusters = clusters.copy()
      attr_std = clusters.copy()
      for i in range(k):
        new_clusters[i] = dataset.loc[dataset['labels']==i].mean(0)
        attr_std[i] = dataset.loc[dataset['labels']==i].std(0)
          
      # if new cluster means = old clusters, have found optimal labels
      new_clusters_v = new_clusters.values
      clusters_v = clusters.values

      # Return the labels and the cluster means
      if np.all(clusters_v == new_clusters_v):
        return dataset['labels'], clusters_v, attr_std
      
      else:
        clusters = new_clusters
        
    return dataset['labels'], clusters.values, attr_std
# 2=k, euclidian distance used for each. 
km2 = myKMeans()
labels, clusters, std_attr = km2.find_k_clusters(water_num_tr)

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

print("Clustering for k=6:")
water_num_tr.plot(kind="scatter", x="lon", y="lat", c=labels, alpha=0.5, cmap='Accent')
save_fig("myKMeans-6")
