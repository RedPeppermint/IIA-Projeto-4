import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def plot_frontiers(X,y,model,feature_names,target_names,nclasses=3,pcolors="ryb"):

  x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
  y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                      np.arange(y_min, y_max, 0.02))
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
  plt.xlabel(feature_names[0])
  plt.ylabel(feature_names[1])

  # Plot the training points
  for i, color in zip(range(nclasses), pcolors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=target_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=20)
  plt.legend()
  
  
def load_data(fname, testdata=False):
    """Load CSV file with any number of consecutive features, 
       starting in column 0, where last column is the class
       
       Arguments:
           fname is a string with the filename 
           testdata is False if the csv file has the last column 
                    with class labels; and True if the csv file 
                    does not have that column (i.e., read a test 
                    file without class)
       Returns:
           table_X matrix with data
           table_y array with classes (if testdata=False)
           features array with features names
           target array with target name (if testdata=False)
    """
    df = pd.read_csv(fname)
    nc = df.shape[1] # number of columns
    if testdata:
        table_X = df.values # Convert dataframe to darray
        features = df.columns.values #get features names
        return table_X, features
    else:
        matrix = df.values # Convert dataframe to darray
        table_X = matrix [:, 0:nc-1] # get features 
        table_y = matrix [:, nc-1] # get class (last columns)           
        features = df.columns.values[0:nc-1] #get features names
        target = df.columns.values[nc-1] #get target name
        return table_X, table_y, features, target

    
def save_data(fname,prev):
    """Save a CSV file with the content of prev
    
       Arguments:
           fname is a string corresponding to the filename
           prev array to save in the CSV file
    """
    p = pd.DataFrame(prev)
    p.to_csv(fname,header=None,index=None)
    

def one_hot_encode_feature(table_X, col, feat_names):
    nrow = table_X.shape[0]
    ndim = len(np.unique(table_X[:,col]))
    enc = LabelEncoder()
    label_encoder = enc.fit(table_X[:, col])
    integer_classes = label_encoder.transform(label_encoder.classes_).reshape(ndim, 1)
    enc = OneHotEncoder()
    one_hot_encoder = enc.fit(integer_classes)
    # First, convert feature values to 0-(N-1) integers using label_encoder
    num_of_rows = nrow
    t = label_encoder.transform(table_X[:, col]).reshape(num_of_rows, 1)
    # Second, create a sparse matrix with col columns, each one indicating
    # whether the instance belongs to the class
    new_features = one_hot_encoder.transform(t)
    # Add the new features to table_X
    table_X = np.concatenate([table_X, new_features.toarray()], axis = 1)
    feat_names = np.concatenate([feat_names, label_encoder.classes_])
    # Eliminate converted columns
    table_X = np.delete(table_X, [col], 1)
    feat_names = np.delete(feat_names, [col])
    return table_X, feat_names


def encode_class(vect):
    enc = LabelEncoder()
    label_encoder = enc.fit(vect)
    #integer_classes = label_encoder.transform(label_encoder.classes_)
    t = label_encoder.transform(vect)
    return t, label_encoder.classes_


def encode_feature(vect):
    t = encode_class(vect)
    return t[0].astype(float)