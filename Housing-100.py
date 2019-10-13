import pandas as pd
import numpy as np 
import tensorflow as tf 
import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn.model_selection import train_test_split
import seaborn as sns

#importing the dataset
df = pd.read_csv("C:/Users/haddy.haggan/MachineLearningHandsOn/Machine-Learning-Hands-On/Dataset/housing.csv")

#understanding the dataset
df.info()

#Checking the missing values

#quick representation of the data
df.head()

#a quick understanding of the catrgorical values in the dataset
df.ocean_proximity.value_counts()

#a quick represntation of the numerical values in the dataset
df.describe()

#Visualizaing the dataset
df.hist(bins=50, figsize=(20,15))
#splitting the data
x_train, x_test, y_train, y_test = train_test_split(df_x_train, df_y_train, test_size=0.2, random_state=20)
