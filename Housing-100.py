import pandas as pd
import numpy as np 
import tensorflow as tf 
import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

#importing the dataset
df = pd.read_csv("C:/Users/haddy.haggan/MachineLearningHandsOn/Machine-Learning-Hands-On/Dataset/housing.csv")

#understanding the dataset
df.info()

#Checking the missing values
df.isnull().any()

#quick representation of the data
df.head()

#a quick understanding of the catrgorical values in the dataset
df.ocean_proximity.value_counts()

#a quick represntation of the numerical values in the dataset
df.describe()

#Visualizaing the dataset
df.hist(bins=50, figsize=(20,15))

#splitting the data
df_train, df_test = train_test_split(df, test_size=0.2, random_state=20)

#creating categories of the income dataset
df["income_Cat"] = pd.cut(df["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1,2,3,4,5])
df.income_Cat.hist()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(df, df["income_Cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

strat_test_set["income_Cat"].value_counts() / len(strat_test_set)

#Visualizing the samples
s_train_c = strat_train_set.copy()

s_train_c.plot(kind="scatter", x="longitude", y ="latitude")
s_train_c.plot(kind="scatter", x="longitude", y ="latitude", alpha=0.1)
s_train_c.plot(kind="scatter", x="longitude", y ="latitude", alpha=0.4,
    s=s_train_c["population"]/100, label="population", figsize=(10,7), 
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)

plt.legend()

#Visuazling the correlation
corr_matrix = s_train_c.corr()

sns.heatmap(corr_matrix)
corr_matrix["median_house_value"].sort_values(ascending=False)

#another way to do the correlation
# attributes = ["median_house_value", "median_income", "total_rooms", 
#     "housing_median_age"]

# scatter_matrix(df[attributes], figsize=(12,8))

#Checking the missing values
df.isnull().any()

imputer = SimpleImputer(strategy="median")
df_temp = df.drop("ocean_proximity", axis=1)
imputer.fit(df_temp)
X = imputer.transform(df_temp)
df_f = pd.DataFrame(X, columns=df_temp.columns)

df_f.isnull().any()

df_cat = df[["ocean_proximity"]]

ordinal_encoder = OrdinalEncoder()
df_cat_encoded = ordinal_encoder.fit_transform(df_cat)

cat_encoder = OneHotEncoder()
df_cat_Ohot = cat_encoder.fit_transform(df_cat)