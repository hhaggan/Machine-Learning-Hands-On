import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plot
import string
import seaborn as sns

df_train = pd.read_csv("C:/Users/haddy.haggan/MachineLearningHandsOn/Machine-Learning-Hands-On/Dataset/titanic/train.csv")
df_test = pd.read_csv("C:/Users/haddy.haggan/MachineLearningHandsOn/Machine-Learning-Hands-On/Dataset/titanic/test.csv")
df_test_y = pd.read_csv("C:/Users/haddy.haggan/MachineLearningHandsOn/Machine-Learning-Hands-On/Dataset/titanic/gender_submission.csv")

#training dataset

df_train.info()
df_train.describe()
df_train.columns
df_train.shape

df_train.isnull().any()
df_train_x = df_train.copy()
df_train_x = df_train_x.drop(columns='Survived')
df_train_y = pd.Series(df_train["Survived"])
df_train_y = df_train_y.to_frame()

df_train_x.hist(bins = 50, figsize = (20,15))

#creating a new column for the title
titles = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess', 
                    'Don', 'Jonkheer']

def GetTitles(big_string, substrings):
    for substring in substrings:
        if str.find(big_string, substring) != -1:
            return substring
    return np.nan

def replace_titles(name):
    title=name['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if name['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title

df_train_x['Title']=df_train_x['Name'].map(lambda x: GetTitles(x, titles))
df_train_x['Title']=df_train_x.apply(replace_titles, axis=1)

#Modifying the column for the Cabin
cabins = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
df_train_x['Deck'] = df_train_x['Cabin'].map(lambda x: GetTitles(str(x), cabins))

#identifying the cateforical columns
cat_features = ["Parch", "Pclass", "Sex", "Embarked", "Title"]

#Creating new family_size column
df_train_x['Family_Size']=df_train_x['SibSp']+df_train_x['Parch']

#visualizing the data
sns.barplot(x= df_train_y["Survived"].value_counts().index, 
    y=df_train_y["Survived"].value_counts())

sns.barplot(x=df_train_x.Embarked.value_counts().index, 
    y=df_train_x.Embarked.value_counts())

sns.barplot(x=df_train_x.Pclass.value_counts().index, 
    y=df_train_x.Pclass.value_counts())

sns.barplot(x=df_train_x.Sex.value_counts().index, 
    y=df_train_x.Sex.value_counts())

sns.barplot(x=df_train_x.Parch.value_counts().index, 
    y=df_train_x.Parch.value_counts())

sns.barplot(x=df_train_x.Family_Size.value_counts().index, 
    y=df_train_x.Family_Size.value_counts())

sns.barplot(x="Sex", y="Survived", data=df_train, linewidth=2)

sns.countplot(x = "Sex", hue="Survived", data = df_train)
sns.countplot(x = "Pclass", hue="Survived", data = df_train)
sns.countplot(x = "Embarked", hue="Survived", data = df_train)

sns.countplot(x = "Sex", hue="Survived", data = df_train)
sns.countplot(x = "Pclass", hue="Survived", data = df_train)
sns.countplot(x = "Embarked", hue="Survived", data = df_train)

sns.catplot(x="Sex", y="Survived", col="Pclass", data=df_train, kind="bar")
sns.catplot(x="Sex", y="Survived", col="Embarked", data=df_train, kind='bar')

#running a correlation
corr = df_train_x.corr()
sns.heatmap(corr, annot=True)

#imputer 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(startegy='mean', copy=False)

imputer = imputer.fit(result)
imputer = imputer.transform(result)

#test dataset

df_test.info()
df_test.describe()
df_test.columns
df_test.shape

df_test.isnull().any()

df_test.hist(bins = 50, figsize = (20,15))

#replacing the titles for the test dataset

df_test['Title'] = df_test['Name'].map(lambda x: GetTitles(x, titles))
df_test['Title'] = df_test.apply(replace_titles, axis=1)

#working with Cabins
df_test['Deck'] = df_test['Cabin'].map(lambda x: GetTitles(str(x), cabins))

#Creating new family_size column
df_test['Family_Size']=df_test['SibSp']+df_test['Parch']

#Pipeline
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(df_train_x, df_train_y)

#KNN
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(df_train_x, df_train_y)

#Decision Tree
from sklearn. tree import DecisionTreeClassifier

#clf = DecisionTreeClassifier(random_state=42, criterion="entropy", max_depth=3, max_leaf_nodes=15)
clf = DecisionTreeClassifier(random_state=42, criterion="entropy", max_leaf_nodes=15)
clf.fit(df_train_x, df_train_y)