import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import string
import seaborn as sns

df_train = pd.read_csv("C:/Users/haddy.haggan/MachineLearningHandsOn/Machine-Learning-Hands-On/Dataset/titanic/train.csv")
df_test = pd.read_csv("C:/Users/haddy.haggan/MachineLearningHandsOn/Machine-Learning-Hands-On/Dataset/titanic/test.csv")
df_test_y = pd.read_csv("C:/Users/haddy.haggan/MachineLearningHandsOn/Machine-Learning-Hands-On/Dataset/titanic/gender_submission.csv")

# df = pd.concat([df_test, df_test_y], axis=0, join='outer')
df = df_test.set_index('PassengerId').join(df_test_y.set_index('PassengerId'))

df = df_train.append(df)
# df.drop_duplicates(subset ="PassengerId", keep='first', inplace = True)

# df_train.shape
# df_test.shape
# df.shape

#training dataset

# df.info()
# df.describe()
# df.columns
# df.shape

# df.isnull().any()

#df.hist(bins = 50, figsize = (20,15))

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

def cabin_sep(data_cabin):
    cabin_type = []

    for i in range(len(data_cabin)):

            if data_cabin.isnull()[i] == True: 
                cabin_type.append('M') #missing cabin = M 
            else:    
                cabin = data_cabin[i]
                cabin_type.append(cabin[:1]) 
            
    return cabin_type

df['Title']=df['Name'].map(lambda x: GetTitles(str(x), titles))
df['Title']=df.apply(replace_titles, axis=1)

#Modifying the column for the Cabin
cabins = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
df['Deck'] = df['Cabin'].map(lambda x: GetTitles(str(x), cabins))

#Creating new family_size column
df['Family_Size']=df['SibSp']+df['Parch']

#visualizing the data
sns.barplot(x= df["Survived"].value_counts().index, 
    y=df["Survived"].value_counts())

sns.barplot(x=df.Embarked.value_counts().index, 
    y=df.Embarked.value_counts())

sns.barplot(x=df.Pclass.value_counts().index, 
    y=df.Pclass.value_counts())

sns.barplot(x=df.Sex.value_counts().index, 
    y=df.Sex.value_counts())

sns.barplot(x=df.Parch.value_counts().index, 
    y=df.Parch.value_counts())

sns.barplot(x=df.Family_Size.value_counts().index, 
    y=df.Family_Size.value_counts())

sns.barplot(x="Sex", y="Survived", data=df, linewidth=2)

sns.countplot(x = "Sex", hue="Survived", data = df)
sns.countplot(x = "Pclass", hue="Survived", data = df)
sns.countplot(x = "Embarked", hue="Survived", data = df)

sns.countplot(x = "Sex", hue="Survived", data = df)
sns.countplot(x = "Pclass", hue="Survived", data = df)
sns.countplot(x = "Embarked", hue="Survived", data = df)

sns.catplot(x="Sex", y="Survived", col="Pclass", data=df, kind="bar")
sns.catplot(x="Sex", y="Survived", col="Embarked", data=df, kind='bar')

sns.catplot(x="Pclass", col="Embarked", data=df, saturation=.5, kind="count", ci=None)
sns.catplot(x="Sex", col="Embarked", data=df, saturation=.5, kind="count", ci=None)

pal = {1:"seagreen", 0:"gray"}
g = sns.FacetGrid(df,size=5, col="Sex", row="Survived", margin_titles=True, hue = "Survived",
                  palette=pal)
g = g.map(plt.hist, "Age", edgecolor = 'white')
g.fig.suptitle("Survived by Sex and Age", size = 25)
plt.subplots_adjust(top=0.90)

g = sns.FacetGrid(df,size=5, col="Sex", row="Embarked", margin_titles=True, hue = "Survived",
                  palette = pal)

g = g.map(plt.hist, "Age", edgecolor = 'white').add_legend()
g.fig.suptitle("Survived by Sex and Age", size = 25)
plt.subplots_adjust(top=0.90)

#working with the missing values
df.Embarked = df.Embarked.fillna('S')

preimputation=sns.kdeplot(data=df["Age"][(df["Survived"] == 0) & (
    df["Age"].notnull())], kernel='gau', color="Red", shade=True, legend=True)

copy5 = df.copy()
missing_age_rows2 = copy5.Age.isna()
age_by_pclass_SibSp = copy5.groupby(['Pclass', 'SibSp']).median()['Age']
age_by_pclass_SibSp[1].index.tolist()
age_by_pclass_SibSp[3][8] = age_by_pclass_SibSp[3][5]

copy5['Age'] = copy5.groupby(['Pclass', 'SibSp'])['Age'].apply(lambda x: x.fillna(x.median()))
copy5['Age'] = copy5.Age.fillna(11)

df = copy5

df['Deck'] = df['Deck'].fillna('M').astype(str).apply(lambda cabin: cabin[0])

df.Fare = df.Fare.fillna(df['Fare'].median())

#running a correlation
corr = df.corr()
sns.heatmap(corr, annot=True)

#replacing the titles for the test dataset

# df_test['Title'] = df_test['Name'].map(lambda x: GetTitles(x, titles))
# df_test['Title'] = df_test.apply(replace_titles, axis=1)

# #working with Cabins
# df_test['Deck'] = df_test['Cabin'].map(lambda x: GetTitles(str(x), cabins))

# #Creating new family_size column
# df_test['Family_Size']=df_test['SibSp']+df_test['Parch']

# #working with the missing values

# df_test.Fare = df_test.Fare.fillna(df_test['Fare'].median())

# test_age_by_pclass_SibSp = df_test.groupby(['Pclass', 'SibSp']).median()['Age']

# df_test['Age'] = df_test.groupby(['Pclass', 'SibSp'])['Age'].apply(lambda x: x.fillna(x.median()))

# idx = df[df['Cabin'] == 'T'].index
# df.loc[idx, 'Cabin'] = 'A'
# df.Cabin.value_counts()

# df_test['Deck'] = df_test['Deck'].fillna('M').astype(str).apply(lambda cabin: cabin[0])

# start

#removing some data
df['Pclass'] = df['Pclass'].astype(np.float64)
df['Parch'] = df['Parch'].astype(np.float64)
df['Family_Size'] = df['Family_Size'].astype(np.float64)
df = df.drop("Cabin", axis=1)
df = df.drop(columns='Name', axis=1)
df = df.drop(columns='PassengerId', axis=1)
df = df.drop(columns='SibSp', axis=1)
df = df.drop(columns='Ticket', axis=1)
df_x = df.drop(columns='Survived', axis=1)

#OneHotEncoder

# df_x = df.copy()
df_y = pd.Series(df["Survived"])
df_y = df_y.to_frame()

#identifying the cateforical columns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

target = df_y.copy()
features = df_x[['Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'Embarked', 'Title', 'Deck', 'Family_Size']].copy()

x_train, x_test, y_train, y_test = train_test_split(features, target, random_state=0)

numerical_features = features.dtypes == 'float'
categorical_features = ~numerical_features

preprocess = ColumnTransformer(transformers=[
    (numerical_features, make_pipeline(SimpleImputer(), StandardScaler())),
    (categorical_features, OneHotEncoder(handle_unknown='ignore'))])

model = Pipeline(steps=[
    ('preprocessor', preprocess),
    ('classifier', LogisticRegression())])

model.fit(x_train, y_train)
print("logistic regression score: %f" % model.score(x_test, y_test))

#Pipeline
from sklearn.ensemble import RandomForestClassifier

forest_clf = make_pipeline(preprocess, RandomForestClassifier(random_state=42))

forest_clf.fit(x_train, y_train)
print("Random Forest Classifier: %f" % forest_clf.score(x_test, y_test))

#KNN
from sklearn.neighbors import KNeighborsClassifier

knn_clf = make_pipeline(preprocess, KNeighborsClassifier(n_neighbors=3))
knn_clf.fit(x_train, y_train)

print("KNN  Classifier: %f" % knn_clf.score(x_test, y_test))

#Decision Tree
from sklearn. tree import DecisionTreeClassifier

clf = make_pipeline(preprocess, DecisionTreeClassifier(random_state=42, criterion="entropy", max_leaf_nodes=15))

clf.fit(x_train, y_train)
print("KNN  Classifier: %f" % clf.score(x_test, y_test))

