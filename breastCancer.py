import pandas as pd

# start by reading the dataset using pandas
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
print("----------------------------dataset----------------------------")
print(df)
# assingn the 30 features to a numpy Array x using a labelEncoder
from sklearn.preprocessing import LabelEncoder
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_ 
print("-----------------------classes with labelencoder transformation----------------------------------")
print(le.classes_)
