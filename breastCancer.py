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
le.transform(['M', 'B'])
print("-----------------------see that the class label Malignant or benign----------------------------------")
print(le.transform(['M', 'B']))
# divide the dataset into training and test in 80/20
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, stratify=y, random_state=1)

# combining transformers and estimators in a pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression())
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
test_acc= pipe_lr.score(X_test, y_test)
print(f'Test accuracy: {test_acc:.3f}')