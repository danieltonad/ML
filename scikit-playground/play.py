import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# print(X, "\n\n", y)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)

# print(predictions, score)

# save model
# joblib.dump(model, 'music-recommender.joblib')

# load model
# model = joblib.load('music-recommender.joblib')
# predictions = model.predict([[21,1]])
# print(predictions)

# grphical representation
from sklearn import tree
tree.export_graphviz(model, out_file='music-recommender.dot', 
    feature_names=['age', 'gender'], class_names=y.unique(),
    label='all', rounded=True, filled=True)
