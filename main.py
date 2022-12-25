import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('dataset/IRIS.csv')
dataset["color"] = dataset["species"].map({"Iris-setosa": "r", "Iris-versicolor": "g", "Iris-virginica": "b"})

print(dataset.head())

plt.figure(figsize=(10, 6))
plt.scatter(dataset['sepal_length'], dataset['sepal_width'], c=dataset['color'])
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.show()

train, test = train_test_split(dataset, test_size=0.2)

clf = svm.SVC(kernel='rbf')
clf.fit(train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']], train['species'])

print(clf.score(test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']], test['species']))
