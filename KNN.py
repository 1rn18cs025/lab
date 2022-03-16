from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier as KNN
import numpy as np
from sklearn.model_selection import train_test_split

iris = load_iris()
print('Iris Data: ',iris.data)
print('Iris Features: ', iris.feature_names)
print('Iris Target: ', iris.target_names)

for i in range(len(iris.target_names)):
    print("[{0}] : [{1}]".format(i, iris.target_names))

x_train, x_test, y_train, y_test = train_test_split(iris['data'],iris['target'],random_state = 0)
knn = KNN(n_neighbors = 3)
knn.fit(x_train, y_train)
print()
for i in range(len(x_test)):
    x = x_test[i]
    x_new = np.array([x])
    pred = knn.predict(x_new)
    print('X:',x_new,'\nActual: {0} => {1}\tPredicted: {2} => {3}'.format(y_test[i],iris['target_names'][y_test[i]], pred[0], iris['target_names'][pred][0]))
    
print('Test Score(Accuracy): {:.2f}'.format(knn.score(x_test, y_test)))







