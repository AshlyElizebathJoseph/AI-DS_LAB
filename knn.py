from sklearn.neighbors import KNeighborClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

#cLoading data
irisData= load_iris()

#Create feature and target arrays
X=irisData.data
y=irisData.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
  X,y,test_size=0.2,random_state=42)

a=int(input("Enter the number for k value: "))
knn=KNeighborsClassifier(n_neighbors=a)
print("Number of datas for prediction is: ",len(X_test))

knn.fit(X_train, y_train)
print("Prediction Accuracy: ",knn.score(X_test, y_test))
#predit on dataset which model has not seen before
x=knn.predict(X_test)
plt.plot(x)
print("Predictec: ",knn.predict(X_test))
