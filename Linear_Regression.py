import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Retreving test data and reading it
data_path = "/content/drive/MyDrive/prodigy_ml_linearreg/housing_price_dataset.csv"   #path should be the location of the file you have saved
data = pd.read_csv(data_path)

y = {
    'price':data.Price
}
y = pd.DataFrame(y)

#creating dummies of neighborhood
dummies_Neighborhood = pd.get_dummies(data.Neighborhood)

#creating new dataset
new_data = data.drop(['YearBuilt','Neighborhood'],axis =1)
new_data = pd.concat([new_data,dummies_Neighborhood],axis =1)

#creatning traininng and testing datasets
X_train,X_test,y_train,y_test = train_test_split(new_data,y.price,test_size=0.2)
test_area = X_test.SquareFeet
#Standarising data 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Training Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
trained = lr.fit(X_train,y_train)

#Predicted price
saleprice_pred = trained.predict(X_test)
saleprice_pred = pd.DataFrame({'Square feet':test_area,'Predicition': saleprice_pred})
print(saleprice_pred)

#Displaying Accuracy
accuracy = lr.score(X_test,y_test)*100
print("Accuracy Score ",accuracy)
