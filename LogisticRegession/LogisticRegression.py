import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

#load csv file or data file
data = pd.read_csv('LogisticRegession/insurance_data.csv')
print(data.head())

# plt.scatter(data.age,data.bought_insurance,marker='*',color='red')
# plt.show()

x_train,x_test,y_train,y_test=train_test_split(data[['age']],data.bought_insurance,train_size=0.8)


print(x_test)
print(x_train)
# modal=LogisticRegression()
# modal.fit()


