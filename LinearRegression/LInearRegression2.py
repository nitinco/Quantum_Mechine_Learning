import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


#define data set as study time (X) and exam Score(y)
X=np.array([1,2,3,4,5,6,7,8,9]).reshape(-1,1) #features
y=np.array([3.4,1.1,2.1,2.4,6.6,7.8,8.8,9.0,7.5]) #target

modal=LinearRegression()
modal.fit(X,y)

y_pred = modal.predict(X)


#print cofficient 
print("slope cofficient ",modal.coef_)
print("Interseptor ",modal.intercept_)


#print errors
print("Mean square error",mean_squared_error(y, y_pred))
print("R^2",r2_score(y, y_pred))


plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Predicted Line')
plt.title('Linear Regression Example')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.legend()
plt.show()