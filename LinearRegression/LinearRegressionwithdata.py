import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd
# import Book from ''


# data=pd.read_excel('./Book1.xlsx')
data = pd.read_csv('LinearRegression/Book1.csv')  # from project root

# print("cwd",os.getcwd())
# print(data.to_string())
print(data.head())