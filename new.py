import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import xgboost as xgb


# Assuming you have a dataset in a CSV file with columns: T, TM, Tm, SLP, H, VV, V, VM, PM2.5
# Replace 'your_dataset.csv' with the actual filename.
data = pd.read_csv('Real_Combine.csv')


# Handling missing values by imputing with column means
data.fillna(data.mean(), inplace=True)

# Extracting features (X) and target (y)
X = data[['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM']].values
y = data['PM 2.5'].values

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgbr = xgb.XGBRegressor()
xgbr.fit(X_train,y_train)

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluating the model
from sklearn.metrics import mean_squared_error, r2_score
y_pred = xgbr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")



# Saving the model as a pickle file

file = open('xgboost.pkl', 'wb')

# dump information to that file
pickle.dump(xgbr, file)


