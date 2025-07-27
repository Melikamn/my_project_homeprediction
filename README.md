# my_project_homeprediction
My First Machine Learning Project: Predicting House Prices 🏡 📊 What Did I Build?  A model to predict house prices using the famous Kaggle dataset and the XGBoost algorithm.
When I started building this project, I initially thought I’d go with a simple model like **Linear Regression**—because it’s easier and widely used in educational resources.

But halfway through, I got curious—maybe there are more powerful algorithms that are used more often in real-world projects? So I tried Decision Tree and Random Forest, played around with them a bit, and then asked myself: “Is there something even better than these two?”

And that’s when I discovered ✨ XGBoost ✨

A powerful Gradient Boosting model that's widely used in Kaggle competitions. And here are some of the things I learned from this small project: 🧐

* Working with Pandas and performing Exploratory Data Analysis
* Using XGBoost for regression tasks
* Calculating R² and MAE to evaluate model performance
* Realizing the importance of Feature Engineering, even with strong models
* The actual code:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb

df = pd.read_csv("C:\\Users\\pc-iran\\Downloads\\train.csv")
df.shape

features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
X = df[features]
y = df['SalePrice']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    objective="reg:squarederror"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")


plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_val, y=y_pred)
plt.xlabel("REAL PRICE")
plt.ylabel("PREDICTED PRICE")
plt.title("PREDICTION OF HOUSE PRICE WITH XGBOOST")
plt.grid(True)
plt.show()

xgb.plot_importance(model)
plt.title("IMPORTANCE OF FEATURES IN XGBoost")
plt.show()


✨ This is just the beginning!
My goal is to build practical projects and document my learning process. The next step is to optimize the model and gain more experience with more complex projects.
