import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
def compareAppandWebsite(data):
    # 1. Pairplot
    sns.pairplot(data)
    plt.show()

    # 2. Heatmap of correlation matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True)
    plt.title("Correlation Matrix")
    plt.show()



# Load the dataset
data = pd.read_csv("Ecommerce Customers.csv")

# Considering only the relevant columns for analysis
data = data[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership', 'Yearly Amount Spent']]
compareAppandWebsite(data)

# Splitting the data into features (X) and target variable (y)
X = data[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = data['Yearly Amount Spent']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a linear regression model
model = LinearRegression()

# Fitting the model to the training data
model.fit(X_train, y_train)

# 1. Intercept
intercept = model.intercept_
print("Intercept:", intercept)

# 2. Slope
slope = model.coef_
print("Slope:", slope)

# 3. Coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("*******************************************************************************")
print("Coefficients:")
print(coefficients)
print("*******************************************************************************")
# 4. Feature to invest more
max_coefficient_feature = coefficients.abs().idxmax()[0]
print("Feature the company should invest more is:", max_coefficient_feature)

# Find coefficients for 'Time on App' and 'Time on Website'
coefficient_time_on_app = coefficients.loc['Time on App', 'Coefficient']
coefficient_time_on_website = coefficients.loc['Time on Website', 'Coefficient']

# Compare coefficients
if abs(coefficient_time_on_app) > abs(coefficient_time_on_website):
    higher_feature = 'App'
    lower_feature = 'Website'
else:
    higher_feature = 'Website'
    lower_feature = 'App'

# Calculate how many times higher the higher coefficient is compared to the lower coefficient
times_higher = abs(coefficient_time_on_app) / abs(coefficient_time_on_website)

print(f"The coefficient of ({higher_feature}) is {times_higher:.2f} times higher than the coefficient of ({lower_feature}).")

print(f"So the company should focus thier efforts on :{higher_feature}")

print("*******************************************************************************")
# 5. Plot test predictions
y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True Values vs. Predictions")
plt.show()


# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R^2 Score:", r2)


