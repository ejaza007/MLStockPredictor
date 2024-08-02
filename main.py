import csv
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Initialize empty lists for dates and prices
dates = []
prices = []


# Function to read data from CSV and populate dates and prices lists
def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)  # Skip the header row
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[1]))  # Use month as a feature
            prices.append(float(row[1]))


# Function to predict prices using Support Vector Regression
def predict_prices(dates, prices, x):
    # Reshape dates for the model
    dates = np.reshape(dates, (len(dates), 1))

    # Scale the data
    scaler_dates = StandardScaler()
    scaler_prices = StandardScaler()
    dates = scaler_dates.fit_transform(dates)
    prices = scaler_prices.fit_transform(np.array(prices).reshape(-1, 1)).flatten()

    print(f"Dates shape: {dates.shape}, Prices shape: {prices.shape}")

    svr_lin = SVR(kernel="linear", C=1e3)
    svr_rbf = SVR(kernel="rbf", C=1e3, gamma=0.1)

    # Fit the models
    print("Fitting linear SVR")
    svr_lin.fit(dates, prices)
    print("Linear SVR fitted")

    print("Fitting RBF SVR")
    svr_rbf.fit(dates, prices)
    print("RBF SVR fitted")

    # Plot the data and predictions
    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_lin.predict(dates), color='blue', label='Linear Model')
    plt.plot(dates, svr_rbf.predict(dates), color='red', linestyle='dashed', label='RBF Model')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title("Support Vector Regression")
    plt.legend()

    # Show the plot
    plt.show()

    # Inverse transform the predictions to get the original scale
    x_scaled = scaler_dates.transform([[x]])
    return (
        scaler_prices.inverse_transform([svr_rbf.predict(x_scaled)])[0],
        scaler_prices.inverse_transform([svr_lin.predict(x_scaled)])[0]
    )


# Load data from the provided CSV file
get_data('AAPL.csv')

# Print dates and prices to check if they are loaded correctly
print(f"Dates: {dates}")
print(f"Prices: {prices}")

# Predict prices for the 20th date
print("Calling predict_prices function")
predict_price = predict_prices(dates, prices, 20)
print("predict_prices function returned")

if predict_price is not None:
    print(f"Predicted prices: {predict_price}")
else:
    print("Prediction was not successful due to an error in fitting one of the models.")
