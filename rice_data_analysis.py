import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("clean_rice_data.csv")
import_data = df[df['Element'] == 'Import quantity']

def display_menu():
    print("\nSelect a figure to display:")
    print("1. Graph Data using Excel Values")
    print("2. Correlation Heatmap")
    print("3. Linear Regression Model")
    print("4. Scaling the Data")
    print("5. Create Model and Predict Sales")
    print("6. Predictive Analytics – Using ARIMA")
    print("7. Simple Moving Average (SMA) Plot")
    print("0. Exit")

def plot_figure(choice):
    if choice == "1":
        print("\nFirst 5 Rows of Import Quantity Data:")
        print(import_data.head())

        plt.figure(figsize=(10, 5))
        plt.plot(import_data['Year'], import_data['Value'], marker='o')
        plt.title('Graph Data using Excel Values (Import Quantity Over Time)')
        plt.xlabel('Year')
        plt.ylabel('Import Quantity (t)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif choice == "2":
        numeric_df = import_data.select_dtypes(include=['float64', 'int64'])
        print("\nCorrelation Matrix:")
        print(numeric_df.corr())
        plt.figure(figsize=(6, 4))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap of Import Data')
        plt.tight_layout()
        plt.show()

    elif choice == "3":
        model = LinearRegression()
        X = import_data[['Year']]
        y = import_data['Value']
        model.fit(X, y)
        predictions = model.predict(X)

        plt.figure(figsize=(8, 5))
        plt.plot(import_data['Year'], import_data['Value'], label='Actual', marker='o')
        plt.plot(import_data['Year'], predictions, label='Predicted', linestyle='--')
        plt.title('Linear Regression: Import Quantity Over Time')
        plt.xlabel('Year')
        plt.ylabel('Import Quantity (t)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif choice == "4":
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(import_data[['Value']])
        import_data['Scaled_Value'] = scaled_values

        plt.figure(figsize=(10, 4))
        plt.plot(import_data['Year'], import_data['Scaled_Value'], color='purple')
        plt.title('Scaled Import Quantity Over Time')
        plt.xlabel('Year')
        plt.ylabel('Scaled Value (0-1)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif choice == "5":
        X = import_data[['Year']]
        y = import_data['Value']
        model = LinearRegression()
        model.fit(X, y)

        future_years = pd.DataFrame({'Year': range(2024, 2030)})
        future_predictions = model.predict(future_years)

        plt.figure(figsize=(10, 5))
        plt.plot(import_data['Year'], import_data['Value'], label='Historical', marker='o')
        plt.plot(future_years['Year'], future_predictions, label='Predicted', marker='x', linestyle='--')
        plt.title('Predict Sales using Linear Model')
        plt.xlabel('Year')
        plt.ylabel('Import Quantity (t)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif choice == "6":
        arima_series = pd.Series(import_data['Value'].values, index=pd.date_range(start='1961', periods=len(import_data), freq='Y'))
        model = ARIMA(arima_series, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=6)

        plt.figure(figsize=(10, 5))
        plt.plot(arima_series, label='Historical')
        plt.plot(pd.date_range(start=arima_series.index[-1] + pd.DateOffset(years=1), periods=6, freq='Y'), forecast, label='Forecast', color='red')
        plt.title('ARIMA Forecast of Rice Import Quantity')
        plt.xlabel('Year')
        plt.ylabel('Quantity (t)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif choice == "7":
        import_data_sorted = import_data.sort_values('Year')
        window_size = 5
        sma = import_data_sorted['Value'].rolling(window=window_size).mean()

        plt.figure(figsize=(10,5))
        plt.plot(import_data_sorted['Year'], import_data_sorted['Value'], label='Original', marker='o')
        plt.plot(import_data_sorted['Year'], sma, label=f'{window_size}-Year SMA', color='orange')
        plt.title(f'Simple Moving Average (Window={window_size}) of Import Quantity')
        plt.xlabel('Year')
        plt.ylabel('Import Quantity (t)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    else:
        print("Invalid choice. Please try again.")

while True:
    display_menu()
    user_choice = input("Enter your choice: ").strip()
    if user_choice == "0":
        print("Exiting. Goodbye!")
        break
    else:
        plot_figure(user_choice)
