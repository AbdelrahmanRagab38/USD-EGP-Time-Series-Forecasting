from flask import Flask, jsonify, request, render_template
import datetime
import joblib
import pandas as pd
from prophet import Prophet
import pickle
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    print(pd.__version__)
    # Render the index.html template
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv('USD_EGP Historical Data.csv')
    # Rename the 'Price' column to 'price'
    df.rename(columns={'Price': 'price'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.resample('D').ffill()
    df = df.drop('Vol.', axis=1)
    df = df.drop('Change %', axis=1)
    # Split into train and test sets
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    prophet_df = pd.DataFrame({'ds': df.index, 'y': df['price']})
    model = Prophet()
    model.fit(prophet_df)


    # Get input data
    data = request.form.to_dict()
    steps = int(data['steps'])

    
    
    # Load the models
    #ARIMA_Model = joblib.load('arima_model.pkl')
    Amodel = ARIMA(train['price'], order=(1,1,1))
    Amodel_fit = Amodel.fit()
    
    #Phrophet_Model = joblib.load("prophet_model.pkl")
    # Make predictions
    # ARIMA PREDICTION
    forecast = Amodel_fit.forecast(steps=steps)

    # ARIMA Plotting_Forecasting
    fig, ax = plt.subplots()
    ax.plot(df.index, df['price'], label='Actual')
    ax.plot(forecast, label='Forecast')
    ax.legend(loc='upper left')
    # Set the y-axis limits
    ax.set_ylim((16, 40))
    # Set the y-axis ticks
    ax.set_yticks(range(16, 40, 2))
    ax.set_title('ARIMA Forecast')
    plot_path1 = 'static/ARIMAForecast.png'
    fig.savefig(plot_path1)
    plt.show()

    # Phrophet Prediction
   # future = Phrophet_Model.make_future_dataframe(periods=steps)  # 1 year of future predictions
    #prophet_forecast = Phrophet_Model.predict(future)
    
    future = model.make_future_dataframe(periods=steps)
    Pforecast = model.predict(future)
    # Phrophet Plotting_Forecasting

    model.plot(Pforecast, xlabel='Date', ylabel='Price')
    # Set the y-axis limits
    plt.ylim((16, 40))
    # Set the y-axis ticks
    plt.yticks(range(16, 40, 2))
    # Save the plot to a file
    plot_path2 = 'static/PForecast.png'
    plt.savefig(plot_path2)
    plt.show()
    
    # Prophet figure:
    fig, ax = plt.subplots()
    model.plot(Pforecast, xlabel='Date', ylabel='Price', ax=ax)
    # Set the y-axis limits
    ax.set_ylim((16, 40))
    # Set the y-axis ticks
    ax.set_yticks(range(16, 40, 2))
    # Save the plot to a file
    plot_path2 = 'static/PForecast.png'
    fig.savefig(plot_path2)
    plt.show()



    # Render the template with the plot image
    return render_template('HomePage.html', plot_path1=plot_path1,plot_path2=plot_path2)


if __name__ == '__main__':
    app.run(debug=True)
