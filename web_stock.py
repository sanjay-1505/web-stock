import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date, datetime
from keras.models import load_model
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.preprocessing import MinMaxScaler


st.title("Stock Price Prediction Web App 📈")

stock = st.text_input("Enter The Stock Symbol", "GOOGL").upper()

end = date.today().strftime("%Y-%m-%d")
start = "2015-01-01"

try:
    stock_data = yf.download(stock, start, end)
    if stock_data.empty:
        st.error(f"No data found for {stock}. Please enter a valid stock ticker.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

st.subheader("Stock Data")
st.write(stock_data)

if "Close" not in stock_data.columns:
    st.error("The dataset does not contain a 'Close' column.")
    st.stop()

splitting_len = int(len(stock_data) * 0.7)

x_test = stock_data[['Close']].iloc[splitting_len:].copy()

def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange', label="Moving Average")
    plt.plot(full_data.Close, 'b', label="Original Close Price")
    
    if extra_data:
        plt.plot(extra_dataset, label="Extra Data", color='g')
    
    plt.legend()
    
   
    plt.xlabel("Year")  
    plt.ylabel("Close Price") 
    
    return fig

for days in [250, 200, 100]:
    stock_data[f'MA_{days}_days'] = stock_data['Close'].rolling(days).mean()

st.subheader('Original Close Price, MA for 100 Days & MA for 250 Days')
st.pyplot(plot_graph((15,6), stock_data['MA_100_days'], stock_data, 1, stock_data['MA_250_days']))

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

try:
    model = load_model("Stock_Price_Model.keras")
except Exception as e:
    st.error(f"Error loading LSTM model: {e}")
    st.stop()

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame({
    'Original': inv_y_test.reshape(-1),
    'Predicted': inv_pre.reshape(-1)
}, index=stock_data.index[splitting_len+100:])

st.subheader("Original vs Predicted Values (LSTM)")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close Price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([stock_data.Close[:splitting_len+100], ploting_data], axis=0))
plt.legend(["Data-not Used", "Original Test Data", "Predicted Test Data"])

plt.xlabel("Year")
plt.ylabel("Close Price") 

st.pyplot(fig)

st.title("Stock Price Forecasting with Prophet")

prediction_periods = st.slider("Select Prediction Period (in days)", min_value=1, max_value=365, value=30)
period = prediction_periods * 1

if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = ['_'.join(col) for col in stock_data.columns]

close_column = [col for col in stock_data.columns if 'Close' in col][0]
df = stock_data[[close_column]].reset_index()
df.rename(columns={'Date': 'ds', close_column: 'y'}, inplace=True)

df['y'] = pd.to_numeric(df['y'], errors='coerce')  
df['ds'] = pd.to_datetime(df['ds']) 
df = df.dropna() 


st.subheader("Prepared Data for Prophet")
st.write(df.tail())

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=period) 
forecast = model.predict(future)

st.subheader("Forecast Data")
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

st.subheader("Forecast Plot")
fig = plot_plotly(model, forecast)

fig.update_layout(
    xaxis_title="Year", 
    yaxis_title="Close Price"  
)

st.plotly_chart(fig)


