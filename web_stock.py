import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go



st.title("Stock Price Prediction Web App")

stock=st.text_input("Enter The Stock Id","GOOGL")



end=date.today().strftime("%Y-%m-%d")
start="2015-01-01"


google_data=yf.download(stock,start,end)

model=load_model("Stock_Price_Model.keras")

st.subheader("Stock Data")
st.write(google_data)

splitting_len=int(len(google_data)*0.7)

x_test=pd.DataFrame(google_data.Close[splitting_len:])

def plot_graph(figsize,values,full_data,extra_data=0,extra_dataset=None):
    fig=plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.Close,'b')
    if extra_data:
        plt.plot(extra_data)
    
    
    return fig

st.subheader('Original Close Price and MA for 250 Days')
google_data['MA_for_250_days']=google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6),google_data['MA_for_250_days'],google_data,0))


st.subheader('Original Close Price and MA for 200 Days')
google_data['MA_for_200_days']=google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6),google_data['MA_for_200_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 Days')
google_data['MA_for_100_days']=google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6),google_data['MA_for_100_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 Days and MA for 250 days ')
st.pyplot(plot_graph((15,5),google_data['MA_for_100_days'],google_data,1,google_data['MA_for_250_days']))


from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(x_test[['Close']])


x_data=[]
y_data=[]


for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])


x_data,y_data=np.array(x_data),np.array(y_data)
predictions=model.predict(x_data)


inv_pre=scaler.inverse_transform(predictions)
inv_y_test=scaler.inverse_transform(y_data)



ploting_data=pd.DataFrame(
    {
        'original_test_data':inv_y_test.reshape(-1),
        'predictions':inv_pre.reshape(-1)
    },
    index=google_data.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted Values")
st.write(ploting_data)


st.subheader('Original Close Price vs Predicted Close Price')
fig=plt.figure(figsize=(15,6))
plt.plot(pd.concat([google_data.Close[:splitting_len+100],ploting_data],axis=0))
plt.legend(["Data-not Used","Original Test Data","Predicted Test Data"])
st.pyplot(fig)

#Future

prediction_periods = st.slider("Select Prediction Period (in days)", min_value=1, max_value=365, value=30)
#n_years = st.slider('Years of prediction:', 1, 4)
period = prediction_periods*1


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start, end)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(stock)
data_load_state.text('Loading data... done!')


st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())


st.subheader(f'Forecast plot for {prediction_periods} Days')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)