from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO
import base64

app = Flask(__name__)
model = load_model('best_model.keras')

def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)  # Close the figure to free memory
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    symbol = None
    table_html = None
    plots = {}

    if request.method == 'POST':
        symbol = request.form['symbol'].upper()
        start = '2012-01-01'
        end = '2024-12-31'

        try:
            data = yf.download(symbol, start=start, end=end)
            if data.empty:
                return render_template('index.html', error=f"No data found for symbol {symbol}")
            data.reset_index(inplace=True)  # Make Date a column instead of index

            # Format the date column to show only YYYY-MM-DD
            data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
                


            data_features = data[['Open', 'High', 'Low', 'Close', 'Volume']]

            df_train = pd.DataFrame(data_features[0: int(len(data)*0.80)])
            df_test = pd.DataFrame(data_features[int(len(data)*0.80): len(data)])

            # Moving Averages
            ma_50 = data['Close'].rolling(50).mean()
            ma_100 = data['Close'].rolling(100).mean()
            ma_200 = data['Close'].rolling(200).mean()

            # Plot 1: MA50
            fig1, ax1 = plt.subplots()
            ax1.plot(data['Close'], label='Close')
            ax1.plot(ma_50, label='MA50', color='red')
            ax1.legend()
            plots['ma50'] = plot_to_base64(fig1)

            # Plot 2: MA100
            fig2, ax2 = plt.subplots()
            ax2.plot(data['Close'], label='Close')
            ax2.plot(ma_50, label='MA50', color='red')
            ax2.plot(ma_100, label='MA100', color='blue')
            ax2.legend()
            plots['ma100'] = plot_to_base64(fig2)

            # Plot 3: MA200
            fig3, ax3 = plt.subplots()
            ax3.plot(data['Close'], label='Close')
            ax3.plot(ma_100, label='MA100', color='red')
            ax3.plot(ma_200, label='MA200', color='blue')
            ax3.legend()
            plots['ma200'] = plot_to_base64(fig3)

            # Data prep
            scaler = MinMaxScaler(feature_range=(0,1))
           
            past_100 = df_train.tail(100)
            df_test = pd.concat([past_100, df_test], ignore_index=True)
            input_data = scaler.fit_transform(df_test)

            x = []
            y = []

            for i in range(100, input_data.shape[0]):
                x.append(input_data[i-100:i])
                y.append(input_data[i,0])
            x, y = np.array(x), np.array(y)

            
            predicted = model.predict(x)

            scale = 1/scaler.scale_[0]
            predicted = predicted * scale
            y = y * scale

            # Plot 4: Prediction
            fig4, ax4 = plt.subplots()
            ax4.plot(predicted, label='Predicted', color='red')
            ax4.plot(y, label='Actual', color='green')
            ax4.legend()
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Price')
            plots['prediction'] = plot_to_base64(fig4)

            # Last 10 rows table
            table_html = df_test.tail(10).to_html(classes='data', header="true", index=False)

            
            table_data = data.tail(100).copy()
            table_html = table_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].to_html(
            classes='table table-striped',
            header="true",
            index=False,
            float_format='%.4f',  # 4 decimal places for prices
            formatters={'Volume': '{:,.0f}'.format}  # Comma format for volume
        )

        except Exception as e:
            return render_template('index.html', error=f"Error processing data: {str(e)}")

    return render_template('index.html', symbol=symbol, data=table_html, plots=plots)

if __name__ == '__main__':
    app.run(debug=True)