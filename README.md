
# **Stock Price Prediction Using Machine Learning and Time Series Analysis**

## **Overview**

This project focuses on predicting stock price movements based on historical financial data using machine learning and time series forecasting techniques. The goal is to build a model that predicts future stock prices based on past performance, trading volume, financial statements, and market sentiment analysis.

### **Key Technologies Used:**
- **Time Series Forecasting**: ARIMA, SARIMA, and LSTM for predicting future stock prices.
- **Machine Learning**: XGBoost and Random Forest for predicting trends based on various market features.
- **Data Processing**: Pandas for data manipulation and preprocessing.
- **Feature Engineering**: Use of technical indicators such as Moving Averages, RSI, and sentiment analysis.
- **Visualization**: Matplotlib and Plotly for data visualization and predictions.

## **Features**
- **Historical Stock Data**: Used historical stock prices, trading volumes, and financial statements as input for forecasting.
- **Technical Indicators**: Incorporated indicators like Moving Averages and RSI to improve model accuracy.
- **Sentiment Analysis**: Processed news sentiment to understand market trends and investor mood.
- **Visualization**: Visualized stock price predictions and trends with Plotly and Matplotlib.



### **Data Collection**:
- Data can be obtained from sources like Yahoo Finance or Alpha Vantage API. Alternatively, the data can be manually loaded from a CSV file (e.g., `Hist_BS_Fin_Stmt.csv`).

### **Data Preprocessing**:
- Clean the data by handling missing values and converting date columns into datetime objects.
- Select relevant columns such as `Close`, `Volume`, and technical indicators for input features.

### **Modeling**:
- **ARIMA/SARIMA**: These models are used for time series forecasting to predict future stock prices based on historical trends.
- **Machine Learning Models**: XGBoost and Random Forest models were used to predict stock price trends based on features such as financial data and technical indicators.
- **LSTM**: For capturing long-term dependencies and sequential patterns in stock price movements.

### **Model Evaluation**:
- Evaluate the models using metrics such as **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **R-squared**.
- Cross-validation is used to ensure that the models generalize well on unseen data.

### **Visualization**:
- Visualize the predictions, historical trends, and the correlation between features using libraries like **Matplotlib** and **Plotly**.

---

## **Usage**

### **Running the Jupyter Notebook**:
1. Open the `stock_price_prediction.ipynb` notebook in Jupyter or Google Colab.
2. Run the notebook step by step to load the data, preprocess it, build models, and visualize predictions.

### **Training the Model**:
You can train the models in the notebook by executing the following steps:
1. Load the dataset into a Pandas DataFrame.
2. Preprocess the data: handle missing values and perform feature engineering.
3. Train **ARIMA/SARIMA**, **XGBoost**, and **Random Forest** models.
4. Evaluate the models using performance metrics.
5. Visualize the predictions using **Plotly** and **Matplotlib**.

### **Prediction**:
Once the model is trained, you can use it to make predictions on future stock prices based on the provided input features.

---

## **Results and Evaluation**

The models were evaluated using the following metrics:
- **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors in predictions.
- **Mean Squared Error (MSE)**: Squares the error terms and provides more penalty to larger errors.
- **R-squared**: Indicates how well the model explains the variation in the target variable.

### **Key Findings**:
- XGBoost and Random Forest models performed well in predicting stock price trends.
- ARIMA and SARIMA models are effective for trend forecasting but less suitable for non-linear relationships in stock prices.
- LSTM showed great promise for capturing sequential dependencies in long time-series data.

---

## **Challenges**
- Predicting stock prices is inherently uncertain due to external factors such as market sentiment, political events, and macroeconomic changes.
- Models may be prone to overfitting if not carefully evaluated and validated.

---

## **Future Work**
- Incorporate additional data sources such as social media sentiment or macroeconomic indicators.
- Explore hybrid models combining ARIMA for trend forecasting with machine learning models like XGBoost.
- Develop a real-time prediction system that continuously updates predictions based on incoming data.

---

## **Conclusion**
This project demonstrates how machine learning and time series forecasting models can be used to predict stock price trends. While predicting stock prices with high accuracy is challenging, the project provides insights that could assist investors and traders in making informed decisions based on historical data, financial metrics, and market sentiment.

---

## **License**
This project is licensed under the MIT License - see the LICENSE file for details.

---

## **Acknowledgments**
Special thanks to the open-source community for libraries such as **Pandas**, **XGBoost**, and **TensorFlow** that enabled the successful implementation of this project.
