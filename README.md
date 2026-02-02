Multivariate Time Series Forecasting with Neural Networks & Uncertainty Quantification






 Project Overview

This project implements a Multivariate Time Series Forecasting system using a Deep Learning LSTM network with Monte Carlo Dropout to estimate prediction uncertainty. The system not only predicts future values but also provides confidence intervals for better decision-making.

The dataset is synthetically generated to simulate real-world trend, seasonality, and noise.

 Key Features

 Synthetic multivariate time series generation

 LSTM-based deep learning forecasting

 Monte Carlo Dropout for uncertainty estimation

 90% Prediction Intervals

 RMSE evaluation metric

 Prediction Interval Coverage analysis

 Repository Structure
├── rubyselvam.py        # Main training & forecasting script
├── README.md            # Project documentation

 Dataset

The dataset contains three correlated time series:

Feature	Description
Series1	Target series (trend + seasonality + noise)
Series2	Correlated series (0.8 × Series1 + noise)
Series3	Correlated series (0.5 × Series1 + noise)

 Note: The dataset is generated programmatically inside the script.

 Model Architecture
Input (Window Size = 20)
   ↓
LSTM (64 units, return_sequences=True)
   ↓
Dropout (0.3)
   ↓
LSTM (32 units)
   ↓
Dropout (0.3)
   ↓
Dense (1 output)

 Project Workflow

Generate synthetic multivariate data

Create supervised learning sequences

Train-test split (80/20)

Train LSTM model

Apply Monte Carlo Dropout for inference

Compute prediction intervals (5th–95th percentile)

Evaluate using RMSE and Coverage

 Evaluation Metrics

RMSE (Root Mean Squared Error)
Measures forecasting accuracy.

90% Prediction Interval Coverage
Measures how often true values fall inside prediction intervals.

 How to Run
 Clone Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Install Requirements
pip install numpy pandas tensorflow scikit-learn

 Run the Project
python rubyselvam.py

 Sample Results
RMSE: 13.53
90% Prediction Interval Coverage: 0.09375

 Tech Stack

Python 3.10

NumPy

Pandas

TensorFlow / Keras

Scikit-learn

 Future Improvements

 Add ARIMA & Prophet baseline models

 Add CRPS uncertainty metric

 Visualization of prediction intervals

 Hyperparameter optimization

 Real-world datasets

 Author

Ruby
Multivariate Time Series Forecasting Project
 India

 License

This project is licensed under the MIT License.
