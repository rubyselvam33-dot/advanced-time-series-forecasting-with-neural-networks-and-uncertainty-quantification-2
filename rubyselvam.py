{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "42217534-edd4-412c-81f5-2e842da8f0f3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    Series1   Series2   Series3\n",
      "0  0.993428  1.720920  2.595747\n",
      "1  1.026804  2.730860  1.900352\n",
      "2  3.882276  1.707253  2.030584\n",
      "3  6.877305  6.064813  2.468247\n",
      "4  4.549230  2.988741  3.321950\n",
      "Epoch 1/10\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\bfran\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\keras\\src\\layers\\rnn\\rnn.py:199: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
      "  super().__init__(**kwargs)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m12/12\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 8ms/step - loss: 165.0011\n",
      "Epoch 2/10\n",
      "\u001b[1m12/12\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 8ms/step - loss: 122.0434\n",
      "Epoch 3/10\n",
      "\u001b[1m12/12\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 9ms/step - loss: 96.4226\n",
      "Epoch 4/10\n",
      "\u001b[1m12/12\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 8ms/step - loss: 77.6669\n",
      "Epoch 5/10\n",
      "\u001b[1m12/12\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 8ms/step - loss: 67.2034\n",
      "Epoch 6/10\n",
      "\u001b[1m12/12\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 8ms/step - loss: 58.5029\n",
      "Epoch 7/10\n",
      "\u001b[1m12/12\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 8ms/step - loss: 56.0185\n",
      "Epoch 8/10\n",
      "\u001b[1m12/12\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 8ms/step - loss: 50.9739\n",
      "Epoch 9/10\n",
      "\u001b[1m12/12\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 8ms/step - loss: 48.0000\n",
      "Epoch 10/10\n",
      "\u001b[1m12/12\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 8ms/step - loss: 44.8166\n",
      "RMSE: 13.536433873487743\n",
      "90% Prediction Interval Coverage: 0.09375\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import LSTM, Dense, Dropout\n",
    "from sklearn.metrics import mean_squared_error\n",
    "np.random.seed(42)\n",
    "time_steps = 500\n",
    "t = np.arange(time_steps)\n",
    "trend = 0.05 * t\n",
    "seasonality = 10 * np.sin(2 * np.pi * t / 50)\n",
    "noise = np.random.normal(0, 2, time_steps)\n",
    "series1 = trend + seasonality + noise\n",
    "series2 = 0.8 * series1 + np.random.normal(0, 1, time_steps)\n",
    "series3 = 0.5 * series1 + np.random.normal(0, 1.5, time_steps)\n",
    "data = np.vstack([series1, series2, series3]).T\n",
    "df = pd.DataFrame(data, columns=[\"Series1\", \"Series2\", \"Series3\"])\n",
    "print(df.head())\n",
    "def create_sequences(data, window=20):\n",
    "    X, y = [], []\n",
    "    for i in range(len(data) - window):\n",
    "        X.append(data[i:i+window])\n",
    "        y.append(data[i+window, 0])\n",
    "    return np.array(X), np.array(y)\n",
    "X, y = create_sequences(data)\n",
    "split = int(0.8 * len(X))\n",
    "\n",
    "X_train, X_test = X[:split], X[split:]\n",
    "y_train, y_test = y[:split], y[split:]\n",
    "model = Sequential([\n",
    "    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),\n",
    "    Dropout(0.3),\n",
    "    LSTM(32),\n",
    "    Dropout(0.3),\n",
    "    Dense(1)\n",
    "])\n",
    "model.compile(optimizer=\"adam\", loss=\"mse\")\n",
    "model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)\n",
    "def mc_dropout_prediction(model, X, n_samples=100):\n",
    "    preds = []\n",
    "    for _ in range(n_samples):\n",
    "        preds.append(model(X, training=True).numpy().flatten())\n",
    "    return np.array(preds)\n",
    "\n",
    "mc_preds = mc_dropout_prediction(model, X_test)\n",
    "mean_prediction = mc_preds.mean(axis=0)\n",
    "lower_bound = np.percentile(mc_preds, 5, axis=0)\n",
    "upper_bound = np.percentile(mc_preds, 95, axis=0)\n",
    "rmse = np.sqrt(mean_squared_error(y_test, mean_prediction))\n",
    "\n",
    "coverage = np.mean(\n",
    "    (y_test >= lower_bound) & (y_test <= upper_bound)\n",
    ")\n",
    "print(\"RMSE:\", rmse)\n",
    "print(\"90% Prediction Interval Coverage:\", coverage)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "73199a8d-fbc5-4b69-8a6b-9cfd04b9b149",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
