# EV Battery Degradation Prediction Using Deep Learning

## Project Overview
Electric Vehicle (EV) batteries degrade over time due to charging cycles, temperature, fast charging, and aging effects. This project builds a deep learning model that predicts battery health and Remaining Useful Life (RUL) using battery sensor data. The system analyzes voltage, current, temperature, and charge cycles to estimate future battery capacity. 

## Model Architecture:

Implemented a multi-layer Long Short-Term Memory (LSTM) network using PyTorch. The architecture consists of:

    LSTM Layer
    Dropout Layer
    LSTM Layer
    Dense Layer
    Output Layer (1 neuron for SOH prediction)
    
 **Deployment:** Developed a web dashboard using Streamlit where users can upload historical battery logs via CSV and receive a real-time health percentage and risk assessment.

## Model Evaluation and Results
The model was trained for 50 epochs using the Adam optimizer with a learning rate of 0.0001, weight decay, and gradient clipping to ensure stable convergence. Performance was evaluated using Mean Squared Error (MSE) , Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

* **Training Loss (MSE):** 0.0111
* **Testing Loss (MSE):** 0.0123
