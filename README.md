# EarthquakeMagPredictor
Project for Semester 8 - KIIT (CSE)
# Earthquake Prediction Using Machine Learning and Deep Learning

## Project Overview
This project focuses on predicting earthquake magnitudes using machine learning and deep learning techniques. Two different models were implemented:
1. Traditional machine learning models such as **Linear Regression, SVM, and Random Forest**.
2. A **Long Short-Term Memory (LSTM)** neural network to forecast earthquake magnitudes based on historical seismic data.

The datasets used for this project were sourced from:
- **Dataset for traditional models**: [SOCR Data: California Earthquake Data](https://socr.ucla.edu/docs/resources/SOCR_Data/SOCR_Data_Earthquakes_Over3.html)
- **Dataset for LSTM Model**: [IRIS Seismic Data](https://ds.iris.edu/ds/)

## Project Structure
```
|-- earthquake-prediction/
    |-- Earthquake_Prediction_Project..ipynb   # Traditional ML models (Linear Regression, SVM, Random Forest)
    |-- LSTM Model   # Deep Learning model (LSTM)
        |-- LSTM_Model.ipynb
        |-- requirements.txt
        |-- edata.csv
    |-- Earthquake_Data.csv         # Earthquake dataset
    |-- requirements.txt  # Required dependencies
    |-- README.md         # Project documentation
```

## Dependencies
The project requires the following dependencies, listed in `requirements.txt`:

```
pandas~=2.2.1
matplotlib~=3.8.3
seaborn~=0.13.2
plotly~=5.19.0
numpy~=1.26.4
scikit-learn~=1.4.1.post1
tensorflow~=2.15.0
keras~=2.15.0
```

You can install them using:
```bash
pip install -r requirements.txt
```

## Notebook 1: Traditional Machine Learning Models
### Data Preprocessing:
- The dataset was loaded and preprocessed (handling missing values, normalizing magnitudes and depth).
- Time-ordered splitting (80% training, 20% testing) was applied.

### Models Implemented:
- **Linear Regression**
- **Support Vector Machine (SVM)**
- **Random Forest Regressor**

### Results:
| Model | Mean Squared Error (MSE) | R² Score |
|--------|----------------|-----------|
| Linear Regression | 0.1879 | 0.0235 |
| SVM | 0.6981 | -2.6273 |
| Random Forest | 0.0126 | -0.1832 |

- **Linear Regression:** Shows a reasonable MSE, indicating it captures some patterns in the data. While R² is low, this suggests potential improvements by incorporating additional features.
- **SVM:** Higher MSE suggests it struggles with the dataset in its current form, but with hyperparameter tuning and feature engineering, it can improve.
- **Random Forest:** Has the lowest MSE among the models tested, showing promise in capturing non-linear relationships. Further tuning could enhance its predictive power.

## Notebook 2: Deep Learning with LSTM
### Data Preprocessing:
- The dataset was processed similarly to Notebook 1.
- An LSTM-based sequence model was built to predict earthquake magnitudes.

### Model Architecture:
- **LSTM Layers**: Two stacked LSTM layers.
- **Dense Layer**: Fully connected layer for magnitude prediction.
- **Concatenation**: Merging depth information with LSTM output.
- **Loss Function**: Mean Squared Error (MSE).
- **Optimizer**: Adam.
- **Hyperparameter Tuning**: Using `keras_tuner`.

### Results:
| Metric | Value |
|-------------|----------------|
| MSE | 0.9675 |
| RMSE | 0.9836 |
| MAE | 0.8224 |

These results demonstrate the model's ability to learn from historical earthquake data. While there is room for improvement, LSTM networks have shown potential for capturing trends in seismic activity. Future refinements, such as incorporating additional geophysical features and fine-tuning hyperparameters, can further enhance accuracy.


## Visualizations
The project includes various visualizations:
- **Actual vs Predicted Magnitudes**: Scatter and trend plots comparing real and predicted values.
- **Prediction Error Analysis**: Line plots showing prediction errors.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/earthquake-prediction.git
   ```
2. Navigate into the project directory:
   ```bash
   cd earthquake-prediction
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run **Earthquake_Prediction_Project.ipynb** for traditional ML models.
5. Run **LSTM_Model.ipynb** for LSTM-based deep learning predictions.

## Conclusion
Among traditional machine learning models, **Linear Regression had the best R² score, meaning it explained the most variance**, while **Random Forest had the lowest MSE, indicating the least absolute error**. However, both models had limited predictive power, suggesting that earthquake magnitude prediction is highly complex and may require more advanced techniques.

The **LSTM model, despite having a higher MSE, is more suited for time-series forecasting** and can capture temporal dependencies better than traditional models. While its error metrics suggest room for improvement, deep learning remains a promising approach for earthquake prediction, particularly with further fine-tuning and additional features.

This project demonstrates that while machine learning can provide some insights into earthquake magnitude prediction, **deep learning models like LSTM offer a more scalable and adaptable solution** for capturing the complex patterns of seismic activity.

## Future Improvements
- Incorporate additional seismic features such as location coordinates and fault-line data.
- Use attention mechanisms for better feature weighting.
- Experiment with Transformer-based models for improved predictions.

---

### Author
**Akash Duttachowdhury**

---

Feel free to contribute and improve this project!
