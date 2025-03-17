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

# **Final Model Rankings for Earthquake Prediction**

Based on **Mean Squared Error (MSE)** and **R² Score**, here are the final results:

| **Model**             | **MSE**   | **R² Score**    | **Ranking (Best to Worst)** |
|-----------------------|----------|---------------|----------------------|
| **Random Forest**     | **0.0126**   | **-0.1832**    | 🥇 (Best) |
| **Naïve Bayes**       | **0.0174**   | **-0.6337**    | 🥈 (Surprisingly strong) |
| **Decision Tree**     | **0.0287**   | **-1.6939**    | 🥉 (Overfitting detected) |
| **Linear Regression** | **0.1879**   | **0.0235**     | 🏅 (Decent for simple cases) |
| **LSTM (Deep Learning)** | **0.9675**   | *N/A*        | 🚧 (Potential, but underperforms) |
| **SVM (Support Vector Machine)** | **0.6981**   | **-2.6273**   | ❌ (Worst performance) |

---

## **Analysis & Conclusion**
### **🏆 Random Forest is the Best Model**
- It has the **lowest MSE (0.0126)** and the **highest R² (-0.1832)**.
- Handles **non-linear relationships** well and prevents **overfitting** by using multiple decision trees.

### **📈 Naïve Bayes Performs Surprisingly Well**
- Despite being a simple probabilistic model, its **MSE (0.0174)** is better than Decision Trees and even Linear Regression!
- This suggests that **categorical classification of earthquake magnitudes may work well** with a probabilistic approach.

### **🌳 Decision Tree Shows Overfitting**
- Its **MSE (0.0287)** is higher than Naïve Bayes and RF, and **R² (-1.6939)** indicates a poor fit.
- This suggests **overfitting**, meaning the tree is memorizing the training data instead of generalizing well.

### **📉 Linear Regression is Decent**
- **MSE (0.1879)** is high, but its **R² (0.0235)** is **better than Decision Trees and SVM**.
- Works for simple linear relationships, but earthquakes are highly non-linear!

### **🤖 LSTM Needs Improvement**
- **MSE (0.9675)** is the highest, meaning it struggled to make accurate predictions.
- This suggests that **deep learning needs more data, better tuning, or alternative architectures** (e.g., CNN-LSTM).

### **❌ SVM is the Worst Model**
- **MSE (0.6981)** and **R² (-2.6273)** are terrible.
- This confirms that **SVM is not suitable for earthquake magnitude prediction**.

---

## **Final Recommendation**
🚀 **For the best performance, use Random Forest.**  
📊 If computational resources are limited, **Naïve Bayes is a solid alternative.**  
🔍 If decision interpretability is key, **Decision Trees with pruning may improve results.**  
🤖 If using deep learning, **LSTM needs hyperparameter tuning & more data** to be competitive.  

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
