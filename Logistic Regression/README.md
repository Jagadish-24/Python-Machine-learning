# Logistic Regression

Logistic Regression is a statistical method for predicting binary outcomes from data. It is a type of regression analysis used for predicting the outcome of a categorical dependent variable based on one or more predictor variables.

## Key Concepts

- **Dependent Variable (Y)**: The binary outcome we are trying to predict (e.g., 0 or 1, True or False).
- **Independent Variable (X)**: The variable(s) used to make predictions.
- **Logistic Function (Sigmoid Function)**: The function used to map predicted values to probabilities:
  $$
  \sigma(z) = \frac{1}{1 + e^{-z}}
  $$
  where \( z = \beta_0 + \beta_1X \).

## Steps to Perform Logistic Regression

1. **Data Collection**: Gather data that includes both the dependent and independent variables.
2. **Data Preprocessing**: Clean the data, handle missing values, and perform any necessary transformations.
3. **Model Training**: Use statistical software or programming languages like Python to fit the logistic regression model to the data.
4. **Model Evaluation**: Assess the model's performance using metrics such as accuracy, precision, recall, and the ROC curve.
5. **Prediction**: Use the trained model to make predictions on new data.

