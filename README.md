# LinearRegression

# Predictive Analysis of Diabetes Progression

This project uses a classic diabetes dataset to explore, train, and evaluate different machine learning regression models. The main objective is to predict a quantitative measure of disease progression one year after the initial study, based on a set of predictor variables.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%_Seaborn%_Scikit--Learn-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Concepts Covered](#concepts-covered)
3. [How to Run the Project](#how-to-run-the-project)
4. [Results and Conclusions](#results-and-conclusions)
5. [Notebook Structure](#notebook-structure)

---

## Project Overview
The project follows a standard data science workflow:
1.  **Data Loading**: Using the `diabetes` dataset from the `scikit-learn` library.
2.  **Exploratory Data Analysis (EDA)**: Visualizing the distribution of variables and the correlation between them.
3.  **Preprocessing**: Splitting the data into training and test sets.
4.  **Modeling**: Training four different regression models.
5.  **Evaluation**: Comparing the models using standard error metrics.
6.  **Interpretation**: Analyzing feature importance to understand the factors that most influence disease progression.

---

## Concepts Covered

This section details the main terms and machine learning techniques used in the notebook.

### Exploratory Data Analysis (EDA)
EDA is the process of investigating a dataset to summarize its main characteristics, often with visual methods. The goal is to better understand the data, identify anomalies, test hypotheses, and check assumptions before starting to model.
- **Histogram**: A graph that shows the frequency distribution of a numerical variable.
- **Correlation Heatmap**: A graphical representation of a correlation matrix. It shows the strength and direction of the linear relationship between pairs of variables. Values close to +1 or -1 indicate a strong correlation.

### Data Splitting (Train-Test Split)
This is a fundamental technique for evaluating the performance of a machine learning model. The data is divided into two subsets:
- **Training Set (`train`)**: Used to train the model.
- **Test Set (`test`)**: Used to evaluate the model's performance on "unseen" data. This simulates how the model will behave in a real-world scenario and helps to avoid *overfitting* (when the model memorizes the training data instead of learning to generalize).

### Regression Models
Regression is a supervised learning task aimed at predicting a continuous numerical value.
- **Linear Regression (`LinearRegression`)**: A model that attempts to establish a linear relationship (a straight line, in 2D) between the input variables (features) and the output variable (target).
- **K-Nearest Neighbors (KNN)**: A non-parametric algorithm that makes predictions based on the average of the `k` nearest neighbors in the feature space. It is simple and intuitive.
- **Decision Tree (`DecisionTreeRegressor`)**: A model that learns a series of "questions" (if-then-else) about the features to arrive at a prediction value. It is highly interpretable but prone to overfitting if not controlled.
- **Random Forest (`RandomForestRegressor`)**: An *ensemble* model that consists of constructing multiple decision trees during training. The final prediction is the average of the predictions from all individual trees. It generally outperforms a single decision tree and controls overfitting.

### Regression Evaluation Metrics
These are used to quantify the "error" or the difference between the values predicted by the model and the actual values.
- **Mean Squared Error (MSE)**: The average of the squared errors. It heavily penalizes large errors due to the squaring. $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
- **Mean Absolute Error (MAE)**: The average of the absolute value of the errors. It is easier to interpret as it is in the same unit as the target variable. $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
- **Mean Absolute Percentage Error (MAPE)**: The average of the absolute percentage error. It is useful for understanding the error in relative terms but can be problematic if the actual values are close to zero. $$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

### Model Interpretation
Techniques to understand how the model makes its predictions.
- **Coefficients (`.coef_`)**: In Linear Regression, the coefficients indicate the weight of each feature. A large positive coefficient means that an increase in the feature leads to a large increase in the prediction, and vice-versa.
- **Feature Importance (`.feature_importances_`)**: In tree-based models (like Random Forest), this metric quantifies the contribution of each feature to the reduction of the model's error. Features with higher importance are more decisive for the predictions.

---

## How to Run the Project

### Prerequisites
- Python 3.9+
- `pip` (Python package manager)

### Installation
1.  Clone this repository:
    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    cd seu-repositorio
    ```
2.  Create a `requirements.txt` file with the following content:
    ```
    pandas
    matplotlib
    seaborn
    scikit-learn
    ```
3.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Execution
1.  Open the `analise_diabetes.ipynb` notebook (or whatever name you gave it) in a Jupyter environment (Jupyter Notebook, JupyterLab, VS Code, etc.).
2.  Run the cells sequentially to replicate the analysis and results.

---

## Results and Conclusions
The comparative analysis of the models revealed that **Linear Regression** and **K-Nearest Neighbors (KNN)** showed the lowest error for this dataset. The single Decision Tree performed poorly, likely due to overfitting, a problem that was mitigated by the Random Forest model.

The most important features for predicting diabetes progression were consistently the **Body Mass Index (`bmi`)**, **Serum Triglycerides (`s5`)**, and **Mean Arterial Pressure (`bp`)**.

---

## Notebook Structure
The notebook is organized as follows:
1.  **Cells 1-2**: Installation and import of libraries.
2.  **Cells 3-9**: Data loading and Exploratory Analysis (data visualization, histograms, and correlation).
3.  **Cell 10**: Splitting the data into training and test sets.
4.  **Cells 11-16**: Training, evaluation, and interpretation of the **Linear Regression** model.
5.  **Cells 17-19**: Training and evaluation of the **KNN** model.
6.  **Cells 20-22**: Training and evaluation of the **Decision Tree** model.
7.  **Cells 23-26**: Training, evaluation, and interpretation of the **Random Forest** model.
8.  **Cells 27-30**: Detailed visualization of the Decision Tree and scatter plot of the results.
