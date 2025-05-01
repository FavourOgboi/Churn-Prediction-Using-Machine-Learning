# **Customer Churn Prediction - ConnectTel**

## **Problem Statement**
Customer churn is a major challenge for telecommunications companies. Losing customers can significantly impact revenue and growth. The goal of this project is to develop a machine learning model that accurately predicts customer churn, allowing ConnectTel to implement targeted retention strategies.

## **Dataset Description**
The dataset contains multiple features related to customer demographics, services subscribed, and billing information. Key attributes include:

- **CustomerID**: Unique identifier for each customer
- **Demographics**: Gender, SeniorCitizen, Partner, Dependents
- **Subscription Details**: PhoneService, InternetService, MultipleLines
- **Additional Services**: OnlineSecurity, OnlineBackup, TechSupport, StreamingTV, StreamingMovies
- **Billing Information**: Contract Type, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
- **Churn**: Whether the customer has left (1) or stayed (0)

## **Steps Taken**
### **1. Data Preprocessing**
- Handled missing values and outliers
- Converted categorical variables using One-Hot Encoding
- Scaled numerical features using MinMaxScaler

### **2. Exploratory Data Analysis (EDA)**
- Visualized churn distribution using pie charts and bar graphs
- Analyzed correlations between features and churn rate

### **3. Feature Engineering**
- Created new features to enhance model performance
- Removed highly correlated or redundant features

### **4. Model Training & Evaluation**
- Split data into training and validation sets
- Used **Logistic Regression** as the primary model
- Evaluated performance using **accuracy, precision, recall, and AUC-ROC scores**

### **5. Prediction Function**
- Created a function to predict churn probability based on user input
- Returns a tuple with churn classification (**True/False**) and probability score

## **Installation**
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

## **Usage**
Example input for making predictions:
```python
new_input = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 34,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'DSL',
    'OnlineSecurity': 'Yes',
    'OnlineBackup': 'No',
    'DeviceProtection': 'Yes',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'One year',
    'PaperlessBilling': 'No',
    'PaymentMethod': 'Mailed check',
    'MonthlyCharges': 56.95,
    'TotalCharges': 1889.50
}
predict_input(new_input)
```

## **Model Output**
The model returns a tuple:
- **0** (False) - Customer is **not likely to churn**
- **1** (True) - Customer is **likely to churn**
- **Probability Score** - Confidence level of the prediction

## **Data Visualization**
Below is an example pie chart showing the churn distribution:
```python
import matplotlib.pyplot as plt
df['Churn'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon'])
plt.title('Customer Churn Distribution')
plt.ylabel('')
plt.show()
