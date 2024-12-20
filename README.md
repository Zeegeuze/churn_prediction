## Churn Prediction

<img src="data/master.jpg" alt="background" width="85%">

## Overview
This project implements a churn prediction model using **XGBoost** and **Random Forest** classifiers. It includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation. The models are trained on a dataset of customer information and used to predict whether a customer will churn (Attrited) or stay (Existing).

---

## Dataset

The dataset contains various features related to the customers' demographic details (e.g., age, gender, income, education level) and financial behavior (e.g., credit usage, transaction counts, credit limits). The dataset also includes customer attrition information, which is helpful for understanding the likelihood of churn.

---

## Key Features:
- **Demographics**: Age, Gender, Marital Status, Education Level, Income Category, Dependent Count
- **Financial Behavior**: Credit Limit, Total Revolving Balance, Avg Utilization Ratio, Total Transaction Amount, Transaction Count
- **Customer Relationship**: Months on Book, Card Category, Inactive Months, Contacts in Last 12 Months

---

## **Technologies Used**
- **Python** 3.x
- **Libraries**:
  - `Pandas`: Data manipulation and analysis
  - `Numpy`: Numerical operations
  - `Matplotlib`, `Seaborn`: Data visualization
  - `Scikit-learn`: Machine learning algorithms and evaluation metrics
  - `XGBoost`: Gradient Boosting classifier for churn prediction
  - `Imbalanced-learn`: SMOTE (Synthetic Minority Over-sampling Technique) for handling class imbalance

---
## **Steps Involved**
### **1. Data Loading and Preprocessing**
- Data is loaded and inspected to check for missing values and outliers.
- Features are selected based on their relevance to the prediction.
- Categorical variables are encoded using one-hot encoding.
- **SMOTE** is used for handling class imbalance by oversampling the minority class (Attrited customers).

### **2. Exploratory Data Analysis (EDA)**
- Descriptive statistics are generated for numerical and categorical features.
- Visualizations such as histograms, box plots, and heatmaps are created to analyze the distribution of features and correlations.

### **3. Model Training**
- **Random Forest**: Trained using the default and class-weight-balanced strategies before applying SMOTE.
- **XGBoost**: Hyperparameter tuning is performed using **GridSearchCV** to find the best model configuration.

### **4. Model Evaluation**
- The models are evaluated using key metrics: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and Classification Report.
- **Confusion Matrix** and **Classification Report** visualizations are generated for better understanding.

---
## **Directory Structure**
````
CHURN_PREDICTION/
│
├── data/
│   ├── BankChurners.csv                         # Dataset for churn prediction analysis
│   └── master.jpg                               # Image file
│
├── personal_maps/
│   ├── anastasiia/
│   │   ├── Ana_urs.ipynb                        # Anastasiia's notebook for analysis
│   │   ├── anastasiia_main.ipynb                # Main notebook for Anastasiia
│   │   ├── client_groups.ipynb                  # Notebook for grouping clients
│   │   └── Clustering.ipynb                     # Clustering analysis
│   │
│   ├── majid/
│   │   ├── BankChurners.csv                     # Dataset copy in Majid's folder
│   │   ├── credit_card_churn_prediction.ipynb   # Majid's credit card churn notebook
│   │   ├── majid_ML.ipynb                       # Machine learning notebook by Majid
│   │   └── preprocessing.ipynb                  # Preprocessing steps notebook
│   │
│   ├── petra/
│   │   ├── churn_prediction_final_urson.ipynb   # Petra's churn prediction notebook
│   │   └── petra_main.ipynb                     # Main notebook for Petra
│   │
│   ├── urson/
│   │   ├── Basic Clustering Test.ipynb          # Clustering tests by Urson
│   │   ├── churn_prediction_final_ad_v2.ipynb   # Final churn prediction (v2) by Urson
│   │   ├── churn_prediction_final_corrected.ipynb # Corrected churn prediction notebook
│   │   ├── churn_prediction_final_updated.ipynb # Updated churn prediction notebook
│   │   ├── churn_prediction_final_with_high_accuracy.ipynb # High accuracy model
│   │   ├── churn_prediction_final.ipynb         # Original churn prediction notebook
│   │   ├── churn_prediction.ipynb               # Early churn prediction notebook
│   │   ├── Clustering.ipynb                     # Clustering notebook
│   │   ├── logistic_reg.ipynb                   # Logistic regression notebook
│   │   └── urson_main.ipynb                     # Main notebook for Urson
│   │
│   ├── wouter/
│   │   ├── wouter_main.ipynb                    # Main notebook for Wouter
│   │   └── churn_prediction_readme.ipynb        # README notebook for Wouter's project
│
├── .gitignore                                   # Git ignore file
├── .python-version                              # Python version file
├── main.ipynb                                   # Main Jupyter notebook
├── README.md                                    # Project README file
└── requirements.txt                             # List of dependencies
```
--- 


## **How to Run the Code**
### **1. Install Dependencies**
To run the project, you’ll need to install the required dependencies. You can use the following command to install them using **pip**:
````bash
pip install -r requirements.txt
```
If you don't have a `requirements.txt` file yet, you can generate it by running:
````bash
pip freeze > requirements.txt
```
Here is a sample **requirements.txt**:
````bash
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
```
---
### **2. Run the Notebook**
- Open the **Jupyter notebook** file (`churn_prediction_final_updated_with_xgb.ipynb`) using Jupyter or any IDE that supports Jupyter notebooks (e.g., VSCode).
- Run the cells sequentially to load the data, preprocess it, train the models, and visualize the results.
---

## Future Work
- Extend analysis to include more features (e.g., spending trends).
- Automate customer segmentation reporting.
- Integrate results into a dashboard for business use.

---

📞 **Contact Information:**

In case of any questions or help, feel free to reach out:

- **Lead Developer:** [Majid Askary] (https://www.linkedin.com/in/majidaskary/)
- **Team Members:**
  - [Ursonc](https://www.linkedin.com/in/ursoncallens/)
  - [Wouter Verhaeghe](https://www.linkedin.com/in/wouterverhaeghe/)
  - [Petra]()
  - [Anastasiia Korostelova](https://www.linkedin.com/in/anastasiia-korostelova-136426329/)
