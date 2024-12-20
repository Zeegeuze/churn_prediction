## Churn Prediction

![Project Image](master.jpg)

## Overview
This project involves the application of clustering techniques to customer data from a financial institution. The goal is to segment customers based on demographic and transactional behavior to identify meaningful customer groups. These insights can be used for targeted marketing, personalized offers, and improved customer retention strategies.
---

## Dataset

The dataset contains various features related to the customers' demographic details (e.g., age, gender, income, education level) and financial behavior (e.g., credit usage, transaction counts, credit limits). The dataset also includes customer attrition information, which is helpful for understanding the likelihood of churn.

---

## Key Features:
- **Demographics**: Age, Gender, Marital Status, Education Level, Income Category, Dependent Count
- **Financial Behavior**: Credit Limit, Total Revolving Balance, Avg Utilization Ratio, Total Transaction Amount, Transaction Count
- **Customer Relationship**: Months on Book, Card Category, Inactive Months, Contacts in Last 12 Months

---

## Objectives
1. **Preprocessing**: Clean and prepare the dataset by handling missing values, encoding categorical variables, and scaling numerical features.
2. **Clustering**: Apply various clustering algorithms (e.g., K-means, DBSCAN, Hierarchical Clustering) to segment customers.
3. **Evaluation**: Assess the performance of clustering using metrics like the **elbow method**, **silhouette score**, and **dendrogram**.
4. **Interpretation**: Provide insights into customer segments based on demographic and transactional patterns.

---

## Technologies Used
- **Python** (Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `plotly`)
- **Machine Learning**: K-Means Clustering
- **Data Visualization**: Matplotlib, Plotly, Seaborn

---

## Preprocessing Steps

1. **Handling Missing Values**: Missing values in the dataset are imputed or removed as necessary.
2. **Encoding Categorical Variables**: Categorical features such as `Gender`, `Marital_Status`, and `Card_Category` are encoded using one-hot encoding.
3. **Feature Scaling**: Numerical features (e.g., `Credit_Limit`, `Total_Revolving_Bal`) are standardized using Min-Max Scaling or Standardization to ensure that clustering is not biased by the magnitude of values.
4. **Drop Unnecessary Columns**: Non-informative columns like `CLIENTNUM` (customer ID) and prediction columns (e.g., `Naive_Bayes_Classifier_Attrition_Flag_...`) are dropped from the dataset before clustering.

---

## How to Run

To set up the project environment, follow these steps:

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python script to execute clustering and visualization steps.

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
