![daniele-d-andreti-CVvjUrsDIXE-unsplash](https://github.com/user-attachments/assets/bf388344-8996-4963-9e85-78ed6f5b618a)


# Titanic Survival Prediction using Machine Learning

By

**Dudekula Abid Hussain**

Email - dabaidhussain2502@gmail.com | Kaggle - https://www.kaggle.com/abiabid

This project is based on the classic [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition. The goal is to predict whether a given passenger would survive or not based on features like age, sex, class, and fare.

The dataset provides real-world challenges such as missing data, class imbalance, and feature engineering opportunities — making it ideal for building and comparing multiple classification models.


## Objective

To build a machine learning model that accurately predicts passenger survival on the Titanic using various classification algorithms and evaluate their performance using metrics like accuracy, recall, precision, and F1-score and confusion matrix.


## Tools & Libraries Used

* Python (Pandas, NumPy)
* Scikit-learn (RandomForest, LogisticRegression, SVM)
* XGBoost
* Matplotlib, Seaborn (for EDA and visualization)
* GridSearchCV (for hyperparameter tuning)
* Kaggle Jupyter Notebook


## Methodology

1. **Data Preprocessing** - Handled missing values, Converted categorical variables using one-hot encoding, performed Feature engineering, and Scaled numerical features for models like SVM.

2. **Exploratory Data Analysis (EDA)** - Analyzed survival distributions by gender, class, age group, and Plotted correlation heatmaps and feature importance.

3. **Model Building**

   * Trained and evaluated four different classifiers:

     * Random Forest
     * Logistic Regression
     * XGBoost
     * Support Vector Machine (SVM)
   * Used cross-validation and GridSearchCV to tune hyperparameters.


## Model Comparison & Evaluation

| Model               | Accuracy | Recall (Class 1) | Precision (Class 1) | F1 (Class 1) |
| ------------------- | -------- | ---------------- | ------------------- | ------------ |
| Random Forest       | 0.79     | 0.64             | 0.79                | 0.70         |
| Logistic Regression | 0.81     | 0.67             | 0.81                | 0.73         |
| XGBoost             | 0.76     | 0.65             | 0.70                | 0.68         |
| **SVM** (Final)     | **0.81** | **0.71**         | **0.78**            | **0.74**     |

### Final Model: Support Vector Machine (SVM)

* Chosen due to best balance of recall and precision for predicting survivors.
* Applied to test set with promising performance.

Below is the visual representation of confusion matrix of SVM model
<img width="505" alt="image" src="https://github.com/user-attachments/assets/1eda93f6-c063-41bd-829e-64eb35d810a2" />


## Insights

* **Class 0 (Did not survive)**: All models performed very well with high recall (up to 0.93).
* **Class 1 (Survived)**: Most models had lower recall, meaning they missed several actual survivors.
* **SVM** achieved the best F1 score for Class 1, striking a strong balance between over-predicting and missing real survivors.


## Conclusion

This project demonstrates a full ML pipeline: from cleaning data and feature engineering to training, hyperparameter tuning, and evaluating multiple models. It offers practical insights into classification challenges and how model performance can differ by class, not just overall accuracy.

---

Thanks for stopping by! If you found this helpful or have suggestions, feel free to leave feedback. Happy learning and exploring new data!




