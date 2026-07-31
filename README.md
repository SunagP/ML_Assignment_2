# ML Classification Model Comparison

## a. Problem Statement

Breast cancer is one of the most prevalent cancers worldwide. Early and accurate diagnosis is critical for improving patient outcomes. The objective of this project is to build and compare **five classical machine-learning classification models** that predict whether a breast tumour is **Malignant (0)** or **Benign (1)** based on 30 real-valued diagnostic features extracted from digitised images of fine-needle aspirate (FNA) biopsies.

Each model is evaluated on the same held-out test set using six standard metrics — Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC) — to determine which algorithm generalises best on this dataset. The results are presented through an interactive **Streamlit web application** deployed on Streamlit Community Cloud.

## b. Dataset Description

| Property | Value |
|---|---|
| **Name** | Breast Cancer Wisconsin (Diagnostic) |
| **Source** | UCI Machine Learning Repository / `sklearn.datasets.load_breast_cancer` |
| **Total Instances** | 569 |
| **Training Set** | 455 (80%) |
| **Test Set** | 114 (20%, stratified) |
| **Number of Features** | 30 (all real-valued) |
| **Target Variable** | Binary — 0 = Malignant, 1 = Benign |
| **Class Distribution** | 212 Malignant (37.3%), 357 Benign (62.7%) |
| **Missing Values** | None |

### Feature Details

Ten real-valued measurements are computed for each cell nucleus in the FNA image. For each measurement, three statistics are recorded — **mean**, **standard error**, and **worst** (mean of the three largest values) — yielding 30 features in total:

| # | Base Measurement | Features Generated |
|---|---|---|
| 1 | Radius | `mean radius`, `radius error`, `worst radius` |
| 2 | Texture | `mean texture`, `texture error`, `worst texture` |
| 3 | Perimeter | `mean perimeter`, `perimeter error`, `worst perimeter` |
| 4 | Area | `mean area`, `area error`, `worst area` |
| 5 | Smoothness | `mean smoothness`, `smoothness error`, `worst smoothness` |
| 6 | Compactness | `mean compactness`, `compactness error`, `worst compactness` |
| 7 | Concavity | `mean concavity`, `concavity error`, `worst concavity` |
| 8 | Concave Points | `mean concave points`, `concave points error`, `worst concave points` |
| 9 | Symmetry | `mean symmetry`, `symmetry error`, `worst symmetry` |
| 10 | Fractal Dimension | `mean fractal dimension`, `fractal dimension error`, `worst fractal dimension` |

All features are standardised using `StandardScaler` before model training.

## c. GitHub Repository Link

> **GitHub Repository:** [https://github.com/SunagP/ML_Assignment_2](https://github.com/SunagP/ML_Assignment_2)
>
> **Live Streamlit App:** [https://sunagp-ml-assignment-2.streamlit.app](https://sunagp-ml-assignment-2.streamlit.app/)

## d. Models Used

### Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9825 | 0.9825 | 0.9825 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9161 | 0.9123 | 0.9130 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9561 | 0.9561 | 0.9560 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9298 | 0.9298 | 0.9298 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9932 | 0.9561 | 0.9561 | 0.9560 | 0.9054 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Best overall performer across every metric. It achieves the highest accuracy of **98.25%**, the highest AUC of **0.9954**, and the highest MCC of **0.9623**. Its F1 score of **0.9825** is **5.27 percentage points** higher than the next-best model (KNN/Random Forest at 0.9560). The strong performance indicates that the 30 standardised features create a nearly linearly separable decision boundary between Malignant and Benign classes, which Logistic Regression exploits optimally. Only **2 out of 114** test samples were misclassified. |
| **Decision Tree** | Weakest performer among all five models. Accuracy is **91.23%**, which is **7.02 percentage points** lower than Logistic Regression. AUC is only **0.9157** — the lowest by a large margin (Logistic Regression's AUC is 0.9954, a gap of **0.0797**). MCC of **0.8174** is also the lowest, trailing Logistic Regression by **0.1449**. The single unpruned tree overfits the training data and produces axis-aligned splits that fail to capture the smooth boundary in the feature space. It misclassified **10 out of 114** test samples. |
| **KNN** | Achieves an accuracy of **95.61%** and MCC of **0.9054**, placing it second alongside Random Forest. Its AUC of **0.9788** is solid but **0.0166 below** Logistic Regression. KNN benefits from the StandardScaler normalisation (Euclidean distance becomes meaningful), and with k=5 it captures local neighbourhood patterns well. However, the **2.64 percentage-point accuracy gap** versus Logistic Regression suggests the true decision boundary is more global/linear than local. It misclassified **5 out of 114** test samples. |
| **Naive Bayes** | Gaussian Naive Bayes records an accuracy of **92.98%** and an AUC of **0.9868**. Notably, its AUC is **higher than KNN's (0.9788)** and only **0.0086 behind** Logistic Regression, indicating excellent probabilistic ranking — the model assigns well-separated probability scores to the two classes. However, the hard 0.5 classification threshold leads to **8 misclassifications out of 114**, pulling accuracy **5.27 points below** Logistic Regression. MCC of **0.8492** is moderate, reflecting the impact of correlated features violating the independence assumption. |
| **Random Forest (Ensemble)** | Random Forest matches KNN with **95.61% accuracy**, **0.9560 F1**, and **0.9054 MCC**. Its AUC of **0.9932** is the second-highest overall — only **0.0022 below** Logistic Regression — demonstrating strong probabilistic calibration from the ensemble of 200 trees. Compared to the single Decision Tree, Random Forest improves accuracy by **4.38 percentage points** and AUC by **0.0775**, confirming that bagging and feature randomisation effectively reduce overfitting. It misclassified **5 out of 114** test samples, same as KNN. |
| **Overall Winner** | **Logistic Regression** is the clear winner for the Breast Cancer Wisconsin dataset. It ranks **#1 in all six evaluation metrics** — Accuracy (0.9825), AUC (0.9954), Precision (0.9825), Recall (0.9825), F1 (0.9825), and MCC (0.9623). Its dominance is explained by the fact that the 30 standardised diagnostic features create a feature space where the two classes are nearly linearly separable, making a regularised linear model the optimal choice. Among non-linear models, **Random Forest** is the runner-up by AUC (0.9932), while **KNN and Random Forest** tie as runners-up by accuracy (0.9561). |

## Streamlit App Features

The deployed Streamlit application includes the following interactive features:

| # | Feature | Description |
|---|---|---|
| 1 | **CSV Upload** | Upload custom test data (CSV) via the sidebar file uploader. Falls back to the included `test_data.csv` if nothing is uploaded. |
| 2 | **Model Selection Dropdown** | Choose any of the 5 trained models from a sidebar dropdown to inspect individual results. |
| 3 | **Evaluation Metrics Display** | Side-by-side comparison table of all models with Accuracy, AUC, Precision, Recall, F1, and MCC. Best values are highlighted. A grouped bar chart provides visual comparison. |
| 4 | **Confusion Matrix** | Per-model confusion matrix heatmap showing True Positives, True Negatives, False Positives, and False Negatives. |
| 5 | **Classification Report** | Detailed per-class precision, recall, F1-score, and support for the selected model. |
| 6 | **Metric Cards** | Six large metric cards displaying the selected model's scores at a glance. |
| 7 | **Data Preview** | Expandable section showing the first 20 rows of the uploaded test data. |

## Project Structure

```
ML_Project/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies for deployment
├── README.md                       # Project documentation (this file)
├── test_data.csv                   # Held-out test data (114 samples, 31 columns)
├── .gitignore                      # Git ignore rules
└── model/
    ├── train_models.py             # Model training & evaluation script
    ├── logistic_regression.pkl     # Trained Logistic Regression model
    ├── decision_tree.pkl           # Trained Decision Tree model
    ├── knn.pkl                     # Trained KNN model
    ├── naive_bayes.pkl             # Trained Gaussian Naive Bayes model
    ├── random_forest.pkl           # Trained Random Forest model
    ├── metrics.pkl                 # Evaluation metrics dictionary
    ├── scaler.pkl                  # Fitted StandardScaler
    └── feature_names.pkl           # List of 30 feature names
```

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/SunagP/ML_Assignment_2.git
cd ML_Assignment_2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (generates test_data.csv + model/*.pkl)
python model/train_models.py

# 4. Launch the Streamlit app
streamlit run app.py
```
