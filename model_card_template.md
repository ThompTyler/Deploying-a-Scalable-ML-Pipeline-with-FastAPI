# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a supervised binary classification model trained to predict whether an individual’s income exceeds $50K per year based on demographic and employment-related features from the U.S. Census dataset.

The model used is a Logistic Regression classifier implemented using scikit-learn. The model was trained using one-hot encoding for categorical variables and label binarization for the target variable.

## Intended Use
The intended use of this model is educational and demonstrative. It is designed to:
Demonstrate a full machine learning pipeline
Show preprocessing of categorical data
Illustrate model training and evaluation
Demonstrate performance evaluation on categorical data slices
Serve as part of an ML deployment pipeline using FastAPI

## Training Data
The model was trained using the publicly available U.S. Census Income dataset.

The dataset includes both continuous and categorical features such as:
Age
Workclass
Education
Marital status
Occupation
Race
Sex
Hours per week
Native country
The target variable is:
salary (binary: <=50K or >50K)

The dataset was split into:
80% training data
20% test data
Stratified sampling was used to preserve class distribution.

## Evaluation Data
The evaluation dataset consists of the 20% test split derived from the original dataset.

The same preprocessing steps (one-hot encoding and label binarization) were applied to the test data using the trained encoder and label binarizer from the training phase.

## Metrics
The model was evaluated using:
Precision
Recall
F1 Score (F-beta with beta=1)

Performance on the test dataset:
Precision: 0.7399
Recall: 0.5587
F1 Score: 0.6366

Performance was also evaluated on slices of the data for each categorical feature (e.g., workclass, education, race, sex). These slice-based performance metrics are stored in slice_output.txt.

## Ethical Considerations
The dataset includes sensitive attributes such as race and sex. Models trained on this data may reflect historical bias and show different performance across demographic groups. This model should not be used for real-world decision-making without additional fairness evaluation.

## Caveats and Recommendations
This model uses Logistic Regression without extensive tuning or fairness mitigation. Performance may vary across data slices, and the dataset may not reflect current conditions. The model is intended for educational purposes only. Future improvements could include hyperparameter tuning, cross-validation, and fairness analysis.