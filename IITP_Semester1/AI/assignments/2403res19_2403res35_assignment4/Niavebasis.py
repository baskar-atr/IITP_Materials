import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix

# Custom train-test split function
def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    test_samples = int(test_size * len(X))
    val_samples = int(test_size * len(X))
    test_indices = indices[:test_samples]
    val_indices = indices[test_samples:test_samples+val_samples]
    train_indices = indices[test_samples+val_samples:]
    X_train, X_val, X_test = X.iloc[train_indices], X.iloc[val_indices], X.iloc[test_indices]
    y_train, y_val, y_test = y.iloc[train_indices], y.iloc[val_indices], y.iloc[test_indices]
    return X_train, X_val, X_test, y_train, y_val, y_test

# Custom 4-fold cross-validation function
def custom_cross_val_score(estimator, X, y, cv=4):
    n_samples = len(X)
    fold_size = n_samples // cv
    scores = []
    for i in range(cv):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < cv - 1 else n_samples
        X_train = pd.concat([X.iloc[:start_idx], X.iloc[end_idx:]])
        y_train = pd.concat([y.iloc[:start_idx], y.iloc[end_idx:]])
        X_val = X.iloc[start_idx:end_idx]
        y_val = y.iloc[start_idx:end_idx]
        estimator.fit(X_train, y_train)
        score = estimator.score(X_val, y_val)
        scores.append(score)
    return np.array(scores)

# Load the dataset
data = pd.read_csv("adult.csv")

# Drop null-valued rows
data.dropna(inplace=True)

# Split dataset into features and target variable
X = data.drop('income', axis=1)
y = data['income']

# Encode categorical variables
X = pd.get_dummies(X)

# Split dataset into train, validation, and test sets (60-20-20 split)
X_train, X_val, X_test, y_train, y_val, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Gaussian Naive Bayes and Multinomial Naive Bayes classifiers
gnb = GaussianNB()
mnb = MultinomialNB()

# Train Gaussian Naive Bayes classifier
gnb.fit(X_train, y_train)

# Train Multinomial Naive Bayes classifier
mnb.fit(X_train, y_train)

# Evaluate Gaussian Naive Bayes classifier
gnb_acc = gnb.score(X_test, y_test)
print("Gaussian Naive Bayes Accuracy:", gnb_acc)

# Evaluate Multinomial Naive Bayes classifier
mnb_acc = mnb.score(X_test, y_test)
print("Multinomial Naive Bayes Accuracy:", mnb_acc)

# Perform 4-fold cross-validation for Gaussian Naive Bayes classifier
gnb_cv_scores = custom_cross_val_score(gnb, X, y, cv=4)
print("Gaussian Naive Bayes Cross-Validation Scores:", gnb_cv_scores)

# Perform 4-fold cross-validation for Multinomial Naive Bayes classifier
mnb_cv_scores = custom_cross_val_score(mnb, X, y, cv=4)
print("Multinomial Naive Bayes Cross-Validation Scores:", mnb_cv_scores)

# Compare the accuracies
if gnb_acc > mnb_acc:
    print("Gaussian Naive Bayes has higher accuracy.")
elif gnb_acc < mnb_acc:
    print("Multinomial Naive Bayes has higher accuracy.")
else:
    print("Both classifiers have the same accuracy.")

# Get unique labels from true labels and predictions
unique_labels = np.unique(np.concatenate((y_test, gnb.predict(X_test), mnb.predict(X_test))))

# Confusion Matrix for Gaussian Naive Bayes
gnb_conf_matrix = confusion_matrix(y_test, gnb.predict(X_test), labels=unique_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(gnb_conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix - Gaussian Naive Bayes")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks(ticks=np.arange(len(unique_labels)), labels=unique_labels)
plt.yticks(ticks=np.arange(len(unique_labels)), labels=unique_labels)
plt.show()

# Confusion Matrix for Multinomial Naive Bayes
mnb_conf_matrix = confusion_matrix(y_test, mnb.predict(X_test), labels=unique_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(mnb_conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix - Multinomial Naive Bayes")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks(ticks=np.arange(len(unique_labels)), labels=unique_labels)
plt.yticks(ticks=np.arange(len(unique_labels)), labels=unique_labels)
plt.show()
