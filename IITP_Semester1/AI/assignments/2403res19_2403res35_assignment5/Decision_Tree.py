# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import LabelEncoder

start_time = time.time()

# Load the dataset
data = pd.read_csv("Fraud_check.csv")

# Data preprocessing
data['Income_Category'] = data['Taxable.Income'].apply(lambda x: 'Risky' if x <= 30000 else 'Good')

# Encoding categorical variables
label_encoder = LabelEncoder()
data['Undergrad'] = label_encoder.fit_transform(data['Undergrad'])
data['Marital.Status'] = label_encoder.fit_transform(data['Marital.Status'])
data['Urban'] = label_encoder.fit_transform(data['Urban'])

# Selecting features and target variable
X = data.drop(columns=['Income_Category', 'Taxable.Income'])  # Features
y = data['Income_Category']  # Target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model building - Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Parameter grid for grid search
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search to find the best parameters
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and best accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# Predictions on the test set using the best model
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

# Evaluate the best model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Printing accuracy, confusion matrix, and classification report
print("Accuracy on Test Set:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

# Plotting confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=best_clf.classes_, yticklabels=best_clf.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Printing Decision Tree statistics
print("Decision Tree Statistics:")
print("Number of nodes:", best_clf.tree_.node_count)
print("Depth of tree:", best_clf.tree_.max_depth)

# Printing Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(best_clf, filled=True, feature_names=X.columns.tolist(), class_names=y.unique().tolist())
plt.title("Decision Tree")
plt.show()

# Documenting results
results_document = """
Results Summary:
----------------
Best Parameters: {}
Best Accuracy: {}

Accuracy on Test Set: {}
Confusion Matrix:
{}

Classification Report:
{}

Decision Tree Statistics:
Number of nodes: {}
Depth of tree: {}

""".format(grid_search.best_params_, grid_search.best_score_, accuracy, conf_matrix, classification_rep, best_clf.tree_.node_count, best_clf.tree_.max_depth)

# Saving results to a text file
with open("results_document.txt", "w") as file:
    file.write(results_document)

end_time = time.time()

execution_time = end_time - start_time
print("Time taken for execution:", execution_time, "seconds")