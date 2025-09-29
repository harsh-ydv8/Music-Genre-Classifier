import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
import joblib

CSV_PATH = "features.csv"

try:
    features_df = pd.read_csv(CSV_PATH)
    print("Dataset loaded successfully!")

    X = features_df.drop('genre_label', axis=1)
    y = features_df['genre_label']


    print("\n--- Label Encoding Step ---")
    if np.issubdtype(y.dtype, np.integer):
        print("Labels are already numerically encoded. No action needed.")
    else:
        print("Labels are not numerical. Applying LabelEncoder.")

    print("\n--- Splitting Data into Training and Testing Sets ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("\n--- Scaling Features ---")
    scaler = StandardScaler()
    scaler.fit(X_train)


    print("StandardScaler has been fitted to the training data.")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nFeatures have been scaled")


    print("\n--- Training Logistic Regression Model ---")
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    print("Logistic Regression model trained successfully!")


    print("\n--- Training Support Vector Machine (SVM) Model ---")
    svm_model = SVC(kernel='rbf', C=1.0, random_state=42, probability=True)
    svm_model.fit(X_train_scaled, y_train)
    print("Support Vector Machine model trained successfully!")


    print("\n--- Training Random Forest Classifier Model ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    print("Random Forest Classifier model trained successfully!")


    print("\n--- Evaluating Models on the Test Set ---")
    y_pred_log_reg = log_reg.predict(X_test_scaled)
    accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
    print(f"Logistic Regression Accuracy: {accuracy_log_reg * 100:.2f}%")

    y_pred_svm = svm_model.predict(X_test_scaled)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print(f"Support Vector Machine Accuracy: {accuracy_svm * 100:.2f}%")

    accuracy_rf = rf_model.score(X_test_scaled, y_test)
    print(f"Random Forest Classifier Accuracy: {accuracy_rf * 100:.2f}%")


    print("\n--- Saving Models and Scaler to Disk ---")

    # The first argument is the Python object to save.
    # The second argument is the desired filename.
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(log_reg, 'logistic_regression_model.joblib')
    joblib.dump(svm_model, 'svm_model.joblib')
    joblib.dump(rf_model, 'random_forest_model.joblib')

    print("Scaler and models have been successfully saved ")
    print("The following files have been created:")
    print("- scaler.joblib")
    print("- logistic_regression_model.joblib")
    print("- svm_model.joblib")
    print("- random_forest_model.joblib")



except Exception as e:
    print(f"An error occurred: {e}")