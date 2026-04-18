import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def eda_and_visualization(df, images_dir="images"):
    """
    Performs Exploratory Data Analysis and saves corresponding plots.
    """
    print("Performing Exploratory Data Analysis...")
    os.makedirs(images_dir, exist_ok=True)
    
    # 1. Performance Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Performance_Rating', order=['Low', 'Medium', 'High'], palette='viridis')
    plt.title("Distribution of Performance Ratings")
    plt.savefig(f"{images_dir}/performance_distribution.png", bbox_inches='tight')
    plt.close()

    # 2. Training Hours vs Performance
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Performance_Rating', y='Training_Hours', order=['Low', 'Medium', 'High'], palette='Set2')
    plt.title("Training Hours by Performance Rating")
    plt.savefig(f"{images_dir}/training_vs_performance.png", bbox_inches='tight')
    plt.close()

    # 3. Correlation Heatmap (numeric features only)
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=['Employee_ID'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.savefig(f"{images_dir}/correlation_heatmap.png", bbox_inches='tight')
    plt.close()

    print(f"EDA plots saved to '{images_dir}' folder.")


def run_pipeline(data_path="data/synthetic_hr_data.csv", outputs_dir="outputs", images_dir="images"):
    """
    Runs the full machine learning pipeline.
    """
    os.makedirs(outputs_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Do EDA
    eda_and_visualization(df, images_dir)

    # 2. Preprocessing
    print("Preprocessing data...")
    # Drop irrelevant identifier
    df_model = df.drop(columns=['Employee_ID'])
    
    # Encode categorical variables: Department, Education_Level
    df_model = pd.get_dummies(df_model, columns=['Department', 'Education_Level'], drop_first=True)
    
    # Map target variable cleanly 
    target_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    df_model['Performance_Rating'] = df_model['Performance_Rating'].map(target_mapping)
    
    # Define X and y
    X = df_model.drop(columns=['Performance_Rating'])
    y = df_model['Performance_Rating']

    # Train Test Split
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale continuous features
    scaler = StandardScaler()
    continuous_cols = ['Age', 'Experience_Years', 'Training_Hours', 'Salary']
    X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

    # 3. Model Training
    print("Training Random Forest Model...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    # 4. Evaluation and Predictions
    print("Evaluating Model...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'])
    
    metrics_path = os.path.join(outputs_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("--- Employee Performance Predictor Evaluation ---\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nFeature Importances:\n")
        # Log feature importances
        feature_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
        f.write(feature_importances.to_string())

    print(f"Metrics saved to {metrics_path}.")

    # 5. Visualizing the Results (Confusion Matrix & Feature Importances)
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{images_dir}/confusion_matrix.png", bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    feature_importances[:7].plot(kind='barh', color='teal')
    plt.title("Top 7 Feature Importances")
    plt.gca().invert_yaxis()
    plt.savefig(f"{images_dir}/feature_importances.png", bbox_inches='tight')
    plt.close()
    
    print("Pipeline execution completed successfully.")


if __name__ == "__main__":
    run_pipeline()
