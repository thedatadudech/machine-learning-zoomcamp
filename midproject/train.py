# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
)

print("Reading data")
data = pd.read_excel(url)

# Rename columns for readability
data.columns = [
    "Relative_Compactness",
    "Surface_Area",
    "Wall_Area",
    "Roof_Area",
    "Overall_Height",
    "Orientation",
    "Glazing_Area",
    "Glazing_Area_Distribution",
    "Heating_Load",
    "Cooling_Load",
]

# Create a directory to save plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Descriptive analysis
print("Descriptive statistics:")
print(data.describe())

# Check for missing values
print("\nMissing values in each column:")
missing_values = data.isnull().sum()
print(missing_values)

# Automated correction: Fill missing values with the median of each column
if missing_values.any():
    print("\nFilling missing values with median of each column.")
    data.fillna(data.median(), inplace=True)

# Correlation analysis
correlation_matrix = data.corr()
print("\nCorrelation matrix:")
print(correlation_matrix)


# Generate a report on correlations
def generate_correlation_report(corr_matrix, threshold=0.7):
    print("\nHigh correlation pairs (threshold > 0.7):")
    for col in corr_matrix.columns:
        for idx in corr_matrix.index:
            if col != idx and abs(corr_matrix.loc[idx, col]) > threshold:
                print(f"{idx} and {col}: {corr_matrix.loc[idx, col]:.2f}")


generate_correlation_report(correlation_matrix)

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix Heatmap")
plt.savefig(os.path.join(output_dir, "correlation_matrix_heatmap.png"))
plt.show()

# Pairplot to visualize relationships
sns.pairplot(data)
plt.savefig(os.path.join(output_dir, "pairplot.png"))
plt.show()

# Features and target
X = data.iloc[:, :-2]  # All features
y = data["Heating_Load"]  # Target: Heating Load

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# Preprocessing for categorical data
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Define models to test
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Support Vector Regressor": SVR(),
}

# Split the dataset into train and test sets
print("splitting data set")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Evaluate each model
results = {}
for name, model in models.items():
    # Create a pipeline for each model
    model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Evaluate the model
    predictions = model_pipeline.predict(X_test)
    print(X_test.head())
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    results[name] = {"MAE": mae, "R2": r2}
    print(f"{name} - MAE: {mae:.2f}, R2: {r2:.2f}")

# Select the best model based on R2 score
best_model_name = max(results, key=lambda k: results[k]["R2"])
best_model = models[best_model_name]

print(f"Best model: {best_model_name} with R2: {results[best_model_name]['R2']:.2f}")

# Save the best model pipeline
best_model_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("model", best_model)]
)
with open("best_heating_load_model_pipeline.pkl", "wb") as f:
    pickle.dump(best_model_pipeline, f)
