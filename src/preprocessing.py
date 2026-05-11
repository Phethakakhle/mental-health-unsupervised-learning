import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(file_path):
    """Load dataset from CSV"""
    return pd.read_csv(file_path)


def clean_data(df):
    df = df.copy()

    # split columns properly
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    # fill numeric columns safely
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # fill categorical columns safely
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_categorical(df):
    """Convert categorical columns into numeric form"""
    df = df.copy()
    encoders = {}

    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    return df, encoders


def scale_features(df):
    """Standardize features for ML models"""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    return scaled, scaler