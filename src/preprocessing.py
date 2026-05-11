import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(file_path):
    """Load dataset from CSV"""
    return pd.read_csv(file_path)


def clean_data(df):
    """Handle missing values and duplicates"""
    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

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