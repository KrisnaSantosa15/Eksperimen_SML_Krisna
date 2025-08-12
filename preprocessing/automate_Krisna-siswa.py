import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import LabelEncoder


def drop_id_column(df):
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    return df


def drop_duplicates(df):
    return df.drop_duplicates()


def fill_missing_values(df):
    if 'bmi' in df.columns:
        df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    return df


def remove_outliers_iqr(df, columns):
    df_out = df.copy()
    for col in columns:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_out = df_out[(df_out[col] >= lower_bound) &
                        (df_out[col] <= upper_bound)]
    return df_out


def label_encode_categoricals(df):
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    return df, label_encoders


def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Tahapan preprocessing
    df = drop_id_column(df)
    df = drop_duplicates(df)
    df = fill_missing_values(df)

    # Hapus outlier
    numerical_cols = ['age', 'avg_glucose_level', 'bmi']
    df = remove_outliers_iqr(df, numerical_cols)

    # Encoding kategorikal
    df, mappings = label_encode_categoricals(df)

    print("Mapping Label Encoding:")
    for col, mapping in mappings.items():
        print(f"{col}: {mapping}")

    return df


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python automate_Krisna-siswa.py <input_csv_path> <output_csv_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    df_cleaned = preprocess_data(input_path)

    # Buat folder output jika belum ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Simpan dataframe hasil preprocessing lengkap ke CSV
    df_cleaned.to_csv(output_path, index=False)
    print(f"Preprocessing selesai, file disimpan di {output_path}")
