import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def remove_outliers_iqr(df, columns, factor=1.5):
    """Fungsi helper untuk menghapus outliers menggunakan metode IQR."""
    df_filtered = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        df_filtered = df_filtered[(df_filtered[col] >= lower_bound) & (df_filtered[col] <= upper_bound)]
    return df_filtered

def preprocess_data(file_path):
    """
    Fungsi utama untuk melakukan preprocessing otomatis.
    Input: Path file CSV mentah
    Output: DataFrame yang sudah siap latih (X dan y)
    """
    # 1. Load Dataset
    df = pd.read_csv(file_path)
    
    # 2. Pembersihan Dasar (Missing Values & Duplicates)
    df = df.dropna()
    df = df.drop_duplicates()
    
    # 3. Menghapus Outliers (Hanya pada kolom numerik)
    num_cols = df.select_dtypes(include=['number']).columns
    df_cleaned = remove_outliers_iqr(df, num_cols)
    
    # 4. Drop kolom yang tidak relevan (ID)
    # Gunakan 'errors=ignore' agar tidak error jika kolom sudah tidak ada
    df_cleaned = df_cleaned.drop(columns=['Transaction_ID', 'User_ID'], errors='ignore')
    
    # 5. Feature Engineering (Timestamp)
    if 'Timestamp' in df_cleaned.columns:
        df_cleaned['Timestamp'] = pd.to_datetime(df_cleaned['Timestamp'])
        df_cleaned['Transaction_Hour'] = df_cleaned['Timestamp'].dt.hour
        df_cleaned['Transaction_DayOfWeek'] = df_cleaned['Timestamp'].dt.dayofweek
        df_cleaned['Transaction_Month'] = df_cleaned['Timestamp'].dt.month
        df_cleaned = df_cleaned.drop(columns=['Timestamp'])
    
    # 6. Encoding Data Kategorikal (One-Hot Encoding)
    categorical_cols = df_cleaned.select_dtypes(include='object').columns
    df_cleaned = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)
    
    # 7. Pemisahan Fitur dan Target
    # Menghapus 'Risk_Score' untuk menghindari data leakage sesuai eksperimen Anda
    if 'Fraud_Label' in df_cleaned.columns:
        X = df_cleaned.drop(columns=['Fraud_Label', 'Risk_Score'], errors='ignore')
        y = df_cleaned['Fraud_Label']
    else:
        raise ValueError("Kolom target 'Fraud_Label' tidak ditemukan dalam dataset.")
    
    # 8. Standarisasi (Scaling)
    scaler = StandardScaler()
    numeric_features = X.select_dtypes(include=[np.number]).columns
    X[numeric_features] = scaler.fit_transform(X[numeric_features])
    
    return X, y

if __name__ == "__main__":
    # Ganti 'synthetic_fraud_dataset.csv' dengan nama file Anda
    try:
        X_ready, y_ready = preprocess_data("detection_fraud_dataset.csv")
        
        # Simpan hasil untuk dicek (Opsional)
        final_df = X_ready.copy()
        final_df['Fraud_Label'] = y_ready
        final_df.to_csv('train_fraud_automated.csv', index=False)
        
        print("Preprocessing Berhasil!")
        print(f"Shape Data Akhir: {X_ready.shape}")
        print("File 'train_fraud_automated.csv' telah dibuat.")
        
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")