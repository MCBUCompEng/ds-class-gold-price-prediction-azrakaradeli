import pandas as pd
import numpy as np

def clean_number(x):
    x = str(x).strip()
    if x.count('.') > 1:
        parts = x.split('.')
        x = ''.join(parts[:-1]) + '.' + parts[-1]
    if ',' in x:
        x = x.replace('.', '').replace(',', '.')
    try:
        return float(x)
    except:
        return 0.0

def create_polynomial_features(X, degree):
    if degree <= 1:
        return X
    X_poly = X.copy()
    for d in range(2, degree + 1):
        X_poly = np.concatenate((X_poly, X ** d), axis=1)
    return X_poly

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, sep=None, engine='python')
    df.columns = df.columns.str.strip()

    tarih_col = [c for c in df.columns if "Tarih" in c][0]
    acilis_col = [c for c in df.columns if "Acilis" in c or "Açılış" in c][0]
    kapanis_col = [c for c in df.columns if "Kapanis" in c or "Kapanış" in c][0]

    ay_map = {
        "Ocak": "January", "Şubat": "February", "Mart": "March", "Nisan": "April", 
        "Mayıs": "May", "Haziran": "June", "Temmuz": "July", "Ağustos": "August", 
        "Eylül": "September", "Ekim": "October", "Kasım": "November", "Aralık": "December"
    }
    for tr, en in ay_map.items():
        df[tarih_col] = df[tarih_col].astype(str).str.replace(tr, en)

    df[tarih_col] = pd.to_datetime(df[tarih_col], errors='coerce')
    df = df.dropna(subset=[tarih_col]).sort_values(tarih_col).reset_index(drop=True)
    
    df[acilis_col] = df[acilis_col].apply(clean_number)
    df[kapanis_col] = df[kapanis_col].apply(clean_number)

    # Feature Engineering
    df['day'] = df[tarih_col].dt.day
    df['month'] = df[tarih_col].dt.month
    df['lag1'] = df[kapanis_col].shift(1)
    df['ma7'] = df[kapanis_col].rolling(7).mean()

    df = df.dropna().reset_index(drop=True)
    features = [acilis_col, 'day', 'month', 'lag1', 'ma7']
    
    return df[features].values, df[kapanis_col].values, df, kapanis_col

def train_model(X_train, y_train, learning_rate=0.05, iterations=3000):
   
    X_b = np.c_[np.ones(X_train.shape[0]), X_train]
    m, n = X_b.shape
    theta = np.zeros(n)
    
    for _ in range(iterations):
        prediction = np.dot(X_b, theta)
        error = prediction - y_train
        gradient = (1/m) * np.dot(X_b.T, error)
        theta = theta - learning_rate * gradient
    return theta

def evaluate_model(theta, X_test, y_test):
    X_b = np.c_[np.ones(X_test.shape[0]), X_test]
    y_pred = np.dot(X_b, theta)
    
    # MSE
    mse = np.mean((y_test - y_pred) ** 2)
    # R2
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return mse, r2, y_pred
