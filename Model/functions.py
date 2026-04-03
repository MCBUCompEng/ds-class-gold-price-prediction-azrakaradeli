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

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, sep=None, engine='python')
    df.columns = df.columns.str.strip()

    tarih_col = [c for c in df.columns if "Tarih" in c][0]
    acilis_col = [c for c in df.columns if "Acilis" in c or "Açılış" in c][0]
    kapanis_col = [c for c in df.columns if "Kapanis" in c or "Kapanış" in c][0]

    ay_map = {
        "Ocak": "January", "Şubat": "February", "Mart": "March",
        "Nisan": "April", "Mayıs": "May", "Mayis": "May", "Haziran": "June",
        "Temmuz": "July", "Ağustos": "August", "Eylül": "September",
        "Ekim": "October", "Kasım": "November", "Aralık": "December"
    }

    for tr, en in ay_map.items():
        df[tarih_col] = df[tarih_col].astype(str).str.replace(tr, en)

    df[tarih_col] = pd.to_datetime(df[tarih_col], errors='coerce')
    df = df.dropna(subset=[tarih_col])
    
    df[acilis_col] = df[acilis_col].apply(clean_number)
    df[kapanis_col] = df[kapanis_col].apply(clean_number)
    df = df.sort_values(tarih_col).reset_index(drop=True)

    # Feature Engineering
    df['day'] = df[tarih_col].dt.day
    df['month'] = df[tarih_col].dt.month
    df['lag1'] = df[kapanis_col].shift(1)
    df['lag2'] = df[kapanis_col].shift(2)
    df['ma7'] = df[kapanis_col].rolling(7).mean()

    df['target'] = df[kapanis_col]
    df = df.dropna().reset_index(drop=True)

    features = [acilis_col, 'day', 'month', 'lag1', 'lag2', 'ma7']
    X = df[features].values
    y = df['target'].values

    return X, y, df, kapanis_col

def create_polynomial_features(X, degree):
    X_poly = X.copy()
    for d in range(2, degree + 1):
        X_poly = np.concatenate((X_poly, X ** d), axis=1)
    return X_poly

def gradient_descent(X, y, learning_rate=0.01, iterations=5000):
    m, n = X.shape
    theta = np.zeros(n)
    for _ in range(iterations):
        prediction = np.dot(X, theta)
        error = prediction - y
        gradient = (1/m) * np.dot(X.T, error)
        theta = theta - learning_rate * gradient
    return theta

def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
