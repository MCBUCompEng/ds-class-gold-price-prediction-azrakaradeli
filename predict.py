import numpy as np
import matplotlib.pyplot as plt
from Model import (
    load_and_preprocess_data, 
    create_polynomial_features, 
    gradient_descent, 
    calculate_r2
)

# 1. Veriyi Yükle
file_path = "Veri Kümesi/gram_gold_10yrs.csv"
X, y, df, kapanis_col = load_and_preprocess_data(file_path)

# 2. Optimizasyon Parametreleri
degrees = [1, 2, 3]
splits = [0.7, 0.8, 0.9]

best_r2 = -np.inf
best_config = {}

print("\n Model Eğitim ve Optimizasyon Başladı ")

for d in degrees:
    for s in splits:
        X_poly = create_polynomial_features(X, d)
        
        # Scaling
        mean, std = X_poly.mean(axis=0), X_poly.std(axis=0)
        X_scaled = (X_poly - mean) / std
        X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
        
        split_idx = int(len(X_scaled) * s)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        theta = gradient_descent(X_train, y_train, learning_rate=0.05, iterations=3000)
        
        y_pred = np.dot(X_test, theta)
        r2 = calculate_r2(y_test, y_pred)
        
        print(f"Derece {d} - Split %{int(s*100)} | R2: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_config = {'degree': d, 'split': s, 'theta': theta, 'mean': mean, 'std': std}

# 3. Sonuçları Yazdır
print("\n EN İYİ MODEL SONUCU")
print(f"En İyi Derece: {best_config['degree']}")
print(f"En İyi R2 Skoru: {best_r2:.4f}")

# 4. Yarın İçin Tahmin
last_row = X[-1].reshape(1, -1)
last_row_poly = create_polynomial_features(last_row, best_config['degree'])
last_row_scaled = (last_row_poly - best_config['mean']) / best_config['std']
last_row_final = np.insert(last_row_scaled, 0, 1)

predicted_price = np.dot(last_row_final, best_config['theta'])

print(f"Son Kapanış Fiyatı: {df[kapanis_col].iloc[-1]:.2f} TL")
print(f"YARIN İÇİN TAHMİN EDİLEN FİYAT: {predicted_price:.2f} TL")


