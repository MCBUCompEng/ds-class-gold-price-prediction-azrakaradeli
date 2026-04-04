import numpy as np
from Model.functions import (
    load_and_preprocess_data, 
    create_polynomial_features, 
    train_model, 
    evaluate_model
)

file_path = "DataSet/gram_gold_10yrs.csv"
X_raw, y_raw, df, kapanis_col = load_and_preprocess_data(file_path)

# Optimizasyon Parametreleri
degrees = [1, 2, 3]
splits = [0.7, 0.8, 0.9]

best_r2 = -np.inf
best_config = {}

print("\n Model Eğitim ve Optimizasyon Başladı ")

for d in degrees:
    for s in splits:
        
        X_poly = create_polynomial_features(X_raw, d)
        
        # Kronolojik Split (Data Leakage önlemek için)
        split_idx = int(len(X_poly) * s)
        X_train, X_test = X_poly[:split_idx], X_poly[split_idx:]
        y_train, y_test = y_raw[:split_idx], y_raw[split_idx:]
        
        # Feature Scaling 
        mean, std = X_train.mean(axis=0), X_train.std(axis=0)
        
        std = np.where(std == 0, 1e-9, std) 
        
        X_train_scaled = (X_train - mean) / std
        X_test_scaled = (X_test - mean) / std
        
        # Model Eğitme
        theta = train_model(X_train_scaled, y_train)
        
        # Değerlendirme
        mse, r2, _ = evaluate_model(theta, X_test_scaled, y_test)
        
        print(f"Derece {d} - Split %{int(s*100)} | R2: {r2:.4f}")
        
        # En iyi model
        if r2 > best_r2:
            best_r2 = r2
            best_config = {
                'degree': d, 
                'split': s, 
                'theta': theta, 
                'mean': mean, 
                'std': std, 
                'mse': mse
            }

# En İyi Sonuçlar
print("\n EN İYİ MODEL SONUCU ")
print(f"En İyi Derece: {best_config['degree']}")
print(f"En İyi Split Oranı: %{int(best_config['split']*100)}")
print(f"En İyi R2 Skoru: {best_r2:.4f}")
print(f"Hata Payı (MSE): {best_config['mse']:.4f}")

# Yarın İçin Tahmin

last_row = X_raw[-1].reshape(1, -1)
last_row_poly = create_polynomial_features(last_row, best_config['degree'])
last_row_scaled = (last_row_poly - best_config['mean']) / best_config['std']

last_row_final = np.insert(last_row_scaled, 0, 1)
predicted_price = np.dot(last_row_final, best_config['theta'])

print(f"Son Kapanış Fiyatı: {df[kapanis_col].iloc[-1]:.2f} TL")
print(f"YARIN İÇİN TAHMİN EDİLEN FİYAT: {predicted_price:.2f} TL")
