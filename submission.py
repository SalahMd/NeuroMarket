# # submission.py
# import torch
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from src.helpers import config

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 1. تحميل النموذج المدرب
# model = EnsembleModel(num_features=7, dropout=0.4).to(device)
# model.load_state_dict(torch.load('best_ensemble_model.pt'))
# model.eval()

# # 2. تحميل بيانات Test
# test_data = pd.read_csv("/kaggle/input/predicting-stock-trends-rise-or-fall/test.csv", parse_dates=["Date"])
# test_data = test_data.sort_values(["Ticker", "Date"])

# features = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]

# # 3. Normalization (استخدم نفس الـ scaler من التدريب)
# # يجب حفظ الـ scaler أثناء التدريب:
# # import joblib
# # joblib.dump(scaler, 'scaler.pkl')

# import joblib
# scaler = joblib.load('scaler.pkl')
# test_data[features] = scaler.transform(test_data[features])

# # 4. إنشاء التنبؤات
# predictions = []
# window_size = 60

# for ticker in test_data['Ticker'].unique():
#     ticker_data = test_data[test_data['Ticker'] == ticker].reset_index(drop=True)
    
#     if len(ticker_data) < window_size:
#         continue
    
#     # خذ آخر 60 يوم
#     X = ticker_data.iloc[-window_size:][features].values
#     X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         output = model(X_tensor)
#         pred = output.argmax(1).item()
    
#     predictions.append({
#         'Ticker': ticker,
#         'Prediction': pred  # 0 or 1
#     })

# # 5. إنشاء ملف Submission
# submission_df = pd.DataFrame(predictions)
# submission_df.to_csv('submission.csv', index=False)

# print(f"✓ Submission file created with {len(submission_df)} predictions")
# print(f"  Up predictions: {(submission_df['Prediction']==1).sum()}")
# print(f"  Down predictions: {(submission_df['Prediction']==0).sum()}")