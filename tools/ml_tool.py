import joblib
import pandas as pd
from mLmodel import feature_engineering


def ml_score(txn: dict):
    try:
        model = joblib.load("fraud_model.pkl")
        df = pd.DataFrame([txn])

        df = feature_engineering(df)

        # XGBoost modelinin eğitimde gördüğü ve tam olarak beklediği kolon isimlerini çekiyoruz
        expected_columns = model.get_booster().feature_names

        # Modelin beklediği ama API'den gelmeyen tüm kolonları 0.0 ile dolduruyoruz
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0.0

        # Veri setimizi tam olarak modelin beklediği sıraya ve kolonlara göre kırpıyoruz.
        # Bu sayede modelin bilmediği 'user_id', 'merchant_id', 'country' gibi API parametreleri otomatik olarak dışarıda kalıyor.
        df_for_model = df[expected_columns]

        return float(model.predict_proba(df_for_model)[0][1])

    except Exception as e:
        print(f"⚠️ ML Model Uyarısı: {e}")
        return 0.50