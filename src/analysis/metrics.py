import numpy as np

def compute_metrics(history, target, tolerance=0.1):
    """
    Step 10: Calculate Metrics for app.py.
    """
    history = np.array(history)
    
    # 1. Mean Squared Error
    mse = np.mean((history - target)**2)
    
    # 2. RMSE
    rmse = np.sqrt(mse)
    
    # 3. MAE
    mae = np.mean(np.abs(history - target))
    
    # 4. Max Deviation
    max_dev = np.max(np.abs(history - target))
    
    # 5. Stability Score (Hypothetical)
    # 100 - Steps Outside Tolerance percentage
    outside_tolerance = np.sum(np.abs(history - target) > tolerance)
    stability_score = 100.0 - (outside_tolerance / len(history) * 100.0) if len(history) > 0 else 0.0
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Max Deviation (°C)": max_dev,
        "Stability Score": stability_score
    }

def improvement_pct(trad_mse, ai_mse):
    """
    Calculates the percentage improvement of AI over Traditional.
    Positive means AI is better (lower MSE).
    """
    if trad_mse == 0:
        return 0.0
    return ((trad_mse - ai_mse) / trad_mse) * 100.0
