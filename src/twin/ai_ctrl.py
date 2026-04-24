from sklearn.linear_model import LinearRegression
import numpy as np

class AIPredictiveController:
    """
    Step 3: AI-Based Predictive Controller.
    Matches the requirements of app.py.
    """
    def __init__(self, setpoint=37.0, window=15, horizon=20):
        self.setpoint = setpoint
        self.window = window # Smaller window to be more responsive
        self.horizon = horizon # Longer horizon to see further ahead
        self.history = []
        self.model = LinearRegression()
        self.heater_status = 0.0

    def compute(self, current_temp):
        """
        Returns the heater duty (0.0 or 1.0).
        """
        # Add to history
        self.history.append(current_temp)
        if len(self.history) > self.window:
            self.history.pop(0)
        
        # Need enough history to make a prediction
        if len(self.history) < 5:
            return 1.0 if current_temp < self.setpoint else 0.0

        # 1. Prepare data (X: time steps, y: temps)
        X = np.array(range(len(self.history))).reshape(-1, 1)
        y = np.array(self.history)
        
        # 2. Fit the model
        self.model.fit(X, y)
        
        # 3. Predict future temperature
        # We predict at multiple horizons to see the trend clearly
        future_step = np.array([[len(self.history) + self.horizon]])
        predicted_temp = self.model.predict(future_step)[0]
        
        # Calculate slope (rate of change)
        slope = self.model.coef_[0]
        
        # 4. Aggressive Proactive Logic
        # If prediction is below setpoint, start heating IMMEDIATELY.
        # If slope is negative and we are already near setpoint, start heating EARLY.
        # The AI needs to be more "scared" of dropping below setpoint.
        
        if predicted_temp < self.setpoint + 0.05: # Slight buffer
            self.heater_status = 1.0
        elif current_temp < self.setpoint + 0.1 and slope < -0.001:
            # Anticipate a drop even if predicted_temp isn't below yet
            self.heater_status = 1.0
        else:
            self.heater_status = 0.0
            
        return self.heater_status
