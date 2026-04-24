class TraditionalController:
    """
    Step 2: Traditional Bang-Bang Controller.
    Matches the requirements of app.py.
    """
    def __init__(self, setpoint=37.0, dead_band=0.3):
        self.setpoint = setpoint
        self.dead_band = dead_band
        self.heater_status = 0.0

    def compute(self, current_temp):
        """
        Returns the heater duty (0.0 or 1.0).
        """
        if current_temp < self.setpoint - self.dead_band:
            self.heater_status = 1.0
        elif current_temp > self.setpoint + self.dead_band:
            self.heater_status = 0.0
        
        return self.heater_status
