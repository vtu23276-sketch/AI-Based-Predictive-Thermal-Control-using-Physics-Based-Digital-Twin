import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from configs import settings

class DigitalTwin:
    """
    Physics-based digital twin of a neonatal incubator.
    Matches the requirements of app.py.
    """
    def __init__(self, T_initial=settings.T_INITIAL, T_ambient=settings.T_AMBIENT, T_setpoint=settings.SETPOINT):
        self.T = T_initial
        self.T_ambient = T_ambient
        self.T_setpoint = T_setpoint
        
        # Physical constants
        self.heat_capacity = settings.HEAT_CAPACITY
        self.heat_loss_coeff = settings.HEAT_LOSS_COEFF
        self.heater_power_max = settings.HEATER_POWER_MAX
        self.noise_std = settings.NOISE_STD
        self.dt = 1.0
        
        self.external_disturbance = 0.0

    def step(self, duty):
        """
        Advances the simulation by one step.
        duty: float between 0.0 and 1.0 (heater output)
        """
        # 1. Heat Input
        heat_input = duty * self.heater_power_max * self.dt
        
        # 2. Heat Loss
        heat_loss = self.heat_loss_coeff * (self.T - self.T_ambient) * self.dt
        
        # 3. Physics update
        dT = (heat_input - heat_loss) / self.heat_capacity
        
        # 4. Noise
        noise = np.random.normal(0, self.noise_std)
        
        # 5. Final Update
        self.T += dT + self.external_disturbance + noise
        
        return self.T

    def clear_disturbance(self):
        self.external_disturbance = 0.0

    def set_disturbance(self, value):
        self.external_disturbance = value
