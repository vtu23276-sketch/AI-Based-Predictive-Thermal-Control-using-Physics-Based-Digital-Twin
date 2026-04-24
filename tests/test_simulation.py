import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from twin.digital_twin import DigitalTwin

def test_simulation_step():
    twin = DigitalTwin(T_initial=37.0, T_ambient=25.0)
    # With 0 duty and no disturbance, temp should decrease due to heat loss
    initial_temp = twin.T
    twin.step(duty=0.0)
    assert twin.T < initial_temp

def test_heating_effect():
    twin = DigitalTwin(T_initial=37.0, T_ambient=25.0)
    # With 1.0 duty, temp should increase
    initial_temp = twin.T
    twin.step(duty=1.0)
    assert twin.T > initial_temp

def test_disturbance_impact():
    twin = DigitalTwin(T_initial=37.0, T_ambient=25.0)
    twin.set_disturbance(-1.0)
    initial_temp = twin.T
    twin.step(duty=0.5) # Some heating
    # Large negative disturbance should override heating
    assert twin.T < initial_temp
