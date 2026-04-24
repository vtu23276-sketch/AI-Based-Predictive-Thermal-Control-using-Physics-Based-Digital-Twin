import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from twin.digital_twin import DigitalTwin as IncubatorSimulator
from twin.traditional_ctrl import TraditionalController
from twin.ai_ctrl import AIPredictiveController
from disturbances.events import DISTURBANCE_MAP
from analysis.metrics import compute_metrics as calculate_metrics
from plot import generate_static_plot
from configs import settings

def run_simulation(duration=500, target_temp=settings.SETPOINT):
    """
    Main entry point for the simulation runner.
    """
    # Initialize Simulators
    sim_trad = IncubatorSimulator(T_initial=settings.T_INITIAL, T_ambient=settings.T_AMBIENT, T_setpoint=target_temp)
    sim_ai = IncubatorSimulator(T_initial=settings.T_INITIAL, T_ambient=settings.T_AMBIENT, T_setpoint=target_temp)
    
    # Initialize Controllers
    ctrl_trad = TraditionalController(setpoint=target_temp, dead_band=settings.TRAD_DEAD_BAND)
    ctrl_ai = AIPredictiveController(setpoint=target_temp, window=settings.AI_WINDOW, horizon=settings.AI_HORIZON)
    
    # Planned Disturbances
    planned_events = {
        50: "porthole",
        150: "draft",
        250: "ambient",
        350: "heater",
        450: "cover"
    }
    
    trad_temps = []
    ai_temps = []
    events_log = []

    print(f"Starting Simulation for {duration} time steps...")
    print(f"Target Temperature: {target_temp}°C\n")

    # Tracking active disturbances
    active_dist_val = 0.0
    dist_remaining = 0

    for t in range(duration):
        if t in planned_events:
            dist_id = planned_events[t]
            dist_obj = DISTURBANCE_MAP[dist_id]
            active_dist_val = dist_obj.temp_change
            dist_remaining = dist_obj.duration
            events_log.append((t, dist_obj.name, dist_obj.description))
            print(f"Time {t:03d}: Disturbance Triggered -> {dist_obj.name}")
        
        # Apply disturbance to twins
        if dist_remaining > 0:
            sim_trad.set_disturbance(active_dist_val)
            sim_ai.set_disturbance(active_dist_val)
            dist_remaining -= 1
        else:
            sim_trad.clear_disturbance()
            sim_ai.clear_disturbance()

        # Traditional System
        action_trad = ctrl_trad.compute(sim_trad.T)
        sim_trad.step(action_trad)
        trad_temps.append(sim_trad.T)
        
        # AI Predictive System
        action_ai = ctrl_ai.compute(sim_ai.T)
        sim_ai.step(action_ai)
        ai_temps.append(sim_ai.T)

    # Calculate Enhanced Metrics
    metrics_trad = calculate_metrics(trad_temps, target_temp)
    metrics_ai = calculate_metrics(ai_temps, target_temp)

    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"{'Metric':<25} | {'Traditional':<15} | {'AI Predictive':<15}")
    print("-" * 60)
    for key in metrics_trad.keys():
        print(f"{key:<25} | {metrics_trad[key]:<15.4f} | {metrics_ai[key]:<15.4f}")

    # Generate Static Graph
    filename = generate_static_plot(trad_temps, ai_temps, target_temp, events_log)
    print(f"\nStatic graph saved as '{filename}'")
    
    return trad_temps, ai_temps, metrics_trad, metrics_ai

if __name__ == "__main__":
    run_simulation()
