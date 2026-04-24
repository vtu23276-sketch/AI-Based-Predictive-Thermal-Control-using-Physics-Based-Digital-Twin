# Architecture Overview

This project implements a digital twin of a neonatal incubator to compare traditional reactive control with AI-based predictive control.

## System Components

### 1. Digital Twin (`src/twin/digital_twin.py`)
A physics-based simulator that uses fundamental heat transfer equations:
$$T_{next} = T_{curr} + \frac{Q_{in} - Q_{out}}{C} + \text{disturbance} + \text{noise}$$

### 2. Traditional Controller (`src/twin/traditional_ctrl.py`)
A Bang-Bang (on/off) controller with a dead band. It only reacts after the temperature deviates from the setpoint.

### 3. AI Predictive Controller (`src/twin/ai_ctrl.py`)
Uses Linear Regression to predict future temperature trends based on a sliding window of historical data. It proactively adjusts the heater duty before significant deviations occur.

### 4. Disturbance System (`src/disturbances/events.py`)
Simulates real-world incidents such as:
- Open Porthole (Sudden heat loss)
- Cold Draft (Constant external cooling)
- Ambient Shift (Room temperature changes)
- Heater Spike (External heat source interference)
- Cover Removed (Large-scale insulation loss)

### 5. Metrics Engine (`src/analysis/metrics.py`)
Calculates KPIs to evaluate performance:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Max Deviation
- Stability Score
