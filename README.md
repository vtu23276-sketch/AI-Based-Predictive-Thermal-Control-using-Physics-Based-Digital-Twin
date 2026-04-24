# AI-Based Predictive Thermal Control using Physics-Based Digital Twin

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://share.streamlit.io/)

This project demonstrates the advantage of AI-based predictive control over traditional reactive control in the context of a neonatal incubator. By utilizing a physics-based digital twin, the system can anticipate temperature deviations caused by external disturbances and proactively adjust heating levels.

## 📂 Project Structure

```text
.
├── configs/            # Configuration settings and constants
├── data/               # Placeholder for simulation data exports
├── docs/               # Detailed documentation and architecture
├── models/             # Placeholder for saved ML models
├── scripts/            # CLI simulation runner (main.py)
├── src/                # Core source code
│   ├── analysis/       # Metrics and performance evaluation
│   ├── disturbances/   # Simulated incident logic
│   ├── twin/           # Digital twin and controller implementations
│   ├── app.py          # Streamlit dashboard
│   └── plot.py         # Static plotting utility
├── tests/              # Unit tests
├── requirements.txt    # Project dependencies
└── README.md           # Project overview
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/vtu23276-sketch/AI-Based-Predictive-Thermal-Control-using-Physics-Based-Digital-Twin.git
   cd AI-Based-Predictive-Thermal-Control-using-Physics-Based-Digital-Twin
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### 1. Interactive Dashboard (Streamlit)
To launch the interactive demo where you can manually trigger disturbances and see real-time performance:
```bash
streamlit run src/app.py
```

### 2. CLI Simulation
To run a full automated simulation with all planned disturbances and generate a performance report:
```bash
python scripts/main.py
```

## 🧪 Testing
Run unit tests to verify the simulation logic:
```bash
pytest tests/
```

## 📈 Key Metrics
The system evaluates performance using:
- **MSE (Mean Squared Error)**: Overall stability measure.
- **Max Deviation**: Peak deviation from setpoint.
- **Settling Time**: Time taken to recover after a disturbance.

## 🤝 Contributing
1. Fork the project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
