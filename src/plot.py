import matplotlib.pyplot as plt

def generate_static_plot(trad_history, ai_history, target_temp, events_log, filename='simulation_comparison.png'):
    """
    Step 6: Visualization using Matplotlib.
    Generates a clean static graph showing traditional vs AI control.
    """
    plt.figure(figsize=(14, 8))
    
    # Plot temperatures
    plt.plot(trad_history, label='Traditional (Bang-Bang)', color='blue', alpha=0.7)
    plt.plot(ai_history, label='AI Predictive (Regression)', color='red', alpha=0.9)
    plt.axhline(y=target_temp, color='green', linestyle='--', label='Target Temp (37.0°C)')
    
    # Annotate Disturbances
    for t, name, desc in events_log:
        plt.annotate(name, xy=(t, target_temp), xytext=(t, target_temp + 0.5),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                     fontsize=9, rotation=45)

    plt.title('Incubator Temperature Control Comparison: Traditional vs. AI Predictive')
    plt.xlabel('Time Steps')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(filename)
    return filename
