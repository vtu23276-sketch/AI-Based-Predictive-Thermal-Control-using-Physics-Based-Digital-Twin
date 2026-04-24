class DisturbanceEvent:
    """
    Disturbance logic for app.py.
    """
    def __init__(self, id, name, temp_change, duration, color, icon, description):
        self.id = id
        self.name = name
        self.temp_change = temp_change
        self.duration = duration
        self.color = color
        self.icon = icon
        self.description = description

    def apply_to_twin(self, twin):
        """
        Sets the disturbance value in the digital twin.
        """
        # twin.set_disturbance is a method I added to twin/digital_twin.py
        twin.set_disturbance(self.temp_change)

# List of Disturbances
DISTURBANCES = [
    DisturbanceEvent("porthole", "Open Porthole", -0.5, 20, "#fb7185", "🚪", "Port hole opened for care."),
    DisturbanceEvent("draft", "Cold Draft", -0.3, 15, "#60a5fa", "🌬️", "Sudden external cooling."),
    DisturbanceEvent("ambient", "Ambient Shift", 0.2, 30, "#fbbf24", "📉", "Room AC adjusted."),
    DisturbanceEvent("heater", "Heater Spike", 0.4, 10, "#f87171", "🔥", "External heat source nearby."),
    DisturbanceEvent("cover", "Cover Removed", -0.8, 10, "#94a3b8", "🧺", "Top cover removed temporarily.")
]

# Map for easy access
DISTURBANCE_MAP = {d.id: d for d in DISTURBANCES}
