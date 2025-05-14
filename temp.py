import matplotlib.pyplot as plt
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(8, 10))

# Depth bins and labels
depths = [0, 200, 500, 1000]
zones = ["Epipelagic", "Upper Mesopelagic", "Lower Mesopelagic"]
depth_labels = ["0 m", "200 m", "500 m", "1000 m"]

# Draw depth bins
for depth in depths:
    ax.axhline(depth, color="black", linewidth=0.8)

# Add vertical axis with labels
for i, depth in enumerate(depth_labels):
    ax.text(-0.03, depths[i], depth, va="center", ha="right", transform=ax.get_yaxis_transform(), fontsize=11)

# Zone labels
zone_positions = [(depths[i] + depths[i+1]) / 2 for i in range(len(depths)-1)]
for i, zone in enumerate(zones):
    ax.text(-0.03, zone_positions[i], zone, va="center", ha="right", transform=ax.get_yaxis_transform(), fontsize=12, fontweight="bold")

# X range (dummy values for plotting)
x = np.linspace(0, 1, 100)
flux1 = 0.95 * np.exp(-5 * x) + 0.05  # Less efficient
flux2 = 0.8 * np.exp(-10 * x) + 0.02  # More efficient (dashed)

# Plot flux curves
ax.plot(flux1, np.linspace(0, 1000, 100), label='Lower export efficiency', color='black')
ax.plot(flux2, np.linspace(0, 1000, 100), label='Higher export efficiency', linestyle='--', color='red')

# Arrows and text annotations
ax.annotate("Efficiency of POC export is driven by\nenvironmental forcing (e.g., phytoplankton\ncommunity structure, stratification)", 
            xy=(0.4, 100), xytext=(0.55, 100),
            arrowprops=dict(arrowstyle="->", color="black"), fontsize=11)

ax.annotate("DIC$_{soft}$ accumulation\nfrom remineralized POC", 
            xy=(0.35, 300), xytext=(0.5, 300),
            arrowprops=dict(arrowstyle="->", color="black"), fontsize=11)

ax.annotate("Transport of DIC$_{soft}$ by\nocean circulation", 
            xy=(0.35, 650), xytext=(0.5, 650),
            arrowprops=dict(arrowstyle="->", color="black"), fontsize=11)

# Final plot adjustments
ax.set_ylim(1000, 0)
ax.set_xlim(0, 1)
ax.set_xlabel("Normalized POC Flux", fontsize=12)
ax.set_ylabel("Depth", fontsize=12)
ax.legend(loc="lower left", fontsize=10)
ax.set_title("Vertical Profile of POC Export and DIC$_{soft}$ Dynamics", fontsize=14, weight="bold")
ax.spines[['right', 'top']].set_visible(False)

plt.tight_layout()
plt.show()
