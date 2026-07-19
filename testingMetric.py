import numpy as np
import matplotlib.pyplot as plt

# Number of topological nodes
x = np.linspace(0, 1000, 300)

# Dummy trends (illustrative only)
mlatc = 0.0004 * x + 0.20           # Nearly flat linear
atc_dt = 0.0050 * x + 0.10          # Linear with noticeable slope
proposed = 0.0006 * x + 0.25        # Nearly flat linear

plt.figure(figsize=(8, 5))

plt.plot(x, mlatc, linewidth=2.5, label="MLATC")
plt.plot(x, atc_dt, linewidth=2.5, label="ATC-DT Free Space")
plt.plot(x, proposed, "--", linewidth=2.5, label="Proposed Method")

plt.xlabel("Jumlah node topologi")
plt.ylabel("Waktu pemrosesan T per frame i (ms)")

plt.xlim(0, 1000)
plt.ylim(0, 6)

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()