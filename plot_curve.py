import re
import matplotlib.pyplot as plt

LOG_PATH = "train_log.txt"
OUT_PATH = "train_curve_from_log.png"

pattern = re.compile(r"update\s+(\d+),\s+avg return per episode:\s+([0-9.]+)")
updates = []
returns = []

with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        m = pattern.search(line)
        if not m:
            continue
        updates.append(int(m.group(1)))
        returns.append(float(m.group(2)))

if not updates:
    raise SystemExit("No training lines found in train_log.txt")

plt.figure()
plt.plot(updates, returns)
plt.xlabel("update")
plt.ylabel("avg return per episode")
plt.title("REINFORCE on CartPole")
plt.savefig(OUT_PATH, dpi=150)
print(f"saved: {OUT_PATH}")
