import json
import glob
import matplotlib.pyplot as plt
import os
 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
 
data = {}
for path in glob.glob("*.json"):
    with open(path) as f:
        r = json.load(f)
    parts = path.replace(".json", "").split("_")
    label = f"{parts[0]}\n{parts[1]}ep"
    data[label] = r

labels     = sorted(data.keys())
clip_means = [data[l]["clip_mean"] for l in labels]
clip_stds  = [data[l]["clip_std"]  for l in labels]
fids       = [data[l]["fid"]       for l in labels]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.bar(labels, clip_means, yerr=clip_stds, capsize=4)
ax1.set_title("CLIP Score")
ax1.set_ylabel("CLIP Mean")

ax2.bar(labels, fids, color="orange")
ax2.set_title("FID Score (lower is better)")
ax2.set_ylabel("FID")

plt.tight_layout()
plt.savefig("results.png", dpi=150)
plt.show()