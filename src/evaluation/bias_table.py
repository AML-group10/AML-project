import json
import pandas as pd
import matplotlib.pyplot as plt

def save_bias_table_png(json_path: str, output_path: str = "bias_table.png"):
    with open(json_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data["bias_table"])

    fig, ax = plt.subplots(figsize=(7, len(df) * 0.5 + 1))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Style header row
    for col in range(len(df.columns)):
        table[0, col].set_facecolor("#333333")
        table[0, col].set_text_props(color="white", fontweight="bold")

    plt.title(f"Bias Table — {data.get('model', '')}", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


save_bias_table_png("test_results/baseline_results.json", "test_results/bias_table.png")