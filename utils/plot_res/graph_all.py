import pandas as pd
import matplotlib.pyplot as plt

elec = True
birth = True
if elec and not birth:
    df = pd.read_excel("../elec/result_elec.xlsx", sheet_name="Sheet1")
if birth and not elec:
    df = pd.read_excel("../birth/result_birth.xlsx", sheet_name="Sheet1")
else:
    df = pd.read_excel("../google/result_google.xlsx", sheet_name="Sheet1")

model_rows = {
    "LSTM": 2,
    "RNN": 10,
    "GRU": 18
}

attack_colors = {
    "REV": "#1f77b4",
    "REV_NO_EQUAL": "#ff7f0e",
    "REV_BIM": "#2ca02c",
    "FGSM_SURRO": "#d62728",
    "FGSM": "#9467bd",
    "BOUNDARY": "#8c564b",
}

if elec:
    attacks = {
        "REV": slice(2, 10),
        "REV_BIM": slice(10, 18),
        "FGSM_SURRO": slice(18, 26),
        "FGSM": slice(26, 34),
        "BOUNDARY": slice(34, 42),
    }
else:
    attacks = {
        "REV": slice(2, 10),
        "REV_NO_EQUAL": slice(10, 18),
        "REV_BIM": slice(18, 26),
        "FGSM_SURRO": slice(26, 34),
        "FGSM": slice(34, 42),
        "BOUNDARY": slice(42, 50),
    }

epsilon = ["0.01", "0.02", "0.03", "0.04", "0.05", "0.07", "0.10", "0.20"]

for model_name, row_index in model_rows.items():
    mae_values = df.iloc[row_index, 2:]

    plt.figure(figsize=(10, 6))

    for attack, col_slice in attacks.items():
        color = attack_colors.get(attack, None)
        plt.plot(epsilon, mae_values[col_slice], label=attack, marker='o', color=color)

    plt.xlabel("ε (Epsilon)")
    plt.ylabel("MAE")
    plt.title(f"MAE vs ε for each attack on {model_name} model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"mae_all_attacks_{model_name}.png")
    plt.show()
