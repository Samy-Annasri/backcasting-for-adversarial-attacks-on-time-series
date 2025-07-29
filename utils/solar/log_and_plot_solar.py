import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def log_and_plot_solar(true_values, predictions,
                        model_name, attack_name, epsilon,
                        res_tab, similarity_fn,
                        save_path=None, save_png=False):
    
    mae = mean_absolute_error(true_values, predictions)
    sim = similarity_fn(true_values, predictions)
    rmse = np.sqrt(mean_squared_error(true_values, predictions))

    eps_str = f"{epsilon:.2f}"

    res_tab.loc[(model_name, 'MAE'), (attack_name, eps_str)] = mae
    res_tab.loc[(model_name, 'SIM'), (attack_name, eps_str)] = sim
    res_tab.loc[(model_name, 'RMSE'), (attack_name, eps_str)] = rmse

    print(f"{model_name} | {attack_name} – Epsilon {eps_str} – MAE: {mae:.4f} | SIM: {sim:.4f} | RMSE: {rmse:.4f}")

    x = list(range(len(true_values)))

    plt.figure(figsize=(12, 6))
    plt.plot(x, true_values, label='true values', color='blue')
    plt.plot(x, predictions, label='Prédictions', color='red', linestyle='--')
    plt.xlabel('days')
    plt.ylabel('sunspot number')
    plt.title(f"sunspot – {model_name} ({attack_name}, eps={eps_str})")
    plt.legend()
    plt.tight_layout()

    if save_png and save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f"{model_name.lower()}_{attack_name.lower()}_eps_{epsilon:.2f}.png"
        plt.savefig(os.path.join(save_path, filename))

    plt.show()
