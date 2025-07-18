import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
def log_and_plot_elec(true_values, predictions,
                      model_name, attack_name, epsilon,
                      res_tab, similarity_fn,
                      save_path=None, save_png=False):
    # Calcul des métriques
    mae = mean_absolute_error(true_values, predictions)
    sim = similarity_fn(true_values, predictions)
    rmse = np.sqrt(mean_squared_error(true_values, predictions))

    eps_str = f"{epsilon:.2f}"

    # Log dans la table de résultats
    res_tab.loc[(model_name, 'MAE'), (attack_name, eps_str)] = mae
    res_tab.loc[(model_name, 'SIM'), (attack_name, eps_str)] = sim
    res_tab.loc[(model_name, 'RMSE'), (attack_name, eps_str)] = rmse

    # Affichage console
    print(f"{model_name} | {attack_name} – Epsilon {eps_str} – MAE: {mae:.4f} | SIM: {sim:.4f}")

    # Génération de l'axe X simple : 0, 1, 2, ..., N-1
    x = list(range(len(true_values)))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(x, true_values, label='Valeurs Réelles', color='blue')
    plt.plot(x, predictions, label='Prédictions', color='red', linestyle='--')
    plt.xlabel('Heures')
    plt.ylabel('Consommation (normalisée ou dénormalisée)')
    plt.title(f'Électricité – {model_name} ({attack_name}, eps={eps_str})')
    plt.legend()
    plt.tight_layout()

    # Sauvegarde
    if save_png and save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f"{model_name.lower()}_{attack_name.lower()}_eps_{epsilon:.2f}.png"
        plt.savefig(os.path.join(save_path, filename))

    plt.show()
