import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

def log_and_plot_predictions(true_values, predictions, test_dates,
                             true_values_rolling, predictions_rolling,
                             model_name, attack_name, epsilon,
                             res_tab, similarity_fn,google=False,
                             reverse=False,save_path=None, save_png=False):

    mae = mean_absolute_error(true_values, predictions)
    sim = similarity_fn(true_values, predictions)
    rmse = np.sqrt(mean_squared_error(true_values, predictions))

    eps_str = f"{epsilon:.2f}"

    res_tab.loc[(model_name, 'MAE'), (attack_name, eps_str)] = mae
    res_tab.loc[(model_name, 'SIM'), (attack_name, eps_str)] = sim
    res_tab.loc[(model_name, 'RMSE'), (attack_name, eps_str)] = rmse

    print(f"{model_name} | {attack_name} – Epsilon {eps_str} – MAE: {mae:.4f} | SIM: {sim:.4f}")

    if google == False:
      plt.figure(figsize=(12, 6))
      plt.plot(test_dates, true_values_rolling, label='real rolling average', color='blue')
      plt.plot(test_dates, predictions_rolling, label='predict rolling average', color='red')
      plt.xlabel('Date')
      plt.ylabel('Température (°C)')
      plt.title(f'Température en France – {model_name} ({attack_name}, eps={eps_str})')
      plt.legend()
      plt.xticks(rotation=45)
      plt.tight_layout()
      plt.show()
    else:
      if reverse == False:
        plt.figure(figsize=(10,6))
        plt.plot(test_dates, true_values, label='Real Values', color='blue')
        plt.plot(test_dates, predictions, label='Predictions', color='red', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title('Real vs Predicted Stock Prices')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_png and save_path:
          os.makedirs(save_path, exist_ok=True)
          filename = f"{model_name.lower()}_{attack_name.lower()}_eps_{epsilon:.2f}.png"
          plt.savefig(os.path.join(save_path, filename))
          plt.show()
      else:
        plt.figure(figsize=(10,6))
        plt.gca().invert_xaxis()
        plt.plot(test_dates, true_values, label='Real Values', color='blue')
        plt.plot(test_dates, predictions, label='Predictions', color='red', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title('Real vs Predicted Stock Prices (Reversed)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_png and save_path:
          os.makedirs(save_path, exist_ok=True)
          filename = f"{model_name.lower()}_{attack_name.lower()}_eps_{epsilon:.2f}.png"
          plt.savefig(os.path.join(save_path, filename))
          plt.show()

