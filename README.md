# Backcasting for Adversarial attacks on Time Series (BATS)<br>Adversarial Attacks on Time Series Forecasting without Future Ground Truth

## General Description

This project introduces an adversarial attack method for time series forecasting models called **Backcasting for Adversarial attacks on Time Series (BATS)**. Unlike traditional FGSM attacks, which require access to the true future values ("ground truth"), **BATS** works without any knowledge of the actual target outputs of the forecasting model.

### Main Goal

To demonstrate that it is possible to significantly degrade the predictions of a forecasting model (LSTM, RNN, GRU...) at inference time only, without knowing the true future values and in black box.

---

## Principle of the BATS Attack

The key idea is to train a "mirror" model, named `model_rev`, which predicts the past from the present like a proxy. That can be used to apply an FGSM-style perturbation on reversed input sequences. (work also with another attack like BIM)

### Pipeline of BATS (Backcasting for Adversarial attacks on Time Series)

1. Take a time window `x_t` (e.g., 30 values)  
2. Reverse it to obtain `x_rev`  
3. Feed it into `model_rev` (the model that predicts the past)  
4. Use this output as a proxy to apply FGSM  
5. Re-reverse the perturbed window to restore the original temporal order  
6. Send it to `model_normal` (the standard forecasting model)

### BATS_NO_EQUAL Variant
The **BATS_NO_EQUAL** variant is a realistic extension of **BATS** designed to simulate black-box scenarios where the attacker does not know the target model architecture.

Unlike the standard **BATS** setup, which uses a reverse model (`model_rev`) trained with the same architecture as the target forecasting model (`model_normal`), **BATS_NO_EQUAL** intentionally introduces an architectural mismatch.

This variant is designed to assess the influence and weight of the reverse model's architecture on the effectiveness of the **BATS** attack. In other words, to evaluate how much the success of **BATS** depends on using a reverse model that closely matches the victim model.

---

## Why This Approach Matters

| Feature                       | Description                                                                  |
| ------------------------------- | ---------------------------------------------------------------------------- |
| ✅ No need for ground truth            | The attack does not rely on knowing the real future values.                   |
| ✅ Inference-only viable | Fully applicable in black-box settings for the forecasting model.             |
| ✅ Realistic scenario                      | The reverse model can be trained by the attacker using publicly available data. |

---

## Baseline Attacks for Comparison

To evaluate the effectiveness of **BATS**, we compared it against several existing adversarial attacks (white-box and black-box) adapted for time series forecasting:

| Attack | Description | Requires Ground Truth | Black-box Applicable |
|--------|-------------|------------------------|-----------------------|
| **FGSM** | Fast Gradient Sign Method – applied with access to future true values. | ✅ Yes | ❌ No |
| **BIM** | Basic Iterative Method – iterative version of FGSM. | ✅ Yes | ❌ No |
| **FGSM Proxy** | FGSM using a surrogate model trained on same data. | ❌ No | ✅ Yes |
| **BATS** | Uses a reverse model to perturb inputs without future knowledge. | ❌ No | ✅ Yes |
| **BATS_NO_EQUAL** | BATS variant with different architectures between attacker and victim. | ❌ No | ✅ Yes |
| **Boundary Attack** | Decision-based black-box attack for classification, adapted here to regression.| ❌ No | ✅ Yes |

These baselines help quantify how much performance degradation **BATS** induces compared to traditional or proxy-based attacks.

---

## Code Structure

* `models` : folder containing PyTorch model implementations  
* `data` : folder containing the Google stock CSV file  
* `attack` : folder with the attack implementation  
* `utils` :  folder with utility functions needed to run code  
* `result` :  folder with all the result for each seed

---

## Run the code

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Samy-Annasri/ReverseForecastAttack/blob/main/ReverseForecastAttack.ipynb)

1. Click the badge/link to open the notebook on Google Colab.  
2. Go to **Runtime → Run all** to execute the entire notebook.  
3. No installation required. Works entirely in Colab.

## Reproducibility

For reproducibility of the paper’s experiments, we provide one notebook per dataset:

- `google_stock.ipynb` → Google Stock Prices
- `electricity.ipynb` → Electricity Load
- `sunspots.ipynb` → Sunspot Numbers
- `births.ipynb` → U.S. Daily Births
- `pedestrian.ipynb` → Pedestrian Counts

Each notebook:
1. Preprocesses the dataset
2. Trains forecasting models (RNN, GRU, LSTM)
3. Runs adversarial attacks (FGSM, BIM, BATS, BATS_NO_EQUAL, Boundary Attack)
4. Exports results (CSV + plots) into the my `results/` git folder

### Seeds
We use **10 seeds**: `0–123-1337-2024-314-42-7-77-888-999`.  
All results reported in the paper correspond to averages over these seeds.

---

## Sources

- **Google Stock Data** — Nasdaq. "Alphabet Inc. Class C Common Stock (GOOG) Historical Data". *Nasdaq*.  
  [https://www.nasdaq.com/market-activity/stocks/goog/historical](https://www.nasdaq.com/market-activity/stocks/goog/historical)  

- **General Time Series Datasets** — *Forecasting Data Repository*.  
  [https://forecastingdata.org/](https://forecastingdata.org/)  

- **Electricity Load Diagrams 2011–2014** — *Zenodo* record 4656140.  
  [https://zenodo.org/records/4656140](https://zenodo.org/records/4656140)  

- **Sunspot Numbers** — *Zenodo* record 4654722.  
  [https://zenodo.org/records/4654722](https://zenodo.org/records/4654722)  

- **U.S. Daily Births** — *Zenodo* record 4656049.  
  [https://zenodo.org/records/4656049](https://zenodo.org/records/4656049)  

- **Pedestrian Counts** — *Zenodo* record 4656626.  
  [https://zenodo.org/records/4656626](https://zenodo.org/records/4656626)  


