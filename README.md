# Reverse Forecast Attack (RFA)<br>Adversarial Attacks on Time Series Forecasting without Future Ground Truth

## General Description

This project introduces an adversarial attack method for time series forecasting models called Reverse Forecast Attack (RFA). Unlike traditional FGSM attacks, which require access to the true future values ("ground truth"), RFA works without any knowledge of the actual target outputs of the forecasting model.

### Main Goal

To demonstrate that it is possible to significantly degrade the predictions of a forecasting model (LSTM, RNN, GRU...) at inference time only, without knowing the true future values.

---

## Principle of the RFA Attack

The key idea is to train a "mirror" model, named model_rev, which predicts the past from the present like a proxy. That can be used to apply an FGSM-style perturbation on reversed input sequences.

### Pipeline of the REV (Reverse Forecast Attack)

1. Take a time window x_t (e.g., 30 values)
2. Reverse it to obtain x_rev
3. Feed it into model_rev (the model that predicts the past)
4. Use this output as a proxy to apply FGSM
5. Re-reverse the perturbed window to restore the original temporal order
6. Send it to model_normal (the standard forecasting model)

### REV_AUTO Variant

An **autoregressive** version of the attack, where the perturbation is reapplied over successive predictions, simulating an attack over multiple forecasting steps (J+1, J+2, etc.).

---

## Why This Approach Matters

| Feature                       | Description                                                                  |
| ------------------------------- | ---------------------------------------------------------------------------- |
| ✅ No need for ground truth            | The attack does not rely on knowing the real future values.                   |
| ✅ Inference-only viable | Fully applicable in black-box settings for the forecasting model.             |
| ✅ Realistic scenario                      | The reverse model can be trained by the attacker using publicly available data. |

---

## Code Structure

* `models` : folder containing PyTorch model implementations
* `data` : folder containing the Google stock CSV file
* `attack` : folder with the attack implementation
* `utils` :  folder with utility functions needed to run code

---

## Run the code

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Samy-Annasri/ReverseForecastAttack/blob/main/ReverseForecasAttack.ipynb)


1. Click the badge/link to open the notebook on Google Colab.
2. Go to **Runtime → Run all** to execute the entire notebook.
3. No installation required. Works entirely in Colab.

---

## Limitations & Future Work

* Currently, the attack is limited to step J+1 because model_rev only predicts the immediate past.
* To go beyond that :

  * Use REV_AUTO to propagate the attack autoregressively (in development).
  * other...
---
