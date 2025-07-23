import pandas as pd
from datetime import datetime

def load_us_births_tsf(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    data_started = False
    all_series = []

    for line in lines:
        line = line.strip()
        if line == "@data":
            data_started = True
            continue
        if not data_started or line.startswith("@"):
            continue

        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        series_name = parts[0]
        start_timestamp = parts[1]
        series_values = list(map(int, parts[2].split(",")))

        all_series.append({
            "series_name": series_name,
            "timestamp": start_timestamp,
            "series": series_values
        })

    return all_series

def load_us_births_tsf_flat(file_path):
    data = load_us_births_tsf(file_path)
    series = data[0]

    start_str = series["timestamp"].replace(" 00-00-00", " 00:00:00")

    start_date = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")

    return pd.DataFrame({
        "date": pd.date_range(start=start_date, periods=len(series["series"])),
        "births": series["series"]
    })
