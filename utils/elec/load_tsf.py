def load_tsf(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    data_started = False
    series_data = []

    for line in lines:
        line = line.strip()
        if line == "@data":
            data_started = True
            continue
        if not data_started or line.startswith("@"):
            continue

        parts = line.split(",", 2)
        timestamp = parts[0]
        series_name = parts[1]
        series_values = list(map(float, parts[2].strip('"').split(",")))

        series_data.append({
            "series_name": series_name,
            "timestamp": timestamp,
            "series": series_values
        })

    return series_data

data = load_tsf("data/electricity_hourly_dataset.tsf")

df = pd.DataFrame([
    {"series_name": d["series_name"], **{f"t_{i}": v for i, v in enumerate(d["series"])}}
    for d in data
])
