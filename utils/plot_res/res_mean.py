import os
import pandas as pd
from glob import glob

#change the root_dir for witch dataset i want
root_dir = r"C:\Users\samys\Downloads\ReverseForecastAttack-main\ReverseForecastAttack-main\results\birth"

xlsx_files = glob(os.path.join(root_dir, "SEED_*", "*.xlsx"))

dataframes = []

for file in xlsx_files:
    df = pd.read_excel(file, header=[0, 1], index_col=[0, 1])
    dataframes.append(df)

if not dataframes:
    raise ValueError("Aucun fichier XLSX trouvé.")

df_mean = sum(dataframes) / len(dataframes)

output_path = os.path.join(root_dir, "result_birth.xlsx")
df_mean.to_excel(output_path)

print(f"✅ folder mean save on : {output_path}")
