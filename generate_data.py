import os
import numpy as np
import pandas as pd

np.random.seed(42)

num_samples = 10000

rpm = np.random.uniform(800, 4000, num_samples)
throttle = np.random.uniform(5, 80, num_samples)
coolant_temp = np.random.uniform(70, 105, num_samples)
maf = (rpm * 0.02) + (throttle * 0.5) + np.random.normal(0, 5, num_samples)
o2_sensor = np.random.uniform(0.1, 0.9, num_samples)

df = pd.DataFrame({
    'RPM': rpm,
    'Throttle_Pos': throttle,
    'Coolant_Temp': coolant_temp,
    'MAF': maf,
    'O2_Sensor_Voltage': o2_sensor
})

cond1 = (df['MAF'] > 80) & (df['O2_Sensor_Voltage'] < 0.25)
cond2 = (df['Coolant_Temp'] < 75)
condition = cond1 | cond2

df['Emission_Anomaly'] = np.where(condition, 1, 0)

noise = np.random.choice([0, 1], size=num_samples, p=[0.98, 0.02])
df['Emission_Anomaly'] = df['Emission_Anomaly'] ^ noise

save_path = '/Users/drkrm/Documents/IRIS/obd_emissions_data.csv'

os.makedirs(os.path.dirname(save_path), exist_ok=True)

df.to_csv(save_path, index=False)
print(f" Simulated OBD-II Dataset saved successfully at:\n {save_path}")
print("\nAnomaly counts:")
print(df['Emission_Anomaly'].value_counts())
