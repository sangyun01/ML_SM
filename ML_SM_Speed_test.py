import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor

# =====================================================================
# 1. 벤치마크 환경 Set
# =====================================================================
df = pd.read_csv("simulation_data.csv")
df = df[df["Vd"] == 0.05]

input_features = ["Lg", "LDD_Dose", "LDD_Energy", "SD_Dose", "SD_Energy"]
output_targets = ["Vth", "Id", "SS", "gm"]

X = df[input_features].values
Y = df[output_targets].values

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, Y)
print("Model training completed\n")

# =====================================================================
# 2. CPU 병렬 처리 성능 증명 벤치마크
# =====================================================================
print("=== CPU 병렬 처리 성능 벤치마크 ===")
# 가상의 소자 파라미터(Random parameter) 50만 개 생성
dummy_candidates = np.random.uniform(0, 1, (500000, 5))

# Single Core 속도 측정
model.n_jobs = 1
start_single = time.time()
model.predict(dummy_candidates)
end_single = time.time()
time_single = end_single - start_single
print(f"Single-core time: {time_single:.4f} sec")

# Multi Core 속도 측정
model.n_jobs = -1
start_multi = time.time()
model.predict(dummy_candidates)
end_multi = time.time()
time_multi = end_multi - start_multi
print(f"Multi-core time:  {time_multi:.4f} sec")

speedup = time_single / time_multi if time_multi > 0 else 0

# times faster 표현 적용
print(f"Result: Multi-core is {speedup:.1f} times faster than single-core.\n")
