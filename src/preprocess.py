import pandas as pd
import numpy as np

# Generate synthetic dataset
np.random.seed(42)
n = 200
df = pd.DataFrame({
    'study_hours': np.random.uniform(1, 10, n),
    'attendance': np.random.uniform(50, 100, n),
    'prev_score': np.random.uniform(40, 100, n),
})
df['pass'] = ((df['study_hours'] > 5) & 
              (df['attendance'] > 75) & 
              (df['prev_score'] > 60)).astype(int)

df.to_csv('data/student_data.csv', index=False)
print("Dataset saved!")
print(df.head())
print(df['pass'].value_counts())