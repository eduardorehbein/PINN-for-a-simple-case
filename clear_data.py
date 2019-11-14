import pandas as pd

# Drops repeated data and shuffles the whole dataset
df = pd.read_csv('./matlab/t_i_v.csv')\
        .drop_duplicates()\
        .sample(frac=1)

# Prints cleared data to a csv file
df.to_csv(r'./cleared_t_i_v.csv', index=False)
