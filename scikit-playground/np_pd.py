import numpy as np
import pandas as pd


df = pd.DataFrame({"Name": [pd.NA, "Alice", "Bob", "Eve", ""], "Age": [21, 22, 23, np.nan, np.inf]})
df.shape
print(df.head(), df.shape)
