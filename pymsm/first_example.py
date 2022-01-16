import numpy as np
import pandas as pd

DATA_BASE_COLS = ["id", "from", "to", "trans", "t_start", "t_stop", "time"]

def get_example_df():

    example_df = pd.DataFrame(
        {
            "id": np.arange(4),
            "from": [1, 1, 1, 1],
            "to": [2, 2, 2, 3],
            "trans": [1, 1, 1, 2],
            "t_start": [0, 0, 0, 0],
            "t_stop": [1, 1, 1, 1],
            "time": [1, 1, 1, 1],
        }
    )
    return example_df

if __name__=="__main__":
    example_df = get_example_df()
    print(example_df)
