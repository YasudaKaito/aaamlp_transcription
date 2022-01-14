import pandas as pd

from chapter5 import config


def remove_cols():
    df = pd.read_csv(config.CENSUS_FILE)
    df = df.drop(config.REMOVE_COLS, axis=1)
    df.to_csv(config.CENSUS_FILE_TMP, index=False)


if __name__ == "__main__":
    remove_cols()
