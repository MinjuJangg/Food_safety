import os
import pandas as pd
from typing import Optional, List
from sklearn.model_selection import train_test_split

def get_train_df(
        path: str,
        val_ratio: float = 0.2,
        random_state: Optional[int] = 42,
        shuffle: bool = True
):
    df = pd.read_parquet(path)

    train_df, val_test_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=random_state,
        shuffle=shuffle
    )
    
    val_df, test_df = train_test_split(
        val_test_df,
        test_size = 0.5,
        random_state=random_state,
        shuffle=shuffle
    )

    return train_df, val_df, test_df

if __name__ == '__main__':
    train_path = '/SSL_NAS/peoples/suhyeon/food/food_data/train.csv'
    val_path = '/SSL_NAS/peoples/suhyeon/food/food_data/val.csv'
    test_path = '/SSL_NAS/peoples/suhyeon/food/food_data/test.csv'