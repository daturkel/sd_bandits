from obp.dataset import BaseRealBanditDataset
import numpy as np
import pandas as pd
from scipy.special import expit


class DeezerDataset(BaseRealBanditDataset):
    def __init__(self, playlist_features: str, user_features: str):
        users_df = pd.read_csv(user_features)
        self.user_features = np.concatenate(
            [np.array(users_df.drop(["segment"], axis=1)), np.ones((len(users_df), 1))],
            axis=1,
        )

        playlists_df = pd.read_csv(playlist_features)
        self.playlist_features = np.array(playlists_df)

        self.user_segments = uses_df["segment"]
