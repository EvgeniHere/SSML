import glob
import sys
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
from carball import decompile_replay
from carball.analysis.analysis_manager import AnalysisManager
from carball.json_parser.game import Game
from keras import Sequential
from keras.layers import Dense

np.set_printoptions(threshold=sys.maxsize)

PREDICT_DIST = 60
PLAYER_FEATURE_LEN = 7
MAX_REPLAYS = 100

BOOST_MAPPING = {
    40.0: 0, 50.0: 1, 160.0: 2,
    190.0: 3, 300.0: 4, 310.0: 5
}


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    norm_specs = {
        'pos_x': (4096, 8192), 'pos_y': (6000, 12000), 'pos_z': (0, 2044),
        'vel_x': (35000, 70000), 'vel_y': (35000, 70000), 'vel_z': (35000, 70000),
        'rot_x': (-np.pi / 2, np.pi / 2), 'rot_y': (-np.pi, np.pi), 'rot_z': (-np.pi, np.pi),
        'boost': (0, 255)
    }
    for col, (offset, scale) in norm_specs.items():
        df.loc[:, (slice(None), col)] = df.loc[:, (slice(None), col)].apply(
            lambda x: (x + offset) / scale if offset != 0 else x / scale
        )
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    relevant_columns = ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "rot_x", "rot_y", "rot_z", "boost"]
    cols_to_keep = df.columns[df.columns.get_level_values(1).isin(relevant_columns)]
    df = df.loc[:, cols_to_keep].drop(columns=[("ball", "rot_x"), ("ball", "rot_y"), ("ball", "rot_z")])
    return normalize_dataframe(df)


def compute_boost_pad_states(boost_collect, num_frames: int, cooldown_ticks: int = 300) -> np.ndarray:
    pad_states = np.ones((num_frames, len(BOOST_MAPPING)))
    frame_to_pad_ids = defaultdict(list)
    for idx, row in boost_collect.iterrows():
        for pad_id in row:
            if not np.isnan(pad_id):
                pad_idx = BOOST_MAPPING.get(pad_id, -1)
                if pad_idx >= 0:
                    frame_to_pad_ids[idx].append(pad_idx)

    for frame_idx, pad_ids in frame_to_pad_ids.items():
        for pad_id in pad_ids:
            pad_states[frame_idx:min(frame_idx + cooldown_ticks, num_frames), pad_id] = 0
    return pad_states


def get_team_data(df: pd.DataFrame, game: Game) -> Tuple[np.ndarray, np.ndarray]:
    df_no_ball = df.drop('ball', level=0, axis=1)
    blue = df_no_ball.drop([p.name for p in game.players if not p.is_orange], level=0, axis=1)
    orange = df_no_ball.drop([p.name for p in game.players if p.is_orange], level=0, axis=1)
    return np.nan_to_num(blue, nan=0.0), np.nan_to_num(orange, nan=0.0)


def concat_section(current: np.ndarray, other: np.ndarray, ball: np.ndarray,
                   boost: np.ndarray, start: int, end: int, player_idx: int) -> np.ndarray:
    return np.concatenate((
        current[start:end, (player_idx * 7):(player_idx * 7 + 7)],
        current[start:end, :(player_idx * 7)],
        current[start:end, (player_idx * 7 + 7):],
        other[start:end],
        ball[start:end],
        boost[start:end]
    ), axis=1)


def shuffle_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices].astype(np.float32), y[indices].astype(np.float32)


X_data, y_data = [], []

for count, filepath in enumerate(glob.iglob('train_replays/*.replay')):
    if count >= MAX_REPLAYS:
        break

    print(f"Processing replay {count + 1}: {filepath}")
    game = Game()
    game.initialize(loaded_json=decompile_replay(filepath))
    manager = AnalysisManager(game)
    manager.create_analysis()

    df = manager.get_data_frame()
    boost_states = compute_boost_pad_states(df.loc[:, (slice(None), 'boost_collect')], len(df))
    normalized_df = prepare_data(df)

    ball = np.nan_to_num(normalized_df['ball'], nan=0.0)
    blue_team, orange_team = get_team_data(normalized_df, game)

    for goal_idx in range(max(len(game.goals), 1)):
        start = 0 if goal_idx == 0 else game.goals[goal_idx - 1].frame_number
        end = game.goals[goal_idx].frame_number if len(game.goals) > 0 else len(df)

        for player_idx in range(blue_team.shape[1] // PLAYER_FEATURE_LEN):
            section = concat_section(blue_team, orange_team, ball, boost_states, start, end, player_idx)

            one_third = PREDICT_DIST // 3
            two_third = (2 * PREDICT_DIST) // 3

            for idx in range(end - start - PREDICT_DIST):
                X_data.append(section[idx])
                y = np.concatenate([
                    section[idx + one_third, :3],
                    section[idx + two_third, :3],
                    section[idx + PREDICT_DIST, :3]
                ])
                y_data.append(y)

X_data = np.array(X_data)
y_data = np.array(y_data)
X_data, y_data = shuffle_data(X_data, y_data)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_data.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(9, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_data, y_data, epochs=30, batch_size=100, validation_split=0.1)
model.save('model')
