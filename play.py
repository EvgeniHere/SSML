import glob
from collections import defaultdict
from typing import Tuple, List

import carball
import keras
import numpy as np
import pandas as pd
from carball.analysis.analysis_manager import AnalysisManager
from carball.json_parser.game import Game

from visualization import setup_plot, plot_frame

model = keras.models.load_model('model')


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


boost_mapping = {
    40.0: 0, 50.0: 1, 160.0: 2,
    190.0: 3, 300.0: 4, 310.0: 5
}


def compute_boost_pad_states(boost_collect, num_frames, cooldown_ticks=300):
    # cooldown_ticks: 10 seconds at 30Hz
    boost_pad_states = np.ones((num_frames, len(boost_mapping)))
    frame_to_pad_ids = defaultdict(list)
    for idx, row in boost_collect.iterrows():
        for pad_id in row:
            if not np.isnan(pad_id):
                boost_idx = boost_mapping.get(pad_id, -1)
                if boost_idx >= 0:
                    frame_to_pad_ids[idx].append(boost_idx)

    for frame_idx, pad_ids in frame_to_pad_ids.items():
        for pad_id in pad_ids:
            end_idx = min(frame_idx + cooldown_ticks, num_frames)
            boost_pad_states[frame_idx:end_idx, pad_id] = 0
    return boost_pad_states


def process_replay(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    print(f"Decompiling replay: {filepath}")
    _json = carball.decompile_replay(filepath)
    print("Replay decompiled successfully")

    game = Game()
    print("Initializing game...")
    game.initialize(loaded_json=_json)
    print("Game initialized successfully")

    analysis_manager = AnalysisManager(game)
    print("Creating analysis...")
    analysis_manager.create_analysis()
    print("Analysis created successfully")

    dataframe = analysis_manager.get_data_frame()

    print("Converting boost pickups...")

    boost_collect = dataframe.loc[:, (slice(None), 'boost_collect')]

    boost_pad_states = compute_boost_pad_states(boost_collect, dataframe.shape[0])

    normalized_data = prepare_data(dataframe)

    ball = np.nan_to_num(normalized_data['ball'], nan=0.0)

    non_ball_data = normalized_data.drop('ball', level=0, axis=1)
    blue_data = non_ball_data.copy()
    orange_data = non_ball_data.copy()

    for player in game.players:
        if player.is_orange:
            blue_data = blue_data.drop([player.name], level=0, axis=1)
        else:
            orange_data = orange_data.drop([player.name], level=0, axis=1)

    blue_data = np.nan_to_num(blue_data, nan=0.0)
    orange_data = np.nan_to_num(orange_data, nan=0.0)

    player_index = next(idx for idx, p in enumerate(game.players) if not p.is_orange)

    print(f"Player selected: {player_index + 1}. {game.players[player_index].name}")

    input_data = np.concatenate((
        blue_data[:, (player_index * 10):(player_index * 10 + 10)],
        blue_data[:, :(player_index * 10)],
        blue_data[:, (player_index * 10 + 10):],
        orange_data,
        ball,
        boost_pad_states
    ), axis=1).astype(np.float32)

    return input_data, boost_pad_states


def normalize_boostpad_positions(boost_pad_positions: List[List[float]]) -> np.ndarray:
    boost_pad_positions = np.array(boost_pad_positions)
    boost_pad_positions[:, 0] = (boost_pad_positions[:, 0] + 4096) / 8192.0
    boost_pad_positions[:, 1] = (boost_pad_positions[:, 1] + 6000) / 12000.0
    return boost_pad_positions


boost_pad_positions = normalize_boostpad_positions([
    [3072.0, -4096.0], [-3072.0, -4096.0], [-3584.0, 0.0], [3584.0, 0.0],
    [3072.0, 4096.0], [-3072.0, 4096.0], [-1792.0, 4184.0], [1792.0, 4184.0],
    [-1792.0, -4184.0], [1792.0, -4184.0], [-940.0, -3308.0], [940.0, -3308.0],
    [0.0, -2816.0], [-3584.0, -2484.0], [3584.0, -2484.0], [-1788.0, -2300.0],
    [1788.0, -2300.0], [-2048.0, -1036.0], [0.0, -1024.0], [2048.0, -1036.0],
    [0.0, -4240.0], [0.0, 4240.0], [-1024.0, 0.0], [1024.0, 0.0],
    [-2048.0, 1036.0], [0.0, 1024.0], [2048.0, 1036.0], [-1788.0, 2300.0],
    [1788.0, 2300.0], [-3584.0, 2484.0], [3584.0, 2484.0], [0.0, 2816.0],
    [-940.0, 3310.0], [940.0, 3308.0]
])

for filepath in glob.iglob('test_replays/*.replay'):
    input_data, boost_pad_states = process_replay(filepath)

    num_entities = input_data.shape[1] // 10

    fig, ax = setup_plot()
    for i in range(len(input_data)):
        plot_frame(ax, input_data, i, model, boost_pad_states, boost_pad_positions)
