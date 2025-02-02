import glob
import sys

import carball
import numpy
import numpy as np
import pandas
from carball.analysis.analysis_manager import AnalysisManager
from carball.json_parser.game import Game
from keras import Sequential
from keras.layers import Dense

numpy.set_printoptions(threshold=sys.maxsize)

boost_mapping = {
    40.0: 0,
    50.0: 1,
    160.0: 2,
    190.0: 3,
    300.0: 4,
    310.0: 5
}

predict_dist = 5


def normalize(dataframe):
    dataframe.loc[:, (slice(None), 'pos_x')] = dataframe.loc[:, (slice(None), 'pos_x')].apply(lambda x: x + 4096)
    dataframe.loc[:, (slice(None), 'pos_x')] = dataframe.loc[:, (slice(None), 'pos_x')].apply(lambda x: x / 8192.0)
    dataframe.loc[:, (slice(None), 'pos_y')] = dataframe.loc[:, (slice(None), 'pos_y')].apply(lambda x: x + 6000)
    dataframe.loc[:, (slice(None), 'pos_y')] = dataframe.loc[:, (slice(None), 'pos_y')].apply(lambda x: x / 12000.0)
    dataframe.loc[:, (slice(None), 'pos_z')] = dataframe.loc[:, (slice(None), 'pos_z')].apply(lambda x: x / 2044.0)

    dataframe.loc[:, (slice(None), 'vel_x')] = dataframe.loc[:, (slice(None), 'vel_x')].apply(lambda x: x + 35000)
    dataframe.loc[:, (slice(None), 'vel_x')] = dataframe.loc[:, (slice(None), 'vel_x')].apply(lambda x: x / 70000.0)
    dataframe.loc[:, (slice(None), 'vel_y')] = dataframe.loc[:, (slice(None), 'vel_y')].apply(lambda x: x + 35000)
    dataframe.loc[:, (slice(None), 'vel_y')] = dataframe.loc[:, (slice(None), 'vel_y')].apply(lambda x: x / 70000.0)
    dataframe.loc[:, (slice(None), 'vel_z')] = dataframe.loc[:, (slice(None), 'vel_z')].apply(lambda x: x + 35000)
    dataframe.loc[:, (slice(None), 'vel_z')] = dataframe.loc[:, (slice(None), 'vel_z')].apply(lambda x: x / 70000.0)

    dataframe.loc[:, (slice(None), 'boost')] = dataframe.loc[:, (slice(None), 'boost')].apply(lambda x: x / 255.0)

    return dataframe


def prepareData(df, boostPads):
    columns = df.columns[df.columns.get_level_values(1).isin([
        "pos_x", "pos_y", "pos_z",
        "vel_x", "vel_y", "vel_z",
        "boost"
    ])]
    df = df.loc[:, columns]

    df_ball = df['ball']

    df_players = df.drop(columns='ball', level=0)
    stationary_mask = (df_players.shift() == df_players).groupby(level=0, axis=1).all()

    rows_to_drop = stationary_mask.any(axis=1)

    df_filtered_ball = df_ball[~rows_to_drop]
    df_filtered_ball.columns = pandas.MultiIndex.from_product([['ball'], df_filtered_ball.columns])

    df_filtered_players = df_players[~rows_to_drop]

    df_filtered = pandas.concat(
        [df_filtered_players, df_filtered_ball],
        axis=1
    )

    return normalize(df_filtered), boostPads[~rows_to_drop]


def getMappedBoostId(pad_id):
    try:
        return boost_mapping.get(pad_id, -1)
    except KeyError:
        raise ValueError(f"Invalid replay pad_id: {pad_id}")


X_data = []
y_data = []
first = True
number = 0


def getBallData(df):
    return np.nan_to_num(df['ball'], nan=0.0)


def getTeamData(df):
    blue_team = df.copy()
    orange_team = df.copy()

    blue_team = blue_team.drop('ball', level=0, axis=1)
    orange_team = orange_team.drop('ball', level=0, axis=1)

    blue_team = np.nan_to_num(blue_team.drop([player.name for player in game.players if not player.is_orange], level=0, axis=1), nan=0.0)
    orange_team = np.nan_to_num(orange_team.drop([player.name for player in game.players if player.is_orange], level=0, axis=1), nan=0.0)

    return blue_team, orange_team


for filepath in glob.iglob('train_replays/*.replay'):
    print(f"Decompiling replay {number + 1}: {filepath}")
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

    boost_collect = dataframe.loc[:, (slice(None), 'boost_collect')]

    print("Converting boost pickups...")
    boostPads = np.ones((len(dataframe), 6))
    for rowNr in range(len(dataframe)):
        for padId in boost_collect.iloc[rowNr]:
            if np.isnan(padId):
                continue
            pickedBigPadId = getMappedBoostId(padId)
            for i in range(30 * 10):  # frame_tick is 0,03333 seconds -> 30 ticks/second
                if i + rowNr >= len(dataframe):
                    break
                boostPads[i + rowNr, pickedBigPadId] = 0

    orig_data, boostPads = prepareData(dataframe, boostPads)

    ball = getBallData(orig_data)
    blue_team, orange_team = getTeamData(orig_data)

    for goalNr in range(len(game.goals)):
        start = 0 if goalNr == 0 else game.goals[goalNr - 1].frame_number
        end = game.goals[goalNr].frame_number

        for player_num in range(3):
            data = np.concatenate(
                (
                    np.concatenate(
                        (
                            np.concatenate(
                                (
                                    blue_team[start:end, (player_num * 7):(player_num * 7 + 7)],
                                    np.concatenate(
                                        (
                                            blue_team[start:end, :(player_num * 7)], blue_team[start:end, (player_num * 7 + 7):]
                                        ), 
                                        axis=1
                                    )
                                ), 
                                axis=1
                            ),
                            orange_team[start:end]
                        ), 
                        axis=1
                    ), 
                    ball[start:end]
                ), 
                axis=1
            )
            data = np.concatenate((data, boostPads[start:end]), axis=1)
            if first:
                X_data = data[:-predict_dist]
                y_data = data[predict_dist:, :6]
                first = False
            else:
                X_data = np.concatenate((X_data, data[:-predict_dist]), axis=0)
                y_data = np.concatenate((y_data, data[predict_dist:, :6]), axis=0)

        for player_num in range(3):
            data = np.concatenate((np.concatenate((np.concatenate((orange_team[start:end, (player_num * 7):(player_num * 7 + 7)], np.concatenate(
                (orange_team[start:end, :(player_num * 7)], orange_team[start:end, (player_num * 7 + 7):]), axis=1)), axis=1),
                                                   blue_team[start:end]), axis=1), ball[start:end]), axis=1)
            data = np.concatenate((data, boostPads[start:end]), axis=1)

            X_data = np.concatenate((X_data, data[:-predict_dist]), axis=0)
            y_data = np.concatenate((y_data, data[predict_dist:, :6]), axis=0)

    if number == 10:
        break
    number += 1

X_data = X_data.astype(np.float32)
y_data = y_data.astype(np.float32)

model = Sequential()
model.add(Dense(100, input_shape=(54,), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(6, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_data, y_data, epochs=30, batch_size=100)

model.save('model')
