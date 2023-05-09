import pandas as pd
import numpy as np
from tqdm import tqdm


def preprocessing(tracks_file, history_file):
    """
    This method retrieves and cleans the dataframe before proceeding to the recommendation system.
    :return: pd.DataFrame
    """
    df_tracks = pd.read_json(tracks_file)
    df_tracks = df_tracks.drop_duplicates()

    df_history = pd.read_csv(history_file)
    df_history = df_history.drop_duplicates()

    reason = 'End Reason Type'

    # Manually Skipped Songs
    df_history[reason][df_history[reason] == 'MANUALLY_SELECTED_PLAYBACK_OF_A_DIFF_ITEM'] = 'END_OF_SEQUENCE'
    # Skipped backwards
    df_history[reason][df_history[reason] == 'TRACK_SKIPPED_BACKWARDS'] = 'NATURAL_END_OF_TRACK'
    # Manually Paused
    df_history[reason][df_history[reason] == 'PLAYBACK_MANUALLY_PAUSED'] = 'END_OF_SEQUENCE'
    # Scrub Begin
    df_history[reason][df_history[reason] == 'SCRUB_BEGIN'] = 'NATURAL_END_OF_TRACK'
    # Scrub End
    df_history[reason][df_history[reason] == 'SCRUB_END'] = 'NATURAL_END_OF_TRACK'

    # Skipped forwards
    mask = ~(df_history[reason] == 'TRACK_SKIPPED_FORWARDS') & (
                (df_history[reason] == 'NATURAL_END_OF_TRACK') | (df_history[reason] == 'END_OF_SEQUENCE'))

    # Flag rows
    df_history["Flag"] = df_history[reason].apply(lambda x: "-" if x=="NATURAL_END_OF_TRACK" else "[EOS]")

    # Create New Field to Identify Tracks
    df_history["Track"] = df_history["Artist Name"] + " - " + df_history["Song Name"]

    # Filtering out unnecessary values
    df_history_treated = df_history[mask]

    # Select columns
    tracks_columns = ["Track Identifier", "Title", "Album", "Artist", "Genre"]
    history_columns = ["Track", "Artist Name", "Song Name", "Flag"]

    return df_tracks[tracks_columns], df_history_treated[history_columns]