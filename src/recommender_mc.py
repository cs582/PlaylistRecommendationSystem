import os

from models.preprocessing import preprocessing
from utils.mc import build_playlist
import numpy as np
import argparse
import pickle

tracks_file = "data/AppleMusic/Apple Music Library Tracks.json"
history_file = "data/AppleMusic/Apple Music Play Activity.csv"

models_dir = "models/markov_chain"

ss_identifier = "SingleMarkovChain"
ds_identifier = "DoubleMarkovChain"

parser = argparse.ArgumentParser(
    prog='Markov Chain-based Playlist Generator',
    description='Playlist generator based on a Markov Chain',
    epilog='This is a playlist music generator based on a Markov Chain'
)


if __name__=="__main__":
    # Retrieve the data
    df_tracks, df_history = preprocessing(tracks_file=tracks_file, history_file=history_file)

    # Retrieve top 100 tracks
    track_rank = df_history.groupby("Track").count().sort_values(by="Flag").index[-100:]

    # Pick 5 indexes at random without repeating a single one
    indexes = np.arange(len(track_rank))
    np.random.shuffle(indexes)

    # Load Models
    latest_ss = sorted([os.path.join(models_dir, x) for x in os.listdir(models_dir) if ss_identifier in x], key=os.path.getmtime)[-1]
    latest_ds = sorted([os.path.join(models_dir, x) for x in os.listdir(models_dir) if ds_identifier in x], key=os.path.getmtime)[-1]
    ss_mc, ds_mc = pickle.load(open(latest_ss, 'rb')), pickle.load(open(latest_ds, 'rb'))

    # Pick pivot songs
    pivot_songs = track_rank[indexes[:5]]

    # Create playlist
    playlist = build_playlist(pivot_songs, ss_mc, ds_mc)






