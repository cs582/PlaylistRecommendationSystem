import os
import pickle

import pandas as pd
import numpy as np

from datetime import datetime

class MarkovChain:
    def __init__(self, music_history, track_ids):
        self.music_history = music_history
        self.tracks = track_ids
        self.markov_chains_dir = "src/models/markov_chains/"

    def single_prior_track_mc(self):
        # Tracks
        tracks = self.tracks
        track_his = self.music_history["track_ID"].tolist()

        # Get all tracks with their followups
        single_track_followups = {
            t_id: [track_his[i + 1] for i, id in enumerate(track_his) if id == t_id and i + 1 < len(track_his)] for t_id
            in tracks
        }

        # --------------------
        # Build Markov Chain
        # --------------------

        # Initialize dictionary
        markov_chain = { t_id: None for t_id in tracks }

        # Compute Probabilities
        for track in tracks:
            followups = np.array(single_track_followups[track])
            probabilities = [ np.sum(t==followups)/len(followups) for t in followups ]
            markov_chain[track] = [ (p, t) for p, t in zip(probabilities, followups) ]

        filename = f"SingleMarkovChain_{datetime.strftime(datetime.now(), '%Y-%m-%dZ%H:%M:%S')}"
        with open(os.path.join(self.markov_chains_dir, filename)) as f:
            pickle.dump(f)
            print(f"Successfully saved {filename}!!!")

        return markov_chain

    def double_prior_track_mc(self):
        # Tracks
        track_his = self.music_history["track_ID"].tolist()
        tracks_len = len(track_his)

        # Get all tracks with their followups
        track_pair_followups = {
            (a, b): [ track_his[i] for i in range(2, tracks_len) if track_his[i-2]==a and track_his[i-1]==b ] for a, b
            in zip(track_his[:-1], track_his[1:])
        }

        # --------------------
        # Build Markov Chain
        # --------------------

        # Initialize dictionary
        markov_chain = { (a_id, b_id): None for a_id, b_id in zip(track_his[1:], track_his[:-1]) }

        # Compute Probabilities
        for pair in markov_chain.keys():
            followups = np.array(track_pair_followups[pair])
            probabilities = [np.sum(t == followups) / len(followups) for t in followups]
            markov_chain[pair] = [(p, t) for p, t in zip(probabilities, followups)]

        # Save model
        filename = f"DoubleMarkovChain_{datetime.strftime(datetime.now(), '%Y-%m-%dZ%H:%M:%S')}"
        with open(os.path.join(self.markov_chains_dir, filename)) as f:
            pickle.dump(f)
            print(f"Successfully saved {filename}!!!")

        return markov_chain

    def train(self):

        # Markov Chain with a single prior
        markov_chain_single_p = self.single_prior_track_mc()

        # Markov Chain for a double prior
        markov_chain_double_p = self.double_prior_track_mc()