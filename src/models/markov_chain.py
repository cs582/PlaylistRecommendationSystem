import os
import pickle
import logging
import pandas as pd
import numpy as np

from tqdm import tqdm
from datetime import datetime


class MarkovChain:
    def __init__(self, music_history, track_ids, markov_chains_dir):
        self.music_history = music_history
        self.markov_chains_dir = markov_chains_dir

        if not os.path.exists(self.markov_chains_dir):
            os.mkdir(self.markov_chains_dir)

    def single_prior_track_mc(self):
        # Tracks
        tracks = self.music_history["Track"].drop_duplicates().tolist()
        track_his = self.music_history["Track"].tolist()
        flags_his = self.music_history["Flag"].tolist()

        # Check if is a valid followup track
        check_valid_followup = lambda id, t_id, i: (id == t_id and i + 1 < len(track_his) and flags_his[i] != "EOS")

        # Get all tracks with their followups
        single_track_followups = {
            t_id: [track_his[idx + 1] for idx, id in enumerate(track_his) if check_valid_followup(id, t_id, idx) ] for t_id in tqdm(tracks, total=len(tracks), desc="MC Single")
        }

        # --------------------
        # Build Markov Chain
        # --------------------

        # Initialize dictionary
        markov_chain = { t_id: None for t_id in tracks }

        # Compute Probabilities
        for track in tracks:
            followups = np.array(single_track_followups[track])
            unique_followups = np.unique(followups)

            probabilities = [ np.sum(t==followups)/len(followups) for t in unique_followups ]
            markov_chain[track] = [ (p, t) for p, t in zip(probabilities, unique_followups) ]

        filename = f"SingleMarkovChain_{datetime.strftime(datetime.now(), '%Y%m%dT%H:%M:%SZ')}.pkl"
        with open(os.path.join(self.markov_chains_dir, filename), "wb") as f:
            pickle.dump(markov_chain, f)
            logging.info(f"Successfully saved {filename}!!!")

        return markov_chain

    def double_prior_track_mc(self):
        # Tracks
        track_his = self.music_history["Track"].tolist()
        flags_his = self.music_history["Flag"].tolist()
        tracks_len = len(track_his)

        # Check if is a valid followup track
        check_valid_followup = lambda idx, xa, xb: (track_his[idx-2]==xa and track_his[idx-1]==xb and (flags_his[idx-2] != "[EOS]" or flags_his[idx-1] != "[EOS]"))

        # Get all tracks with their followups
        track_pair_followups = {
            (a, b): [ track_his[i] for i in range(2, tracks_len) if check_valid_followup(i, a, b) ] for a, b in tqdm(zip(track_his[:-1], track_his[1:]), total=len(track_his)-1, desc="MC Two")
        }

        # --------------------
        # Build Markov Chain
        # --------------------

        # Initialize dictionary
        markov_chain = { (a_id, b_id): None for a_id, b_id in zip(track_his[:-1], track_his[1:]) }

        # Compute Probabilities
        for pair in markov_chain.keys():
            followups = np.array(track_pair_followups[pair])
            probabilities = [np.sum(t == followups) / len(followups) for t in followups]
            markov_chain[pair] = [(p, t) for p, t in zip(probabilities, followups)]

        # Save model
        filename = f"DoubleMarkovChain_{datetime.strftime(datetime.now(), '%Y-%m-%dZ%H:%M:%S')}.pkl"
        with open(os.path.join(self.markov_chains_dir, filename), "wb") as f:
            pickle.dump(markov_chain, f)
            logging.info(f"Successfully saved {filename}!!!")

        return markov_chain

    def train(self):

        # Markov Chain with a single prior
        markov_chain_single_p = self.single_prior_track_mc()

        # Markov Chain for a double prior
        markov_chain_double_p = self.double_prior_track_mc()

        return markov_chain_single_p, markov_chain_double_p