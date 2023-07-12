from models.preprocessing import preprocessing
from models.markov_chain import MarkovChain

mc_dir = "models/markov_chain/"

tracks_file = "data/AppleMusic/Apple Music Library Tracks.json"
history_file = "data/AppleMusic/Apple Music Play Activity.csv"


if __name__ == "__main__":
    df_tracks, df_history = preprocessing(tracks_file=tracks_file, history_file=history_file)

    markov_chain = MarkovChain(music_history=df_history, track_ids=df_tracks, markov_chains_dir=mc_dir)
    single_mc, double_mc = markov_chain.train()