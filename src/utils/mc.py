from datetime import datetime as dt

def get_best_option(options, playlist):

    found = False
    one_thread_playlist = sum(playlist, [])

    # Traverse through all options
    for prob, song in options:
        # Search if already present in playlist
        if song in one_thread_playlist:
            continue
        else:
            return prob, song

    return None, None

def build_playlist(pivots, ss_mc, ds_mc, playlist_length=25):
    """
    This function generates a music playlist based on the user's previous music history
    its goal is to build an appealing song based on how the user tends to build their music.
    :param pivots:  (list) N pivot songs to use as basis to build the playlist
    :param ss_mc:   (dict) Single Sample Markov Chain model
    :param ds_mc:   (dict) Double Sample Markov Chain model
    :return: (list)
    """

    # Number of pivots
    N = len(pivots)

    # Calculate length of each section of the playlist
    section_length = playlist_length // N - 1

    # Placeholder for the playlist
    playlist = [[ None for _ in range(section_length) ] for x in range(N)]

    for j in range(section_length):
        for i in range(N):
            if j == 0:
                playlist[i][j] = pivots[i]
            elif j == 1:
                options = ss_mc[playlist[i][j - 1]]
                _, song = get_best_option(options, playlist)
                playlist[i][j] = song
            else:
                last_song, sec_song = playlist[i][j-1], playlist[i][j-2]
                ss_prob, ss_song = get_best_option(ss_mc[last_song], playlist)
                ds_prob, ds_song = get_best_option(ds_mc[(sec_song, last_song)], playlist)
                playlist[i][j] = ss_song if (ds_song is None or ss_prob > ds_prob) else ds_song

    # Join playlist
    playlist = sum(playlist, [])

    return {
        "Playlist": playlist,
        "Creation Date": dt.strftime(dt.now(), "%Y%m%dZ_%H%M%ST")
    }