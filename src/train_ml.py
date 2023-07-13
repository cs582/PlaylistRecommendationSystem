import torch
import tqdm
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.utils.data import MusicDataset
from models.transformer_based_recommender import RecommendationSystem
from models.preprocessing import preprocessing
from models.markov_chain import MarkovChain


tracks_file = "data/AppleMusic/Apple Music Library Tracks.json"
history_file = "data/AppleMusic/Apple Music Play Activity.csv"

if __name__=="__main__":
    df_tracks, df_history = preprocessing(tracks_file=tracks_file, history_file=history_file)

    tokenizer = { track: i+2 for i, track in enumerate(df_history["Track"].drop_duplicates().tolist()) }
    tokenizer["[UKN]"] = 0
    tokenizer["[EOS]"] = 1

    vocab_size = len(df_tracks) + 2
    device = torch.device('mps')

    out_size = len(df_tracks)

    min_tracks = 5

    layers = 4

    batch_size = 64

    model_dim = 256
    ff_dim = 512
    nhead = 8

    eos_id = 1
    max_length = 80

    model = RecommendationSystem(
        vocab_size=vocab_size,
        max_length=max_length,
        nhead=nhead,
        model_dim=model_dim,
        ff_dim=ff_dim,
        layers=layers,
        device=device,
        eos_id=eos_id,
        out_size=out_size
    ).to(device)

    x = []
    y = []

    track_history = df_history.iloc[:, 0].tolist()
    for i in range(len(tokenizer) - max_length):
        sequence = [tokenizer[x] for x in track_history[i:i+max_length]]
        for j in range(min_tracks, max_length-1):
            if np.random.rand() < 0.5:
                x.append(sequence[:j] + [1] + [0] * (max_length-j-1))
                y.append(sequence[j])

    x, y = np.asarray(x), np.asarray(y)

    train_size = np.random.rand(len(x)) < 0.75
    x_train = torch.Tensor(x[train_size]).to(device, dtype=torch.int32)
    y_train = torch.Tensor(y[train_size]).to(device, dtype=torch.int32)

    x_test = torch.Tensor(x[~train_size]).to(device, dtype=torch.int32)
    y_test = torch.Tensor(y[~train_size]).to(device, dtype=torch.int32)

    print(f"Train data: x shape {x_train.shape}, y shape {y_train.shape}")
    print(f"Test data: x shape {x_test.shape}, y shape {y_test.shape}")

    train_dataset = MusicDataset(x_train, y_train)
    test_dataset = MusicDataset(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, eps=1e-6, weight_decay=0.2, betas=(0.9, 0.98))

    pbar = tqdm.tqdm(total=len(train_dataloader))

    for user_songs, y in train_dataloader:

        # Set grad to zero
        optimizer.zero_grad(set_to_none=True)

        # Make prediction
        y_hat = model(user_songs)

        # Calculate loss
        loss = criterion(y_hat, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        pbar.update()
        pbar.set_description(f"LOSS:{np.round(loss.item(), 3)}")




