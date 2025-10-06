import os
import torch
from model import LSTM, SimpleRNN


def load_model_checkpoint(word_embedding, vocab, device):
    # ===== Load models =====
    models = {}
    if os.path.exists("models/best_model_lstm.pt"):
        INPUT_DIM = word_embedding.vectors.shape[0]
        EMBEDDING_DIM = 100
        HIDDEN_DIM = 256
        N_LAYERS = 2
        BIDIRECTIONAL = True
        DROPOUT = 0.5
        PAD_IDX = vocab["<pad>"]
        UNK_IDX = vocab["<unk>"]

        model_lstm = LSTM(
            INPUT_DIM,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            N_LAYERS,
            BIDIRECTIONAL,
            DROPOUT,
            PAD_IDX
        )
        model_lstm.load_state_dict(torch.load(
            "models/best_model_lstm.pt", map_location=device))
        model_lstm.to(device).eval()
        models["lstm"] = model_lstm

    if os.path.exists("models/best_model_rnn.pt"):
        INPUT_DIM = word_embedding.vectors.shape[0]
        EMBEDDING_DIM = 100
        HIDDEN_DIM = 256 * 2

        PAD_IDX = vocab["<pad>"]
        UNK_IDX = vocab["<unk>"]

        # vocab_size, embedding_dim, hidden_dim, pad_idx
        model_rnn = SimpleRNN(
            INPUT_DIM,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            PAD_IDX
        )
        model_rnn.load_state_dict(torch.load(
            "models/best_model_rnn.pt", map_location=device))
        model_rnn.to(device).eval()
        models["rnn"] = model_rnn
    return models


def predict_sentiment(model, sentence, vocab, device):
    corpus = [sentence]
    tensor = vocab.corpus_to_tensor(corpus)[0].to(device)
    tensor = tensor.unsqueeze(1)
    length = [len(tensor)]
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    if prediction.item() > 0.5:
        return True, f"{prediction.item():.2f}"
    else:
        return False, f"{(1 - prediction.item()):.2f}"
