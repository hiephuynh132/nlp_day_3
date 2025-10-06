import torch
import torch.nn as nn
import torch.optim as optim
from model import LSTM, SimpleRNN


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    @param preds (torch.Tensor): shape = [batch_size]
    @param y (torch.Tensor): shape = [batch_size]
    @return acc (torch.Tensor): shape = [1]
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, dataloader, optimizer, criterion, device):
    """
    @param model (RNN)
    @param dataloader (DataLoader)
    @param optimizer (torch.optim)
    @param criterion (torch.nn.modules.loss)
    @param device (torch.device)
    @return epoch_loss (float): model's loss of this epoch
    @return epoch_acc (float): model's accuracy of this epoch
    """
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in dataloader:

        optimizer.zero_grad()
        reviews, reviews_lengths = batch["reviews"]
        reviews = reviews.to(device)
        predictions = model(reviews, reviews_lengths).squeeze(1)
        sentiments = batch["sentiments"].to(device)
        loss = criterion(predictions, sentiments)
        acc = binary_accuracy(predictions, sentiments)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    batch_num = len(dataloader)
    return epoch_loss / batch_num, epoch_acc / batch_num


def evaluate(model, dataloader, criterion, device):
    """
    @param model (RNN)
    @param dataloader (DataLoader)
    @param criterion (torch.nn.modules.loss)
    @param device (torch.device)
    @return epoch_loss (float): model's loss of this epoch
    @return epoch_acc (float): model's accuracy of this epoch
    """
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in dataloader:

            reviews, reviews_lengths = batch["reviews"]
            reviews = reviews.to(device)
            predictions = model(reviews, reviews_lengths).squeeze(1)

            sentiments = batch["sentiments"].to(device)
            loss = criterion(predictions, sentiments)
            acc = binary_accuracy(predictions, sentiments)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    batch_num = len(dataloader)
    return epoch_loss / batch_num, epoch_acc / batch_num


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_model(vocab, word_embedding, device, model_name="lstm"):
    if model_name == "rnn":
        INPUT_DIM = word_embedding.vectors.shape[0]
        EMBEDDING_DIM = 100
        HIDDEN_DIM = 256 * 2

        PAD_IDX = vocab["<pad>"]
        UNK_IDX = vocab["<unk>"]

        # vocab_size, embedding_dim, hidden_dim, pad_idx
        model = SimpleRNN(
            INPUT_DIM,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            PAD_IDX
        )
    else:
        INPUT_DIM = word_embedding.vectors.shape[0]
        EMBEDDING_DIM = 100
        HIDDEN_DIM = 256
        N_LAYERS = 2
        BIDIRECTIONAL = True
        DROPOUT = 0.5
        PAD_IDX = vocab["<pad>"]
        UNK_IDX = vocab["<unk>"]

        model = LSTM(
            INPUT_DIM,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            N_LAYERS,
            BIDIRECTIONAL,
            DROPOUT,
            PAD_IDX
        )

    model.embedding.weight.data.copy_(word_embedding.vectors)
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss().to(device)

    model = model.to(device)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    return model, optimizer, criterion
