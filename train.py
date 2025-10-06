import time

import csv
import torch
import torchtext.vocab as torch_vocab

from dataset import Vocabulary, IMDBDataset, split_data
from utils import train, evaluate, epoch_time, init_model
import argparse

parser = argparse.ArgumentParser(description="Train sentiment model")
parser.add_argument("--model_name", type=str, default="lstm", choices=["lstm", "rnn"], help="Chọn mô hình cần huấn luyện")
args = parser.parse_args()

model_name = args.model_name
print(f"🚀 Training model: {model_name}")

# Dataloader
word_embedding = torch_vocab.Vectors(
    name="./downloads/vi_word2vec.txt",
    unk_init=torch.Tensor.normal_
)

vocab = Vocabulary()
words_list = list(word_embedding.stoi.keys())
for word in words_list:
    vocab.add(word)

dataset = IMDBDataset(
    vocab,
    "./downloads/VI_IMDB.csv",
    "./downloads/tokenized.pt"
)

train_dataloader, valid_dataloader, test_dataloader = split_data(dataset)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, optimizer, criterion = init_model(
    vocab, word_embedding, device, model_name=model_name)


N_EPOCHS = 50
best_valid_loss = float("inf")

log_path = f"models/train_log_{model_name}.csv"

# Tạo file log và header
with open(log_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train(
        model, train_dataloader, optimizer, criterion, device
    )
    valid_loss, valid_acc = evaluate(
        model, valid_dataloader, criterion, device
    )

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f"models/best_model_{model_name}.pt")

    print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")

    # ===== Ghi log vào CSV =====
    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [epoch+1, train_loss, train_acc, valid_loss, valid_acc])
