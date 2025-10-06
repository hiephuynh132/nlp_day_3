import gdown
import os

if os.path.exists("downloads"):
    print("downloads is exists")
else:
    url = "https://drive.google.com/drive/folders/1x29Nfs2a6JtLrjs0StlC6m5C_5f_dWwv"
    gdown.download_folder(url, output="downloads", quiet=False)

if os.path.exists("models/best_model_lstm.pt"):
    print("best_model_lstm.pt is exists")
else:
    url = "https://drive.google.com/uc?id=1Fq5fj5usZ3w-mjPEJWEtGRrXCY5WJ2hL"
    gdown.download(url, output="models/best_model_lstm.pt", quiet=False)

if os.path.exists("models/best_model_rnn.pt"):
    print("best_model_rnn.pt is exists")
else:
    url = "https://drive.google.com/uc?id=1Cj5olmuN7zETALikJwyardYqiF9cDOac"
    gdown.download(url, output="models/best_model_rnn.pt", quiet=False)

print("GET DATA FROM DRIVE FINISH.")