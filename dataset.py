from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from scipy.linalg import dft
from underthesea import word_tokenize
from collections import Counter
from itertools import chain


class Vocabulary:
    """ The Vocabulary class is used to record words, which are used to convert
        text to numbers and vice versa.
    """

    def __init__(self):
        self.word2id = dict()
        self.word2id['<pad>'] = 0   # Pad Token
        self.word2id['<unk>'] = 1   # Unknown Token
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __len__(self):
        return len(self.word2id)

    def id2word(self, word_index):
        """
        @param word_index (int)
        @return word (str)
        """
        return self.id2word[word_index]

    def add(self, word):
        """ Add word to vocabulary
        @param word (str)
        @return index (str): index of the word just added
        """
        if word not in self:
            word_index = self.word2id[word] = len(self.word2id)
            self.id2word[word_index] = word
            return word_index
        else:
            return self[word]

    @staticmethod
    def tokenize_corpus(corpus):
        """Split the documents of the corpus into words
        @param corpus (list(str)): list of documents
        @return tokenized_corpus (list(list(str))): list of words
        """
        print("Tokenize the corpus...")
        tokenized_corpus = list()
        for document in tqdm(corpus):
            tokenized_document = [word.replace(
                " ", "_") for word in word_tokenize(document)]
            tokenized_corpus.append(tokenized_document)

        return tokenized_corpus

    def corpus_to_tensor(self, corpus, is_tokenized=False):
        """ Convert corpus to a list of indices tensor
        @param corpus (list(str) if is_tokenized==False else list(list(str)))
        @param is_tokenized (bool)
        @return indicies_corpus (list(tensor))
        """
        if is_tokenized:
            tokenized_corpus = corpus
        else:
            tokenized_corpus = self.tokenize_corpus(corpus)
        indicies_corpus = list()
        for document in tqdm(tokenized_corpus):
            indicies_document = torch.tensor(list(map(lambda word: self[word], document)),
                                             dtype=torch.int64)
            indicies_corpus.append(indicies_document)

        return indicies_corpus

    def tensor_to_corpus(self, tensor):
        """ Convert list of indices tensor to a list of tokenized documents
        @param indicies_corpus (list(tensor))
        @return corpus (list(list(str)))
        """
        corpus = list()
        for indicies in tqdm(tensor):
            document = list(
                map(lambda index: self.id2word[index.item()], indicies))
            corpus.append(document)

        return corpus

    # def add_words_from_corpus(self, corpus, is_tokenized=False):
    #     print("Add words from the corpus...")
    #     if is_tokenized:
    #         tokenized_corpus = corpus
    #     else:
    #         tokenized_corpus = self.tokenize_corpus(corpus)
    #     word_freq = Counter(chain(*tokenized_corpus))
    #     non_singletons = [w for w in word_freq if word_freq[w] > 1]
    #     print(f"Number of words in the corpus: {len(word_freq)}")
    #     print(f"Number of words with frequency > 1: {len(non_singletons)}")
    #     for word in non_singletons:
    #         self.add(word)


class IMDBDataset(Dataset):
    """ Load dataset from file csv"""

    def __init__(self, vocab, csv_fpath=None, tokenized_fpath=None):
        """
        @param vocab (Vocabulary)
        @param csv_fpath (str)
        @param tokenized_fpath (str)
        """
        self.vocab = vocab
        self.pad_idx = vocab["<pad>"]
        df = pd.read_csv(csv_fpath)
        self.sentiments_list = list(df.sentiment)
        self.reviews_list = list(df.vi_review)

        sentiments_type = list(set(self.sentiments_list))
        sentiments_type.sort()

        self.sentiment2id = {
            sentiment: i for i,
            sentiment in enumerate(sentiments_type)
        }

        if tokenized_fpath:
            self.tokenized_reviews = torch.load(tokenized_fpath)
        else:
            self.tokenized_reviews = self.vocab.tokenize_corpus(
                self.reviews_list)

        self.tensor_data = self.vocab.corpus_to_tensor(
            self.tokenized_reviews, is_tokenized=True)
        self.tensor_label = torch.tensor(
            [self.sentiment2id[sentiment]
                for sentiment in self.sentiments_list],
            dtype=torch.float64
        )

    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, idx):
        return self.tensor_data[idx], self.tensor_label[idx]

    def collate_fn(self, examples):
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)

        reviews = [e[0] for e in examples]
        reviews = torch.nn.utils.rnn.pad_sequence(
            reviews,
            batch_first=False,
            padding_value=self.pad_idx
        )
        reviews_lengths = torch.tensor([len(e[0]) for e in examples])
        sentiments = torch.tensor([e[1] for e in examples])

        return {"reviews": (reviews, reviews_lengths), "sentiments": sentiments}


def split_data(dataset):
    split_rate = 0.8
    full_size = len(dataset)
    train_size = (int)(split_rate * full_size)
    valid_size = (int)((full_size - train_size)/2)
    test_size = full_size - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset,
        lengths=[train_size, valid_size, test_size]
    )
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    BATCH_SIZE = 100
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )
    return train_dataloader, valid_dataloader, test_dataloader
