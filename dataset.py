import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# !wget https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/tok/train.lc.norm.tok.en
# !wget https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/tok/train.lc.norm.tok.de
# !wget https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/tok/val.lc.norm.tok.en
# !wget https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/tok/val.lc.norm.tok.de

with open("/content/train.lc.norm.tok.de", "r") as f:
    train_de = f.readlines()

with open("/content/train.lc.norm.tok.en", "r") as f:
    train_en = f.readlines()

with open("/content/val.lc.norm.tok.de", "r") as f:
    valid_de = f.readlines()

with open("/content/val.lc.norm.tok.en", "r") as f:
    valid_en = f.readlines()


train_de = list(map(lambda x: x.replace('\n', '').lower().split(), train_de))
train_en = list(map(lambda x: x.replace('\n', '').lower().split(), train_en))
valid_de = list(map(lambda x: x.replace('\n', '').lower().split(), valid_de))
valid_en = list(map(lambda x: x.replace('\n', '').lower().split(), valid_en))

en_corpus = [token for example in train_en+valid_en for token in example]
de_corpus = [token for example in train_de+valid_de for token in example]

en_cnter = Counter(en_corpus)
de_cnter = Counter(de_corpus)


en_stoi = {tok: idx+4 for idx, tok in enumerate(en_cnter)}
en_stoi['<sos>'] = 0
en_stoi['<eos>'] = 1
en_stoi['<pad>'] = 2
en_stoi['<unk>'] = 3
en_itos = {idx: tok for tok, idx in en_stoi.items()}

de_stoi = {tok: idx+4 for idx, tok in enumerate(de_cnter)}
de_stoi['<sos>'] = 0
de_stoi['<eos>'] = 1
de_stoi['<pad>'] = 2
de_stoi['<unk>'] = 3
de_itos = {idx: tok for tok, idx in de_stoi.items()}

en_encode = lambda sample: [en_stoi.get(tok, en_stoi['<unk>']) for tok in sample]
en_decode = lambda sample: [en_itos[idx] for idx in sample]

de_encode = lambda sample: [de_stoi.get(tok, de_stoi['<unk>']) for tok in sample]
de_decode = lambda sample: [de_itos[idx] for idx in sample]

class TranslationDataset(Dataset):
    def __init__(self, source_samples: list, target_samples: list, source_encode, target_encode):
        assert len(source_samples) == len(target_samples), "The source_samples and target_samples MUST have the same length."
        self.source_samples = source_samples
        self.target_samples = target_samples
        self.source_encode = source_encode
        self.target_encode = target_encode

    def __len__(self):
        return len(self.source_samples)

    def __getitem__(self, idx):
        source_nurmalized = self.source_encode(self.source_samples[idx])
        target_nurmalized = self.target_encode(self.target_samples[idx])

        return torch.tensor(source_nurmalized), torch.tensor(target_nurmalized)
    

def pad_collate_fn(batch):
    src_sents, tgt_sents = zip(*batch)
    
    # Pad source sentences
    src_sents_padded = pad_sequence(src_sents, batch_first=True, padding_value=en_stoi['<pad>'])
    
    # Pad target sentences
    tgt_sents_padded = pad_sequence(tgt_sents, batch_first=True, padding_value=de_stoi['<pad>'])
    
    return src_sents_padded, tgt_sents_padded