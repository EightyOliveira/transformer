"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from conf import *
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer
import json


tokenizer = Tokenizer()
loader = DataLoader(ext=('.en', '.de'),
                    tokenize_en=tokenizer.tokenize_en,
                    tokenize_de=tokenizer.tokenize_de,
                    init_token='<sos>',
                    eos_token='<eos>')
# len(train) == 29000 len(valid) = 1014 len(test) = 1000
train, valid, test = loader.make_dataset()

# Build vocabulary :
#   - Only words that appear at least `min_freq` times in the training data are included.
#   - Words with frequency lower than `min_freq` will be mapped to <unk> (unknown token).
loader.build_vocab(train_data=train, min_freq=2)


# save stoi (str -> int)
with open('.data/vocab/source_stoi.json', 'w', encoding='utf-8') as f:
    json.dump(loader.source.vocab.stoi, f, ensure_ascii=False)

with open('.data/vocab/target_stoi.json', 'w', encoding='utf-8') as f:
    json.dump(loader.target.vocab.stoi, f, ensure_ascii=False)


train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
                                                     batch_size=batch_size,
                                                     device=device)
# source_stoi.json <pad> id
src_pad_idx = loader.source.vocab.stoi['<pad>']
trg_pad_idx = loader.target.vocab.stoi['<pad>']
trg_sos_idx = loader.target.vocab.stoi['<sos>']
# 5893
enc_voc_size = len(loader.source.vocab)
# 7853
dec_voc_size = len(loader.target.vocab)
