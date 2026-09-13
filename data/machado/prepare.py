import os
import pickle
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# ajuste esse número: corpus pequeno pede vocabulário bem menor que os
# 50257 tokens do GPT-2 original — algo entre 2000 e 8000 tende a ser
# um bom ponto de partida para um corpus do tamanho da obra do Machado
VOCAB_SIZE = 4000

# le o texto do Machado que voce ja disponibilizou nesta pasta
input_file_path = os.path.join(os.path.dirname(__file__), 'romances_machado.txt')
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# treina o tokenizador BPE do zero, usando so o texto de treino (nunca o de
# validacao, para nao "vazar" informacao da validacao para dentro do vocabulario)
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=2)
tokenizer.train_from_iterator([train_data], trainer=trainer)

# salva o tokenizador treinado em disco - o sample.py vai precisar dele depois
# para conseguir decodificar o texto gerado pelo modelo
tokenizer.save(os.path.join(os.path.dirname(__file__), 'tokenizer.json'))

def encode(s):
    return tokenizer.encode(s).ids

def decode(ids):
    return tokenizer.decode(ids)

train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# salva o vocab_size para o train.py usar ao montar o modelo
meta = {'vocab_size': tokenizer.get_vocab_size()}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"vocab_size: {tokenizer.get_vocab_size()}")