import tensorflow as tf
import pandas as pd
import numpy as np

import customersupport.common.utils

from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm_notebook as tqdm

class VocabHolder:
    def __init__(self, hparams):
        import csv
        
        self.vocab = None
        self.reverse_vocab = None
        self.glove_words = None
        self.glove_weights = None
        self.analyzer = None
        self.use_glove = hparams.use_glove
        
        if (hparams.glove_path is not None):
            self.glove_words = pd.read_table(hparams.glove_path,
                      sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
    
    def fit(self, x_text, y_text, max_vocab, verbose=True):
        count_vec = CountVectorizer(tokenizer=customersupport.common.utils.tweet_tokenize, max_features=max_vocab - 3)
        data = x_text + y_text
        
        if (verbose):
            print("Fitting CountVectorizer on X and Y text data...")
            data = tqdm(data)
            
        count_vec.fit(data)    
        analyzer = count_vec.build_analyzer()

        # Used to turn seq2seq predictions into human readable strings
        if (self.glove_words is not None):
            count_vec_keys = set(count_vec.vocabulary_.keys())

            known = list(count_vec_keys.intersection(self.glove_words.index))
            unknown = list(count_vec_keys.difference(known))

            self.vocab = dict(zip(unknown + known, list(range(3, max_vocab))))
            self.glove_weights = self.glove_words.loc[known].as_matrix()
            assert len(self.vocab) == max_vocab - 3
            
            if (verbose):
                print("Number of known words {}".format(len(known)))
        else:
            self.vocab = {k: v + 3 for k, v in count_vec.vocabulary_.items()}

        self.vocab['__pad__'] = customersupport.common.utils.PAD
        self.vocab['__unk__'] = customersupport.common.utils.UNK
        self.vocab['__start__'] = customersupport.common.utils.START

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        if (verbose):
            print("Learned vocab of {} items.".format(len(self.vocab)))
        self.analyzer = analyzer
        
        return self.analyzer
    
    def from_word_idx(self, word_idxs):
        return ' '.join(self.reverse_vocab[idx] for idx in word_idxs if idx != customersupport.common.utils.PAD).strip()
    
    def to_word_idx(self, sentence, max_msg_len):
        full_length = [self.vocab.get(tok, customersupport.common.utils.UNK) for tok in self.analyzer(sentence)]
        return np.array(full_length) if max_msg_len <= 0 else tf.keras.preprocessing.sequence.pad_sequences([full_length], 
                                                             maxlen=max_msg_len, 
                                                             padding='post', 
                                                             truncating='post', 
                                                             value = customersupport.common.utils.PAD)[0]

    def get_glove_weight(self, w):
        #np.zeros(glove_weights.shape[1])
        try:
            return self.glove_weights[w]
        except:
            return np.random.normal(.0, 0.1, self.glove_weights.shape[1])
