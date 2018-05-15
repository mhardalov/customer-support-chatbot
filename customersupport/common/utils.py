import re

import numpy as np

from datetime import datetime
from nltk.tokenize import TweetTokenizer

# Tokens needed for seq2seq
PAD = 0  # after message has finished, this fills all remaining vector positions
UNK = 1  # words that aren't found in the vocab
START = 2  # provided to the model at position 0 for every response predicted


def tweet_tokenize(text):
    tknzr = TweetTokenizer(preserve_case=False, strip_handles=False)
    return tknzr.tokenize(text)

def remove_operator_nums(text):
    return re.sub(r'\d+\/\d+$', ' ', text)

def replace_tweeter_handles(text):
    return re.sub(r'@[0-9A-Za-z_\-]+', '<user>', text)

def replace_tweeter_hashtags(text):
    return re.sub(r'#[0-9A-Za-z_\-]+', '<hashtag>', text)

def replace_urls(text):
    return re.sub(r'https://t\.co/\w+', '<url>', text)

def replace_versions(text):
    replace_pattern = '\g<1><version> \g<4>'
    pattern = '(\s)((\d+\.){1,2}\d+)([\.\?!,]{0,1}[\s]|$)'
    text = re.sub(pattern, replace_pattern, text)
    
    return text

def replace_models(text):
    try:
        match_pattern = r'(iphone\s*(([0-9]+[sc]{0,2}(\s((plus)|(\+))){0,1})|([x])|(se)))'
        device_search = re.search(match_pattern, text, re.IGNORECASE)

        if device_search:
            device = device_search.group(0)

            device_pretty = device.lower()
            device_pretty = device_pretty.replace('+', 'plus')
            device_pretty = device_pretty.replace('10', 'x')
            
            device_split = re.search(r'(iphone)\s*([\dsxe]+)\s*(plus){0,1}', device_pretty)
            device_pretty = ' '
            for g in device_split.groups():
                if (g is not None):
                    device_pretty += g + ' '

            device_pretty =  device_pretty.lower()

            text = text.replace(device, device_pretty)
    except:
        pass
        
    return text
    
def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r"’", "'", text)
    text = re.sub(r"i'm", "i am", text, flags=re.I)
    text = re.sub(r"i phone", "iphone", text, flags=re.I)
    text = re.sub(r"he's", "he is", text, flags=re.I)
    text = re.sub(r"she's", "she is", text, flags=re.I)
    text = re.sub(r"it's", "it is", text, flags=re.I)
    text = re.sub(r"that's", "that is", text, flags=re.I)
    text = re.sub(r"what's", "that is", text, flags=re.I)
    text = re.sub(r"where's", "where is", text, flags=re.I)
    text = re.sub(r"how's", "how is", text, flags=re.I)
    text = re.sub(r"y'all", "you all", text, flags=re.I)
    #check if it's correct
    text = re.sub(r"'s", "s", text, flags=re.I)
    text = re.sub(r"\'ll", " will", text, flags=re.I)
    text = re.sub(r"\'ve", " have", text, flags=re.I)
    text = re.sub(r"\'re", " are", text, flags=re.I)
    text = re.sub(r"\'d", " would", text, flags=re.I)
    text = re.sub(r"\'re", " are", text, flags=re.I)
    text = re.sub(r"won't", "will not", text, flags=re.I)
    text = re.sub(r"can't", "cannot", text, flags=re.I)
    text = re.sub(r"n't", " not", text, flags=re.I)
    text = re.sub(r"n'", "ng", text, flags=re.I)
    text = re.sub(r"'bout", "about", text, flags=re.I)
    text = re.sub(r"'til", "until", text, flags=re.I)
    #text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "",text)
    text = re.sub(r"dm", "direct message", text, flags=re.I)
    text = re.sub(r"direct message", "direct message", text, flags=re.I)
    text = re.sub(r'wtf', 'what the fuck', text, flags=re.I)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\sty\s", ' thank you ', text, flags=re.I)
    text = re.sub(r"\su\s", ' you ', text, flags=re.I)
    text = re.sub(r'appletv', 'apple tv', text, flags=re.I)
    
    return text

def clean_tweets_text(text):
    text = text.lower()
    text = clean_text(text)
    text = replace_urls(text)
    text = remove_operator_nums(text)
    text = replace_tweeter_handles(text)
    text = replace_tweeter_hashtags(text)
    text = replace_versions(text)
    text = replace_models(text)
    
    return text

def add_start_token(y_array):
    """ Adds the start token to vectors.  Used for training data. """
    return np.insert(y_array, 0, START, 1)

def add_start_token(y_array):
    """ Adds the start token to vectors.  Used for training data. """
    return np.hstack([
        START * np.ones((len(y_array), 1)),
        y_array[:, :-1],
    ])

def add_start_token(y_array):
    """ Adds the start token to vectors.  Used for training data. """
    return np.hstack([
        START * np.ones((len(y_array), 1)),
        y_array,
    ])

def transform_batch(x, y, rand_idx, weights = None, 
                   batch_size = 128, mask_pads = True):
    b_train_y = y[rand_idx]

    input_train_y = add_start_token(y[rand_idx])
    sample_weights = None
    
    if ((weights is not None) or (mask_pads)):
        max_msg_len = y.shape[1]
        sample_weights = np.zeros([batch_size, max_msg_len])
        if (mask_pads):
            for i, idx in enumerate(rand_idx):
                weight = weights[idx] if (weights is not None) else 1.
                sample_weights[i, :min(max_msg_len, (input_train_y[i].nonzero()[0].argmax()) + 2)] = weight
        else:
            sample_weights = weights[rand_idx]
    input_x = x[rand_idx]
        
    return ([input_x, input_train_y], b_train_y, sample_weights)
