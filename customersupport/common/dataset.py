import re
import random
import string

from datetime import datetime

import numpy as np
import pandas as pd
import customersupport.common.utils

from tqdm import tqdm_notebook as tqdm


class CustomerSupportDataset:
    TOP_COMPANIES = ([
        'AmazonHelp', 'AppleSupport', 'Uber_Support', 'SpotifyCares',
        'AmericanAir', 'Delta', 'comcastcares', 'TMobileHelp', 'SouthwestAir',
        'Ask_Spectrum', 'Tesco', 'British_Airways', 'UPSHelp', 'hulu_support',
        'VirginTrains', 'ChipotleTweets', 'sprintcare', 'XboxSupport',
        'AskPlayStation', 'AskTarget'
    ])

    def __init__(self, hparams, verbose = True):
        self.inbounds_and_outbounds = self.__load_tweets(hparams.tweets_path, 
            hparams.first_day, hparams.last_day, hparams.companies)
        self.verbose = verbose
        
        self.x = None
        self.y = None
        
        self.x_text = None
        self.y_text = None
        
        self.train_weights = None
        self.test_weights = None
        
        self.test_idx = None
        self.train_idx = None

    def __create_context(self, df):
        last_conv_id = -1
        contexts = []
        for _, row in df.iterrows():
            conv_id = row['conv_id_x']
            if (last_conv_id != conv_id):
                context = ''
                last_conv_id = conv_id

            contexts.append(context)
            context = '\n'.join([row['text_y'], row['text_x'], context])
        
        return contexts

    def __calculate_time_diff(self, df):
        TIME_FORMAT = '%a %b %d %H:%M:%S %z %Y'
        tdff = (df.created_at_y
                 #.replace(minute=0, second=0, hour=0)
                 .apply(lambda x: datetime.strptime(x, TIME_FORMAT).timestamp())
                )
        return (tdff.max() - tdff) / 86400

    def __load_tweets(self, path, first_day, last_day, companies):
        tweets = pd.read_csv(path)
        tweets = tweets[tweets.conv_id >= 0]

        author_ids = (tweets[(tweets['inbound'] == False)].groupby(
            ['conv_id',
             'author_id']).first().reset_index()[['conv_id', 'author_id']]
                      .rename(columns={
                          'author_id': 'support_author'
                      }).set_index('conv_id'))

        tweets = tweets.join(author_ids, on='conv_id')

        top_author_tweets = tweets[(tweets['inbound'] == True) & (
            tweets['support_author'].isin(self.TOP_COMPANIES))]
        print('Done support_author {}'.format(top_author_tweets.shape))

        top_author_tweets = top_author_tweets[top_author_tweets[
            'support_author'].isin(companies)]

        inbounds_and_outbounds = pd.merge(
            top_author_tweets,
            tweets,
            left_on='tweet_id',
            right_on='in_response_to_tweet_id').sample(frac=1)

        # Filter to only outbound replies (from companies)
        inbounds_and_outbounds = inbounds_and_outbounds[
            inbounds_and_outbounds.inbound_y ^ True]
        inbounds_and_outbounds = inbounds_and_outbounds[
            inbounds_and_outbounds.author_id_y.isin(companies)]
        inbounds_and_outbounds = inbounds_and_outbounds.sort_values(
            ['conv_id_y', 'created_at_y'])

        inbounds_and_outbounds['context'] = self.__create_context(
            inbounds_and_outbounds)
        inbounds_and_outbounds['time_diff'] = self.__calculate_time_diff(
            inbounds_and_outbounds)

        #remove older then 60 days
        inbounds_and_outbounds = inbounds_and_outbounds[inbounds_and_outbounds[
            'time_diff'].between(first_day, last_day)]
        inbounds_and_outbounds[
            'time_diff'] = inbounds_and_outbounds['time_diff'] - first_day

        return inbounds_and_outbounds

    def __calc_weights(self, idx, decay_rate):
        weights = self.inbounds_and_outbounds.loc[self.x_text.iloc[list(idx)].index,
                                             'time_diff'].values
        weights = decay_rate**weights
        weights /= weights.mean()

        return weights

    def __fit_weights(self, decay_rate):
        train_weights = self.__calc_weights(self.train_idx, decay_rate)
        test_weights = self.__calc_weights(self.test_idx, decay_rate)

        #         assert self.train_x.shape[0] == len(train_weights)
        #         assert self.test_x.shape[0] == len(self.test_x)

        if (self.verbose):
            print(pd.Series(train_weights).describe())
            print(pd.Series(test_weights).describe())
            
        return (train_weights, test_weights)

    def process_utterances(self, masks=None):
        if (self.verbose):
            print("Replacing anonymized screen names in X...")

        x_text = self.inbounds_and_outbounds.progress_apply(
            lambda row: customersupport.common.utils.clean_tweets_text(row['text_x'] + ' ' + row['context']),
            axis=1)
        
        if (self.verbose):
            print("Replacing anonymized screen names in Y...")
        y_text = self.inbounds_and_outbounds.text_y.progress_apply(
            lambda txt: customersupport.common.utils.clean_tweets_text(txt))
        
        mask = x_text.str.match('<user> <url>\s*$') ^ True
        
        if (masks is not None):
            for s in masks:
                mask = mask & (y_text.str.contains(s) ^ True)
                
        x_text = x_text[mask]
        y_text = y_text[mask]

        self.x_text = x_text
        self.y_text = y_text

    def text_to_vec(self, hparams, voc_holder):
        if (self.verbose):
            print("Calculating word indexes for X...")
        self.x = pd.np.vstack(
            self.x_text.progress_apply(
                lambda u: voc_holder.to_word_idx(u, hparams.encoder_length))
            .values)
        
        if (self.verbose):
            print("Calculating word indexes for Y...")
        self.y = pd.np.vstack(
            self.y_text.progress_apply(
                lambda u: voc_holder.to_word_idx(u, hparams.decoder_length))
            .values)

    def train_test_split(self, hparams, do_random):
        random.seed(42)
        all_idx = list(range(len(self.x)))
        if (do_random):
            self.train_idx = set(
                random.sample(all_idx, int(hparams.train_size * len(all_idx))))
        else:
            self.train_idx = set([
                i for i, x in enumerate(
                    self.x_text.index.isin(self.x_text[(
                        self.inbounds_and_outbounds.time_diff >
                        hparams.train_time_diff)].index).tolist()) if x
            ])

        self.test_idx = {idx for idx in all_idx if idx not in self.train_idx}

        self.train_x = self.x[list(self.train_idx)]
        self.test_x = self.x[list(self.test_idx)]
        self.train_y = self.y[list(self.train_idx)]
        self.test_y = self.y[list(self.test_idx)]

        assert self.train_x.shape[0] == self.train_y.shape[0]
        assert self.test_x.shape[0] == self.test_y.shape[0]
        
        if (self.verbose):
            print('Training data of shape {} and test data of shape {}.'.format(
                self.train_x.shape, self.test_y.shape))

        (self.train_weights, self.test_weights) = self.__fit_weights(hparams.decay_rate)
