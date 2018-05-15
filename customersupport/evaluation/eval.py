import string

import tensorflow as tf
import numpy as np

from customersupport.evaluation.bleu import compute_bleu
from customersupport.evaluation.rouge import *

def get_evaluation_conf(sess, hparams, seq_func, loss_func, voc_holder):
    return tf.contrib.training.HParams(
        sess=sess,
        batch_size=hparams.batch_size,
        seq_func=seq_func,
        loss_func=loss_func,
        max_order=hparams.max_order,
        voc_holder=voc_holder,
        beam_width=hparams.beam_width,
        decoder_length=hparams.decoder_length)

def format_metrics(metrics):
    out = []
    for (m, v) in sorted(metrics.items()):
        out.append('{}: {}'.format(m, v))
        
    return '\n'.join(out)

def cos_similarity(a, b):
    # adding contant to avoid division by zero
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def to_glove_vec(words, voc_holder):
    glove_vecs = []
   
    for r in (r for r in words):
        glove_vecs.append(voc_holder.get_glove_weight(r))
        
    return glove_vecs

def utterence_to_glove_vec(words, voc_holder):
    glove_vecs = to_glove_vec(words, voc_holder)
    
    return np.mean(glove_vecs, axis=0)

def utterence_to_vec_extrema(words, voc_holder):
    glove_vecs = to_glove_vec(words, voc_holder)
    
    maxs = np.maximum.reduce(glove_vecs)
    mins = np.minimum.reduce(glove_vecs)
    vec_extrema = [mins[i] if (abs(mins[i]) > maximum) else maximum for  i, maximum in enumerate(maxs)]
    
    return vec_extrema

def strip_punkt(words, reverse_vocab):
    exclude = set(string.punctuation)
    
    return [w for w in words if ((w > 0) and (reverse_vocab[w] not in exclude))]

def greedy_matching(t1, t2, voc_holder):
    score = 0.
    t1_gloves = np.asarray([voc_holder.get_glove_weight(w) for w in t1])    
    t2_gloves = np.asarray([voc_holder.get_glove_weight(w) for w in t2])
    
    import scipy.spatial as sp

    cos_sims =  1 - sp.distance.cdist(t1_gloves, t2_gloves, 'cosine')
    sorted_cos_sims =  np.argsort(cos_sims, axis=1)
    used = set()
    min_words = min(len(t1), len(t2))
    
    for i in range(0, min_words):
        j = sorted_cos_sims.shape[1] - 1
        
        while (sorted_cos_sims[i, j] in used):
            j -= 1
        w_idx = sorted_cos_sims[i, j]
        used.add(w_idx)
        score += cos_sims[i, w_idx]
        
    return score / len(t1)

def evaluate_words_index(references, hypothesis, eval_conf, metrics, verbose = False):
    results = {}
    voc_holder = eval_conf.voc_holder
    
    if ("bleu" in metrics):
        bleu = compute_bleu(
            references.reshape(-1, 1), hypothesis, max_order=eval_conf.max_order, smooth=True)[0]
        results['BLEU@{}'.format(eval_conf.max_order)] = bleu * 100.
        
    if ("rouge_l" in metrics):
        refs = [voc_holder.from_word_idx(r) for r in references]
        hyps = [voc_holder.from_word_idx(a) for a in hypothesis]
        
        # Calculate ROUGE-L F1, precision, recall scores
        rouge_l = [
            rouge_l_sentence_level([hyp], [ref])
            for hyp, ref in zip(hyps, refs)
        ]
        rouge_l_f, rouge_l_p, rouge_l_r = map(np.mean, zip(*rouge_l))

        results['ROUGE_L'] = rouge_l_f * 100.
        del hyps, refs
    
    if ("embedding_average" in metrics):
        emb_sim = []
        seq = zip(references, hypothesis)
        for i, (r, h) in enumerate(seq):
            r = utterence_to_glove_vec(r, voc_holder)
            h = utterence_to_glove_vec(h, voc_holder)
            emb_sim.append(cos_similarity(r, h))

        results['Embedding Average'] = np.mean(emb_sim) * 100.
        
        del emb_sim
        
    if ("vector_extrema" in metrics):
        emb_sim = []
        seq = zip(references, hypothesis)
        for (r, h) in seq:
            r = utterence_to_vec_extrema(r, voc_holder)
            h = utterence_to_vec_extrema(h, voc_holder)
            emb_sim.append(cos_similarity(r, h))

        results['Vector Extrema'] = np.mean(emb_sim) * 100.
        
        del emb_sim
        
    if ("greedy_matching" in metrics):
        emb_sim = []
        seq = zip(references, hypothesis)
        for (r, h) in seq:
            s_t1 = greedy_matching(r, h, voc_holder)
            s_t2 = greedy_matching(h, r, voc_holder)
            
            emb_sim.append((s_t1 + s_t2) / 2)
            
        results['Greedy Matching'] = np.mean(emb_sim) * 100.

        del emb_sim
        
    return results