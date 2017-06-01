#!/usr/bin/env python                                                                                                                                                    
# -*- coding: utf-8 -*-                                                                                                                                                  

import sys
import numpy as np
import mecab as mcb
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L



class seq2seq(chainer.Chain):
    def __init__(self, jv, ev, k, jvocab, evocab):
        super(seq2seq, self).__init__(
            embedx = L.EmbedID(jv, k),
            embedy = L.EmbedID(ev, k),
            H = L.LSTM(k, k),
            W = L.Linear(k, ev),
            )

    def __call__(self, jline, eline, jvocab, evocab):
        for i in range(len(jline)):
            wid = jvocab[jline[i]]
            x_k = self.embedx(Variable(np.array([wid], dtype=np.int32)))
            h = self.H(x_k)
        x_k = self.embedx(Variable(np.array([jvocab['<eos>']], dtype=np.int32)))
        tx = Variable(np.array([evocab[eline[0]]], dtype=np.int32))
        h = self.H(x_k)
        accum_loss = F.softmax_cross_entropy(self.W(h), tx)
        for i in range(1,len(eline)):
            wid = evocab[eline[i]]
            x_k = self.embedy(Variable(np.array([wid], dtype=np.int32)))
            next_wid = evocab['<eos>'] if (i == len(eline) - 1) else evocab[eline[i+1]]
            tx = Variable(np.array([next_wid], dtype=np.int32))
            h = self.H(x_k)
            loss = F.softmax_cross_entropy(self.W(h), tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss

        return accum_loss

def mt(model, jline, id2wd, jvocab, evocab):
    for i in range(len(jline)):
        wid = jvocab[jline[i]]
        x_k = model.embedx(Variable(np.array([wid], dtype=np.int32), volatile='on'))
        h = model.H(x_k)
    x_k = model.embedx(Variable(np.array([jvocab['<eos>']], dtype=np.int32), volatile='on'))
    h = model.H(x_k)
    wid = np.argmax(F.softmax(model.W(h)).data[0])
    if wid in id2wd:
        print id2wd[wid],
    else:
        print wid,
    loop = 0
    while (wid != evocab['<eos>']) and (loop <= 30):
        x_k = model.embedy(Variable(np.array([wid], dtype=np.int32), volatile='on'))
        h = model.H(x_k)
        wid = np.argmax(F.softmax(model.W(h)).data[0])
        if wid in id2wd:
            print id2wd[wid],
        else:
            print wid,
        loop += 1
    print

def constructVocabs(corpus, mod):

    vocab = {}
    id2wd = {}
    lines = open(corpus).read().split('\n')
    for i in range(len(lines)):
        lt = lines[i].split()
        for w in lt:
            if w not in vocab:
                if mod == "U":
                    vocab[w] = len(vocab)
                elif mod == "R":
                    id2wd[len(vocab)] = w
                    vocab[w] = len(vocab)

    if mod == "U":
        vocab['<eos>'] = len(vocab)
        v = len(vocab)
        return vocab, v
    elif mod == "R":
        id2wd[len(vocab)] = '<eos>'
        vocab['<eos>'] = len(vocab)
        v = len(vocab)
        return vocab, v, id2wd

def main(mpath, utt_file, res_file):

    jvocab, jv = constructVocabs(utt_file, mod="U")
    evocab, ev, id2wd = constructVocabs(res_file, mod="R")

    demb = 100
    model = seq2seq(jv, ev, demb, jvocab, evocab)
    serializers.load_npz(mpath, model)

    while True:
        utterance = raw_input()
        if utterance == "exit":
            print "Bye!!"
            sys.exit(0)

        jln = mcb.construct_BoW(utterance)
        jlnr = jln[::-1]
        mt(model, jlnr, id2wd, jvocab, evocab)


if __name__ == "__main__":

    argvs = sys.argv

    _usage = """--                                                                                                                                                       
Usage:                                                                                                                                                                   
    python generating.py [model] [uttranceDB] [responseDB]                                                                                                               
Args:                                                                                                                                                                    
    [model]: The argument is seq2seq model to be trained using dialog corpus.                                                                                            
    [utteranceDB]: The argument is utterance corpus to gain the distributed representation of words.                                                                     
    [responseDB]: The argument is response corpus to gain the distributed representation of words.                                                                       
""".rstrip()

    if len(argvs) < 4:
        print _usage
        sys.exit(0)


    model = argvs[1]
    utt_file = argvs[2]
    res_file = argvs[3]

    main(model, utt_file, res_file)
