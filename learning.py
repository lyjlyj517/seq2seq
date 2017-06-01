#!/usr/bin/env python                                                                                                                                                    
# -*- coding: utf-8 -*-                                                                                                                                                  

import sys
import numpy as np
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
        for i in range(len(eline)):
            wid = evocab[eline[i]]
            x_k = self.embedy(Variable(np.array([wid], dtype=np.int32)))
            next_wid = evocab['<eos>'] if (i == len(eline) - 1) else evocab[eline[i+1]]
            tx = Variable(np.array([next_wid], dtype=np.int32))
            h = self.H(x_k)
            loss = F.softmax_cross_entropy(self.W(h), tx)
            accum_loss += loss

        return accum_loss

def main(epochs, urr_file, res_file, out_path):

    jvocab = {}
    jlines = open(utt_file).read().split('\n')
    for i in range(len(jlines)):
        lt = jlines[i].split()
        for w in lt:
            if w not in jvocab:
                jvocab[w] = len(jvocab)

    jvocab['<eos>'] = len(jvocab)
    jv = len(jvocab)

    evocab = {}
    elines = open(res_file).read().split('\n')
    for i in range(len(elines)):
        lt = elines[i].split()
        for w in lt:
            if w not in evocab:
                evocab[w] = len(evocab)
                ev = len(evocab)

        demb = 100
        model = seq2seq(jv, ev, demb, jvocab, evocab)
        optimizer = optimizers.Adam()
        optimizer.setup(model)

        for epoch in range(epochs):
            for i in range(len(jlines)-1):
                jln = jlines[i].split()
                jlnr = jln[::-1]
                eln = elines[i].split()
                model.H.reset_state()
                model.zerograds()
                loss = model(jlnr, eln, jvocab, evocab)
                loss.backward()
                loss.unchain_backward()
                optimizer.update()
                print i, " finished"        

            outfile = out_path + "/seq2seq-" + str(epoch) + ".model"
            serializers.save_npz(outfile, model)



if __name__ == "__main__":

    argvs = sys.argv

    _usage = """--                                                                                                                                                       
Usage:                                                                                                                                                                   
    python learning.py [epoch] [utteranceDB] [responseDB] [save_link]                                                                                                    
Args:                                                                                                                                                                    
    [epoch]: The argument is the number of max epochs to train models.                                                                                                   
    [utteranceDB]: The argument is input file to train model that is to convert as pre-utterance.                                                                        
    [responseDB]: The argument is input file to train model that is to convert as response to utterance.                                                                 
    [save_link]: The argument is output directory to save trained models.                                                                                                
""".rstrip()

    if len(argvs) < 5:
        print _usage
        sys.exit(0)


    epochs = int(argvs[1])
    utt_file = argvs[2]
    res_file = argvs[3]
    out_path = argvs[4]

    main(epochs, utt_file, res_file, out_path)

