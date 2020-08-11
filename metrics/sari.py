# =======================================================
#  SARI -- Text Simplification Tunable Evaluation Metric
# =======================================================
#
# Author: Wei Xu (UPenn xwe@cis.upenn.edu)
#
# A Python implementation of the SARI metric for text simplification
# evaluation in the following paper
#
#     "Optimizing Statistical Machine Translation for Text Simplification"
#     Wei Xu, Courtney Napoles, Ellie Pavlick, Quanze Chen and Chris Callison-Burch
#     In Transactions of the Association for Computational Linguistics (TACL) 2015
#
# There is also a Java implementation of the SARI metric
# that is integrated into the Joshua MT Decoder. It can
# be used for tuning Joshua models for a real end-to-end
# text simplification model.
#

from __future__ import division
from collections import Counter

import numpy as np


class SARISentenceMetric:

    def evaluate_results(self, sources, predictions, targets):
        """
        We design a new metric SARI that principally compares system output against references
        and against the input sentence. It explicitly measures the goodness of words that are added, deleted
        and kept by the systems

        If you use this project in your research, please cite Wei's awesome paper:

        @article{Xu-EtAl:2016:TACL,
            author = {Wei Xu and Courtney Napoles and Ellie Pavlick and Quanze Chen and Chris Callison-Burch},
            title = {Optimizing Statistical Machine Translation for Text Simplification},
            journal = {Transactions of the Association for Computational Linguistics},
            volume = {4},
            year = {2016},
            url = {https://cocoxu.github.io/publications/tacl2016-smt-simplification.pdf},
            pages = {401--415}
        }

        :param sources: sentences as array of dim (NxP) where N = Batch Size and P = Origin Sentence length
        :param predictions: sentences as array of dim (NxP) where N = Batch Size and P = Predicted Sentence length
        :param targets: sentences as arrays of dim (NxT) where N = Batch Size and T = Target Sentence length
        :return:
        """

        sari_scores = []
        for i in range(len(sources)):
            sari_scores.append(self.sari_sent(sources[i], predictions[i], [targets[i]]))

        mean_score = np.mean(np.asarray(sari_scores))

        return mean_score

    @staticmethod
    def sari_ngram(sgrams, cgrams, rgramslist, numref):
        rgramsall = [rgram for rgrams in rgramslist for rgram in rgrams]
        rgramcounter = Counter(rgramsall)

        sgramcounter = Counter(sgrams)
        sgramcounter_rep = Counter()
        for sgram, scount in sgramcounter.items():
            sgramcounter_rep[sgram] = scount * numref

        cgramcounter = Counter(cgrams)
        cgramcounter_rep = Counter()
        for cgram, ccount in cgramcounter.items():
            cgramcounter_rep[cgram] = ccount * numref

        # KEEP
        keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
        keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
        keepgramcounterall_rep = sgramcounter_rep & rgramcounter

        keeptmpscore1 = 0
        keeptmpscore2 = 0
        for keepgram in keepgramcountergood_rep:
            keeptmpscore1 += keepgramcountergood_rep[keepgram] / keepgramcounter_rep[keepgram]
            keeptmpscore2 += keepgramcountergood_rep[keepgram] / keepgramcounterall_rep[keepgram]
            # print "KEEP", keepgram, keepscore, cgramcounter[keepgram], sgramcounter[keepgram], rgramcounter[keepgram]
        keepscore_precision = 0
        if len(keepgramcounter_rep) > 0:
            keepscore_precision = keeptmpscore1 / len(keepgramcounter_rep)
        keepscore_recall = 0
        if len(keepgramcounterall_rep) > 0:
            keepscore_recall = keeptmpscore2 / len(keepgramcounterall_rep)
        keepscore = 0
        if keepscore_precision > 0 or keepscore_recall > 0:
            keepscore = 2 * keepscore_precision * keepscore_recall / (keepscore_precision + keepscore_recall)

        # DELETION
        delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
        delgramcountergood_rep = delgramcounter_rep - rgramcounter
        delgramcounterall_rep = sgramcounter_rep - rgramcounter
        deltmpscore1 = 0
        deltmpscore2 = 0
        for delgram in delgramcountergood_rep:
            deltmpscore1 += delgramcountergood_rep[delgram] / delgramcounter_rep[delgram]
            deltmpscore2 += delgramcountergood_rep[delgram] / delgramcounterall_rep[delgram]
        delscore_precision = 0
        if len(delgramcounter_rep) > 0:
            delscore_precision = deltmpscore1 / len(delgramcounter_rep)
        delscore_recall = 0
        if len(delgramcounterall_rep) > 0:
            delscore_recall = deltmpscore1 / len(delgramcounterall_rep)
        delscore = 0
        if delscore_precision > 0 or delscore_recall > 0:
            delscore = 2 * delscore_precision * delscore_recall / (delscore_precision + delscore_recall)

        # ADDITION
        addgramcounter = set(cgramcounter) - set(sgramcounter)
        addgramcountergood = set(addgramcounter) & set(rgramcounter)
        addgramcounterall = set(rgramcounter) - set(sgramcounter)

        addtmpscore = 0
        for _ in addgramcountergood:
            addtmpscore += 1

        addscore_precision = 0
        addscore_recall = 0
        if len(addgramcounter) > 0:
            addscore_precision = addtmpscore / len(addgramcounter)
        if len(addgramcounterall) > 0:
            addscore_recall = addtmpscore / len(addgramcounterall)
        addscore = 0
        if addscore_precision > 0 or addscore_recall > 0:
            addscore = 2 * addscore_precision * addscore_recall / (addscore_precision + addscore_recall)

        return keepscore, delscore, addscore

    def sari_sent(self, source_sent, candidate_sent, reference_sents):
        numref = len(reference_sents)

        s1grams = source_sent.lower().split(" ")
        c1grams = candidate_sent.lower().split(" ")
        s2grams = []
        c2grams = []
        s3grams = []
        c3grams = []
        s4grams = []
        c4grams = []

        r1gramslist = []
        r2gramslist = []
        r3gramslist = []
        r4gramslist = []
        for rsent in reference_sents:
            r1grams = rsent.lower().split(" ")
            r2grams = []
            r3grams = []
            r4grams = []
            r1gramslist.append(r1grams)
            for i in range(0, len(r1grams) - 1):
                if i < len(r1grams) - 1:
                    r2gram = r1grams[i] + " " + r1grams[i + 1]
                    r2grams.append(r2gram)
                if i < len(r1grams) - 2:
                    r3gram = r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2]
                    r3grams.append(r3gram)
                if i < len(r1grams) - 3:
                    r4gram = r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2] + " " + r1grams[i + 3]
                    r4grams.append(r4gram)
            r2gramslist.append(r2grams)
            r3gramslist.append(r3grams)
            r4gramslist.append(r4grams)

        for i in range(0, len(s1grams) - 1):
            if i < len(s1grams) - 1:
                s2gram = s1grams[i] + " " + s1grams[i + 1]
                s2grams.append(s2gram)
            if i < len(s1grams) - 2:
                s3gram = s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2]
                s3grams.append(s3gram)
            if i < len(s1grams) - 3:
                s4gram = s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2] + " " + s1grams[i + 3]
                s4grams.append(s4gram)

        for i in range(0, len(c1grams) - 1):
            if i < len(c1grams) - 1:
                c2gram = c1grams[i] + " " + c1grams[i + 1]
                c2grams.append(c2gram)
            if i < len(c1grams) - 2:
                c3gram = c1grams[i] + " " + c1grams[i + 1] + " " + c1grams[i + 2]
                c3grams.append(c3gram)
            if i < len(c1grams) - 3:
                c4gram = c1grams[i] + " " + c1grams[i + 1] + " " + c1grams[i + 2] + " " + c1grams[i + 3]
                c4grams.append(c4gram)

        (keep1score, del1score, add1score) = self.sari_ngram(s1grams, c1grams, r1gramslist, numref)
        (keep2score, del2score, add2score) = self.sari_ngram(s2grams, c2grams, r2gramslist, numref)
        (keep3score, del3score, add3score) = self.sari_ngram(s3grams, c3grams, r3gramslist, numref)
        (keep4score, del4score, add4score) = self.sari_ngram(s4grams, c4grams, r4gramslist, numref)

        avgkeepscore = sum([keep1score, keep2score, keep3score, keep4score]) / 4
        avgdelscore = sum([del1score, del2score, del3score, del4score]) / 4
        avgaddscore = sum([add1score, add2score, add3score, add4score]) / 4
        finalscore = (avgkeepscore + avgdelscore + avgaddscore) / 3

        return finalscore
