from queue import PriorityQueue
import operator

import torch

## https://github.com/MaximumEntropy/Seq2Seq-PyTorch/blob/master/beam_search.py

class BeamSearchNode(object):
    def __init__(self, previousNode, wordId, logProb, length):
        """
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        """
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


class BeamSearch(object):
    def __init__(self, sos_token, eos_token, beam_size=10, number_output_candidates=1):
        self.sos_token = sos_token
        self.eos_token = eos_token

        self.beam_size = beam_size
        self.top_k_output_sents = number_output_candidates

    def beam_decode(self, target_tensor, decoder, encoder_outputs=None, device="cpu"):
        """
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder: decoder to predict next word
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :param device: cpu or gpu:x
        :return: decoded_batch
        """
        decoded_batch = []

        # decoding goes sentence by sentence
        for idx in range(target_tensor.size(0)):
            memory = encoder_outputs[:, idx, :].unsqueeze(1)

            # Start with the start of the sentence token
            decoder_input = torch.LongTensor([[self.sos_token]], device=device)

            # Number of sentence to generate
            endnodes = []
            number_required = min((self.top_k_output_sents + 1), self.top_k_output_sents - len(endnodes))

            # starting node -  previous node, word id, logp, length
            node = BeamSearchNode(None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000:
                    break

                # fetch the best node
                score, n = nodes.get()
                decoder_input = n.wordid

                if n.wordid.item() == self.eos_token and n.prevNode is not None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                decoder_output = decoder(decoder_input, memory)

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoder_output, self.beam_size)
                nextnodes = []

                for new_k in range(self.beam_size):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(self.top_k_output_sents)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = [n.wordid]
                # back trace
                while n.prevNode is not None:
                    n = n.prevNode
                    utterance.append(n.wordid)

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.append(utterances)

        return decoded_batch


class GreedySearch(object):
    def __init__(self, sos_token, eos_token, max_length=512):
        self.sos_token = sos_token
        self.eos_token = eos_token

        self.max_length = max_length

    def greedy_decode(self, target_tensor, decoder, encoder_outputs=None, device="cpu"):
        """
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder: decoder to predict next word
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :param device: cpu or gpu:x
        :return: decoded_batch
        """

        batch_size, seq_len = target_tensor.size()
        decoded_batch = torch.zeros((batch_size, self.max_length))
        decoder_input = torch.LongTensor([[self.sos_token] for _ in range(batch_size)], device=device)

        memory = encoder_outputs

        for t in range(self.max_length):
            decoder_output = decoder(decoder_input, memory)

            top_v, top_i = torch.argmax(decoder_output, dim=1)
            top_i = top_i.view(-1)
            decoded_batch[:, t] = top_i

            decoder_input = top_i.detach().view(-1, 1)

        return decoded_batch
