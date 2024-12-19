import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)


        _, seq_len, batch_size = y_probs.shape

        decoded_paths = [[] for _ in range(batch_size)]
        path_probs = np.ones(batch_size) 

        for t in range(seq_len):
            max_pos = np.argmax(y_probs[:, t, :], axis=0) 
            max_prob = y_probs[max_pos, t, np.arange(batch_size)] 
            path_probs *= max_prob 

            for b in range(batch_size):
                if max_pos[b] == 0:
                    decoded_paths[b].append('SEP')  
                else:
                    decoded_paths[b].append(self.symbol_set[max_pos[b] - 1])  

        compressed_outputs = []
        for b in range(batch_size):
            compressed_path = [decoded_paths[b][0]]  
            for i in range(1, len(decoded_paths[b])):
                if decoded_paths[b][i] != decoded_paths[b][i - 1] and decoded_paths[b][i] != 'SEP':
                    compressed_path.append(decoded_paths[b][i])
            compressed_output = "".join(compressed_path)
            compressed_outputs.append(compressed_output)
        
        return compressed_outputs[0], path_probs[0]


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        BLANK_SYMBOL = "-"
        bestPath, FinalPathScore = None, None

        #dim=(len(symbols) + 1, seq_length, batch_size)
        bestPaths = {BLANK_SYMBOL:1}
        for t in range(T):
            
            tempBestPaths = {}

            probab = y_probs[:, t, 0]

            for seq, seq_prob in bestPaths.items():

                for p in range(len(probab)):

                    if p == 0: 
                        sym = BLANK_SYMBOL
                    else:
                        sym = self.symbol_set[p - 1]

                
                    if seq[-1] == sym:
                        new_seq = seq
                    elif seq[-1] == BLANK_SYMBOL:
                        new_seq = seq[:-1] + sym
                    else:
                        new_seq = seq + sym
                       
                    
                    tempBestPaths[new_seq] = tempBestPaths.get(new_seq, 0) + seq_prob * probab[p]
            
            sortedbestPaths = sorted(tempBestPaths.items(), key=lambda item: item[1], reverse=True)
            bestPaths = dict(sortedbestPaths[:self.beam_width])

        final_scores = {}

        for seq, prob in dict(sortedbestPaths).items():
            if seq[-1] == BLANK_SYMBOL:
                seq = seq[:-1]

            final_scores[seq] = final_scores.get(seq, 0) + prob

        best_seq = max(final_scores, key=final_scores.get)

        return best_seq, final_scores

            
