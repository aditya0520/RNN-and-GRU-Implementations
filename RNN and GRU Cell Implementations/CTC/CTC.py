import numpy as np


class CTC(object):

    def __init__(self, BLANK=0):
        """
        
        Initialize instance variables

        Argument(s)
        -----------
        
        BLANK (int, optional): blank label index. Default 0.

        """

        # No need to modify
        self.BLANK = BLANK


    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """
        extended_symbols = [self.BLANK]
        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)
        
                  

        N = len(extended_symbols)

        # -------------------------------------------->
        # TODO

        skip_connect = [0] * (N)
        for i in reversed(range(1, N-2, 2)):

            if extended_symbols[i] != extended_symbols[i+2]:
                skip_connect[i+2] = 1
    
        # <---------------------------------------------
        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))
        return extended_symbols, skip_connect


    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO: Intialize alpha[0][0]
        # TODO: Intialize alpha[0][1]
        # TODO: Compute all values for alpha[t][sym] where 1 <= t < T and 1 <= sym < S (assuming zero-indexing)
        # IMP: Remember to check for skipConnect when calculating alpha
        # <---------------------------------------------

        alpha[0][0] = logits[0][extended_symbols[0]]
        alpha[0][1] = logits[0][extended_symbols[1]]


        for t in range(1, T):

            for r in range(S):

                alpha[t][r] = alpha[t-1][r]

                if r > 0:
                    alpha[t][r] += alpha[t-1][r-1]

                if skip_connect[r] == 1 and r > 1:

                    alpha[t][r] += alpha[t-1][r-2]
            
                alpha[t][r] *= logits[t][extended_symbols[r]]

        return alpha


    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities
        
        """

        S, T = len(extended_symbols), len(logits)
        beta = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO
        # <--------------------------------------------

        beta[T-1][S-1] = logits[T-1][extended_symbols[S-1]]
        beta[T-1][S-2] = logits[T-1][extended_symbols[S-2]]

        for t in range(T - 2, -1, -1):
            beta[t,S-1] = beta[t+1,S-1] * logits[t,extended_symbols[S-1]]
            for r in range(S - 2, -1, -1):

                beta[t][r] = beta[t+1][r] + beta[t+1][r+1]
                
                if r + 2 < S and skip_connect[r+2] == 1:
                    beta[t][r] += beta[t+1][r+2]
                
                beta[t][r] *= logits[t][extended_symbols[r]]
        
        for t in range(T):

            for r in range(S):

                beta[t][r] /= logits[t][extended_symbols[r]]

        return beta
        

    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """

        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros((T,))

        # -------------------------------------------->
        # TODO
        # <---------------------------------------------
        sumgamma = (alpha * beta).sum(axis=1, keepdims=True)
        gamma = (alpha * beta) / sumgamma
        # return gamma
        return gamma


class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.
        
        """
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

        Computes the CTC Loss by calculating forward, backward, and
        posterior proabilites, and then calculating the avg. loss between
        targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #####  IMP:
        #####  Output losses should be the mean loss over the batch

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # <---------------------------------------------

            trunc_target = self.target[batch_itr, :self.target_lengths[batch_itr]]
            trunc_logits = self.logits[:self.input_lengths[batch_itr], batch_itr, :]

            extended_symbols, skip_connect = self.ctc.extend_target_with_blank(trunc_target)

            batch_alpha = self.ctc.get_forward_probs(trunc_logits, extended_symbols, skip_connect)
            batch_beta = self.ctc.get_backward_probs(trunc_logits, extended_symbols, skip_connect)
            posterior = self.ctc.get_posterior_probs(batch_alpha, batch_beta)

            self.gammas.append(posterior)
            T,_ = posterior.shape
            posterior_times_log = posterior * np.log(trunc_logits[np.arange(T)[:, None], extended_symbols])
            total_loss[batch_itr] = -np.sum(posterior_times_log)
            

        total_loss = np.sum(total_loss) / B

        return total_loss
        

    def backward(self):
        """
        
        CTC loss backard

        Calculate the gradients w.r.t the parameters and return the derivative 
        w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """

        # No need to modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute derivative of divergence and store them in dY
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # <---------------------------------------------
            trunc_target = self.target[batch_itr, :self.target_lengths[batch_itr]]
            trunc_logits = self.logits[:self.input_lengths[batch_itr], batch_itr, :]

            extended_symbols, _ = self.ctc.extend_target_with_blank(trunc_target)
            
            # dim=(seq_length, batch_size, len(extended_symbols))
            T,N = self.gammas[batch_itr].shape
            for t in range(T):
                for n in range(N):
                    dY[t, batch_itr, extended_symbols[n]] -= self.gammas[batch_itr][t, n] / trunc_logits[t, extended_symbols[n]]

        return dY
        # raise NotImplementedError
