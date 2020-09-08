import torch


class WikiBIOCleaner:
    """
    Base class for objects that handles model output and 
    returns source/target/gen as need for the metric.
    """
    def __init__(self, tgt_field):
        self.vocab = tgt_field.vocab
        self.eos_idx = self.vocab.stoi[tgt_field.eos_token]
        self.pad_idx = self.vocab.stoi[tgt_field.pad_token]
        self.bos_idx = self.vocab.stoi[tgt_field.init_token]
        self.unk_idx = self.vocab.stoi[tgt_field.unk_token]
        
    def clean_candidate_tokens(self, candidates, src_map, src_vocab, attns):
        """
        Builds the translation from model output.
        Replace <unk> by the input token with highest attn score
        """
        tokens = list()
        for idx, token in enumerate(candidates):
            token = token.item()
            if token == self.eos_idx:
                break

            if token == self.unk_idx and attns is not None:
                _, max_idx = attns[idx].max(0)
                max_idx = torch.nonzero(src_map[max_idx])
                clean_token = src_vocab.itos[max_idx.item()]        
            elif token < len(self.vocab):
                clean_token = self.vocab.itos[token] 
            else:
                clean_token = src_vocab.itos[token - len(self.vocab)]

            tokens.append(clean_token)
        return tokens, idx