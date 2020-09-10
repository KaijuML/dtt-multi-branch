"""
This provides utilities to train models with RL
For now only HSS is supported since I have only coded `HSS`, but the framework 
coded in this file is generic and can be used for other metric with little change

This code is specific to WikiBIO for now.

We provide code compatible with our version of ONMT, meaning you should
check the shape of the returned tensors from your model to be sure
everything works correctly.

HSS == Halucination Scores Sum
"""

import sys

import torch

import onmt
from onmt.metrics.hss import HSS, HSA
from onmt.metrics.utils import WikiBIOCleaner
from onmt.utils.logging import logger
from onmt.utils.misc import sequence_mask


class RLLossCompute:
    def __init__(self, opt, tgt_field):
        self.cleaner = WikiBIOCleaner(tgt_field)

        # self.log("Initializing RL metric. Loading computed stats.")

        if opt.rl_metric in ["hss", "hsa"]:
            assert all(f is not None for f in [opt.co_occur_file, opt.tables_file]), \
                'Hallucination Score based RL metrics require to set co_occur_file and jsonl_file options!'
            self.metric = {"hss": HSS, "hsa": HSA}[opt.rl_metric](opt.co_occur_file, opt.tables_file)
        else:
            # raise error
            raise ValueError('No other metric than HSS and HSA are supported')

        with open(opt.ref_path, encoding="utf8", mode="r") as f:
            self.references = [ref.strip() for ref in f if ref.strip()]
                
        #self.log("PARENT metric Initialized.")
        
    def log(self, msg):
        logger.info(msg)
        sys.stdout.flush()
    
    def __call__(self, batch, rl_forward, baseline_forward):
        """
        There's no better way for now than a for-loop...
        """
        
        rl_sentences, rl_log_probs, rl_attns = rl_forward
        baseline_sentences, baseline_log_probs, baseline_attns = baseline_forward
        
        device = batch.tgt.device
        
        rl_lengths = list()
        baseline_lengths = list()
        decoded_sequences = list()
        rl_scores = list()
        baseline_scores = list()
        for b in range(batch.batch_size):
            
            rl_candidate, rl_length = self.cleaner.clean_candidate_tokens(rl_sentences[:, b],
                                                                    batch.src_map[:, b], 
                                                                    batch.src_ex_vocab[b],
                                                                    rl_attns[:, b])
            baseline_candidate, baseline_length = self.cleaner.clean_candidate_tokens(baseline_sentences[:, b],
                                                                    batch.src_map[:, b], 
                                                                    batch.src_ex_vocab[b],
                                                                    baseline_attns[:, b])
            
            if rl_length == 0:
                rl_length = 1
            rl_lengths.append(rl_length)
            baseline_lengths.append(baseline_length)
            decoded_sequences.append((self.references[batch.indices[b].item()],
                                      " ".join(baseline_candidate), " ".join(rl_candidate)))

            rl_scores.append(self.metric(rl_candidate, batch.indices[b].item()))
            baseline_scores.append(self.metric(baseline_candidate, batch.indices[b].item()))
            
        rl_lengths = torch.LongTensor(rl_lengths).to(device)
        baseline_lengths = torch.LongTensor(baseline_lengths).to(device)
        mask = sequence_mask(rl_lengths, max_len=len(rl_sentences))
        
        sequences_scores = rl_log_probs.masked_fill(~mask.transpose(0,1), 0)
        sequences_scores = sequences_scores.sum(dim=0) / rl_lengths.float()
        
        # we reward the model according to f1_score
        
        rl_rewards = torch.FloatTensor(rl_scores).to(device)
        baseline_rewards = torch.FloatTensor(baseline_scores).to(device)
        rewards = baseline_rewards - rl_rewards
    
        loss = (rewards * sequences_scores).mean()
        stats = self._stats(loss, baseline_rewards.mean(), rl_rewards.mean(),
                            baseline_lengths, rl_lengths,
                            decoded_sequences)
        
        return loss, stats
    
    def _stats(self, loss, baseline_rewards, rl_rewards, 
               baseline_lengths, rl_lengths, decoded_sequences):
        
        loss = loss.item()
        lengths = baseline_lengths.sum().item() + rl_lengths.sum().item()
        
        return onmt.rl_trainer.RLStatistics(rl_loss=loss, rl_rewards=rl_rewards.item(), 
                                            baseline_rewards=baseline_rewards.item(),
                                            n_tgt_words=lengths, 
                                            decoded_sequences=decoded_sequences)