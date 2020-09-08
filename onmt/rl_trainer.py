"""
    This is a copy of the loadable seq2seq trainer library from onmt.
    This implementation trains models using RL instead of teacher 
    forcing with the target. 
    
    I removed a lot of stuff: moving_average, validation and reporting.
    Also multi gpu. It's only to make life easier at first, it can be
    implemented again.
"""

from copy import deepcopy

import traceback
import random
import torch
import time
import math
import sys

import onmt.utils
from onmt.utils.logging import logger
from onmt.modules.copy_generator import collapse_copy_scores


class RLTrainer(object):
    """
    Class that controls the training process with RL on mPARENT metric.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            gamma_loss(float): loss is gamma_loss * RL + (1-gamma_loss) * ML
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, fields, ml_loss, rl_loss,
                 optim, gamma_loss=1,
                 trunc_size=0,
                 norm_method="sents", accum_count=[1],
                 accum_steps=[0], n_gpu=1, gpu_rank=1,
                 gpu_verbose_level=0, report_manager=None, model_saver=None,
                 average_decay=0, average_every=1, model_dtype='fp32',
                 earlystopper=None, dropout=[0.3], dropout_steps=[0]):
        # Basic attributes.
        self.model = model
        self.optim = optim
        self.trunc_size = trunc_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.dropout_steps = dropout_steps
        
        if n_gpu > 1:
            raise ValueError('No multi gpu for now')
        
        if isinstance(self.model.generator, onmt.modules.CopyGenerator):
            self.copy_attn = True
        else:
            self.copy_attn = False
        
        # for translation-inspired steps
        self.fields = fields
        tgt_field = self.fields["tgt"].base_field
        self._tgt_vocab = tgt_field.vocab
        self._tgt_eos_idx = self._tgt_vocab.stoi[tgt_field.eos_token]
        self._tgt_pad_idx = self._tgt_vocab.stoi[tgt_field.pad_token]
        self._tgt_bos_idx = self._tgt_vocab.stoi[tgt_field.init_token]
        self._tgt_unk_idx = self._tgt_vocab.stoi[tgt_field.unk_token]
        self._tgt_dot_idx = self._tgt_vocab.stoi['.']  # custom rule
        self._tgt_vocab_len = len(self._tgt_vocab)

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0 #
            if self.accum_count_l[i] > 1:
                assert self.trunc_size == 0, \
                    """To enable accumulated gradients,
                       you must disable target sequence truncating."""
        

        self.rl_loss = rl_loss
        self.ml_loss = ml_loss
        self.gamma_loss = gamma_loss

        # Set model in training mode.
        self.model.train()
        
    def log_msg(self, msg):
        logger.info(msg)
        sys.stdout.flush()

    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.model.update_dropout(self.dropout[i])
                logger.info("Updated dropout to %f from step %d"
                            % (self.dropout[i], step))

    def _accum_batches(self, iterator):
        batches = list()
        normalization = 0
        self.accum_count = self._accum_count(self.optim.training_step)
        for batch in iterator:
            batches.append(batch)
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:, :, 0].ne(
                    self.train_loss.padding_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size
            if len(batches) == self.accum_count:
                yield batches, normalization
                self.accum_count = self._accum_count(self.optim.training_step)
                batches = list()
                normalization = 0
        if batches:
            yield batches, normalization

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [params.detach().float()
                           for params in self.model.parameters()]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay,
                                1 - (step + 1)/(step + 10))
            for (i, avg), cpt in zip(enumerate(self.moving_average),
                                     self.model.parameters()):
                self.moving_average[i] = \
                    (1 - average_decay) * avg + \
                    cpt.detach().float() * average_decay

    def train(self,
              train_iter,
              train_steps,
              save_checkpoint_steps=5000,
              valid_iter=None,
              valid_steps=10000):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...',
                        valid_steps)

        total_stats = RLStatistics()
        report_stats = RLStatistics()
        self._start_report_manager(start_time=total_stats.start_time)

        for i, (batches, normalization) in enumerate(
                self._accum_batches(train_iter)):
            step = self.optim.training_step

            # UPDATE DROPOUT
            self._maybe_update_dropout(step)

            if self.gpu_verbose_level > 1:
                logger.info("GpuRank %d: index: %d", self.gpu_rank, i)
            if self.gpu_verbose_level > 0:
                logger.info("GpuRank %d: reduce_counter: %d \
                            n_minibatch %d"
                            % (self.gpu_rank, i + 1, len(batches)))

            self._gradient_accumulation(
                batches, normalization, total_stats,
                report_stats)

            report_stats = self._maybe_report_training(
                step, train_steps,
                self.optim.learning_rate(),
                report_stats)

            if (self.model_saver is not None
                and (save_checkpoint_steps != 0
                     and step % save_checkpoint_steps == 0)):
                self.model_saver.save(step, moving_average=self.moving_average)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        return total_stats

    def _gradient_accumulation(self, true_batches, normalization, total_stats,
                               report_stats):
        if self.accum_count > 1:
            self.optim.zero_grad()
        
        for k, batch in enumerate(true_batches):
            if self.accum_count == 1:
                self.optim.zero_grad()
            
            target_size = batch.tgt.size(0)

            src, src_lengths = batch.src
            device = src.device
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()
                report_stats.n_batches += 1
                
                
            # Encoder forward.
            enc_states, memory_bank, src_lengths = self.model.encoder(src, src_lengths)
            
            # Teacher forcing
            self.model.decoder.init_state(src, memory_bank, enc_states)
            ml_outputs, ml_attns = self.model.decoder(batch.tgt[:-1], memory_bank,
                                                  memory_lengths=src_lengths)
            
            
            # Sampling a path 
            rl_forward = self._forward_model(batch, enc_states, memory_bank, sample="reinforce")
            
            # baseline computing doesn't need gradient
            with torch.no_grad():
                baseline_forward = self._forward_model(batch, enc_states, memory_bank, sample="topk")

            # 3. Compute loss.
            try:
                rl_loss, rl_stats = self.rl_loss(
                    batch,
                    rl_forward,
                    baseline_forward)
                
                ml_loss, ml_stats = self.ml_loss(
                    batch,
                    ml_outputs,
                    ml_attns,
                    normalization=normalization,
                    shard_size=0,
                    trunc_start=0,
                    trunc_size=target_size)

                loss = rl_loss * self.gamma_loss + ml_loss * (1 - self.gamma_loss)
                self.optim.backward(loss)

                total_stats.update(rl_stats, ml_stats)
                report_stats.update(rl_stats, ml_stats)

            except Exception:
                traceback.print_exc()
                logger.info("At step %d, we removed a batch - accum %d",
                            self.optim.training_step, k)

            # 4. Update the parameters and statistics.
            if self.accum_count == 1:
                self.optim.step()      
                
                # report the grad norms by modules
                #GradNorm.output_norms(self.model)

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.accum_count > 1:
            self.optim.step()
            
            # report the grad norms by modules
            #GradNorm.output_norms(self.model)
            
    def _forward_model(self, batch, enc_states, memory_bank, sample="reinforce"):
        """
        Forward pass inspired by onmt.Translator.
        Replaces teacher forcing.
        """

        src, src_lengths = batch.src
        device = src.device
        
        # Again Decoder init. for new decoder forward
        self.model.decoder.init_state(src, memory_bank, enc_states)
        decoder_input = torch.full(
            [1, batch.batch_size], self._tgt_bos_idx,
            dtype=torch.long, device=device)

        # Where we store the words for each sentences
        sentences = list()

        # Where we store the sequence of log-probs for each chosen word
        log_probs = list()

        # Where we store attention scores
        attns = list()

        # Decoder forward loop. We decode words one step at a time
        # and feed them back for next word prediction.
        for step in range(batch.tgt.size(0) + 10):                
            scores, attn, attn_key = self._decode_and_generate(
                decoder_input.unsqueeze(-1),
                memory_bank,
                batch,
                memory_lengths=src_lengths,
                src_map=batch.src_map if self.copy_attn else None,
                step=step)
            
            
            distribution = torch.distributions.Categorical(probs=scores)
            
            if sample == "reinforce":
                # Sample from dist. for REINFORCE
                decoder_input = distribution.sample()
            else:
                _, decoder_input = scores.topk(1, dim=-1)
                decoder_input = decoder_input.view(1, -1)

            # Keeping track to compute rewards at the end
            sentences.append(decoder_input)
            log_probs.append(distribution.log_prob(decoder_input))
            attns.append(attn[attn_key])

        sentences = torch.cat(sentences, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        attns = torch.cat(attns, dim=0)

        return sentences, log_probs, attns
            
                
    def _decode_and_generate(
            self,
            decoder_in,
            memory_bank,
            batch,
            memory_lengths,
            src_map=None,
            step=None):
        if self.copy_attn:
            # Turn any copied words into UNKs.
            decoder_in = decoder_in.masked_fill(
                decoder_in.gt(self._tgt_vocab_len - 1), self._tgt_unk_idx
            )

        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        dec_out, dec_attn = self.model.decoder(
            decoder_in, memory_bank, memory_lengths=memory_lengths, step=step
        )

        # Generator forward.
        if not self.copy_attn:
            if "std" in dec_attn:
                attn_key = 'std'
            else:
                attn_key = None
            log_probs = self.model.generator(dec_out.squeeze(0))
            scores = log_probs.exp()  # unfortunate but torch.Categorical want softmax
        else:
            attn = dec_attn["copy"]
            attn_key= 'copy'
            scores = self.model.generator(dec_out.view(-1, dec_out.size(2)),
                                          attn.view(-1, attn.size(2)),
                                          src_map)

            scores = scores.view(batch.batch_size, -1, scores.size(-1))
            
            scores = collapse_copy_scores(
                scores,
                batch,
                self._tgt_vocab,
                src_vocabs=None,
                batch_dim=0,
                batch_offset=0
            )
            scores = scores.view(decoder_in.size(0), -1, scores.size(-1))
            
        return scores, dec_attn, attn_key

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)


        
class RLReportManager(object):
    """Report Manager class when doing RL"""

    def __init__(self, report_every, start_time=-1.):
        """
        Args:
            report_every(int): Report status every this many sentences
            start_time(float): manually set report start time. Negative values
                means that you will need to set it later or use `start()`
        """
        self.report_every = report_every
        self.start_time = start_time

    def start(self):
        self.start_time = time.time()

    def log(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    def report_training(self, step, num_steps, learning_rate,
                        report_stats, multigpu=False):
        """
        This is the user-defined batch-level traing progress
        report function.
        Args:
            step(int): current step count.
            num_steps(int): total number of batches.
            learning_rate(float): current learning rate.
            report_stats(Statistics): old Statistics instance.
        Returns:
            report_stats(Statistics): updated Statistics instance.
        """
        if self.start_time < 0:
            raise ValueError("""ReportMgr needs to be started
                                (set 'start_time' or use 'start()'""")

        if step % self.report_every == 0:
            report_stats.output(step, num_steps,
                            learning_rate, self.start_time)
            
            return RLStatistics()
        else:
            return report_stats
        

class RLStatistics(object):
    """
    Accumulator for metric statistics.
    Currently works for:
    * parent
    * bleu
    """
    
    LOSS_NAME = None

    def __init__(self, rl_loss=0, ml_loss=0, rl_rewards=0, baseline_rewards=0, 
                 n_tgt_words=0, n_words=0, n_correct=0, decoded_sequences=None):
        assert self.LOSS_NAME is not None, "Error in setting up LOSS_NAME"
        
        self.rl_loss = rl_loss
        self.ml_loss = ml_loss
        self.rl_rewards = rl_rewards
        self.baseline_rewards = baseline_rewards
        self.n_src_words = 0
        self.n_tgt_words = n_tgt_words  # for free decoding
        self.n_words = n_words  # for teacher forcing
        self.n_correct = n_correct  # for teacher forcing
        self.n_batches = 0
        self.start_time = time.time()
        self.decoded_sequences = decoded_sequences if decoded_sequences else list()

    def update(self, rl_stat, ml_stat):
        """
        Update statistics by suming values with another `Statistics` object
        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not
        """
        self.rl_loss += rl_stat.rl_loss
        self.ml_loss += ml_stat.loss
        self.rl_rewards += rl_stat.rl_rewards
        self.baseline_rewards += rl_stat.baseline_rewards
        self.n_batches += rl_stat.n_batches
        
        # rl
        self.n_tgt_words += rl_stat.n_tgt_words
        
        # teacher forcing
        self.n_words += ml_stat.n_words
        self.n_correct += ml_stat.n_correct
        
        # instance tracking
        self.decoded_sequences.extend(rl_stat.decoded_sequences)

    @property
    def loss(self):
        """ returns parent """
        return self.rl_loss / self.n_batches
    
    @property
    def accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct / self.n_words)

    @property
    def xent(self):
        """ compute cross entropy """
        return self.ml_loss / self.n_words

    @property
    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.loss / self.n_words, 100))

    @property
    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time
    
    @property
    def examples(self):
        return [random.choice(self.decoded_sequences) for _ in range(3)]

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.
        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time
        step_fmt = "%2d" % step
        if num_steps > 0:
            step_fmt = f"{step_fmt}/{num_steps}"
            
        sent = "Step {}; {}-loss: {:4.2f}; xent: {:4.2f}; "\
               "rewards: {:4.2f}/{:4.2f};  lr: {:7.5f}; {:3.0f}/{:3.0f} tok/s;"\
               " {:6.0f} sec"
            
        logger.info(
            sent.format(step_fmt,
                        self.LOSS_NAME,
                        self.loss,
                        self.xent,
                        self.rl_rewards / self.n_batches,
                        self.baseline_rewards / self.n_batches,
                        learning_rate,
                        self.n_src_words / (t + 1e-5),
                        (self.n_tgt_words + self.n_words) / (t + 1e-5),
                        time.time() - start)
        )
        sys.stdout.flush()
        
        examples = "\nPrinting 3 examples below\n"
        for ref, baseline, sample in self.examples:
            examples += f"ref     >>> {ref}\n"
            examples += f"sample  >>> {sample}\n"
            examples += f"baseline>>> {baseline}\n\n"
        
        logger.info(examples)
        sys.stdout.flush()
        
        
class GradNorm:
    
    @classmethod
    def output_norms(cls, model):
        norms = cls.grad_norm(model)
        output = ""
        for key, vals in norms.items():
            output += f"{key}: ["
            for v in vals:
                output += f"{v:.3f}/"
            output += ']; '
            
        logger.info(output.strip())
        sys.stdout.flush()
        
    @staticmethod
    def grad_norm(model):
        norms = dict()
        for name, tensor in model.named_parameters():
            name = name.split(".")
            name = name[0][:3]+"."+name[1]
            norms.setdefault(name, list())
            norms[name].append(tensor.grad.data.norm(2).item())
        return norms