from .sparse_activations import sparsemax
from onmt.utils.misc import aeq, sequence_mask

import torch
import onmt


class AttentionScorer(torch.nn.Module):
    """
    dim_query is dim of the decoder
    dim_key is dim of the encoder output
    """
    def __init__(self, dim, attn_type):
        super().__init__()
        if isinstance(dim, int):
            dim = dim, dim
        if not isinstance(dim, tuple):
            raise ValueError('dim should a one or two ints')
            
        assert len(dim) == 2
        assert isinstance(dim[0], int)
        assert isinstance(dim[1], int)
        assert attn_type != 'dot'
        self.dim_query, self.dim_context = dim
            
        self.attn_type = attn_type
        
        if self.attn_type == "general":
            self.linear_in = torch.nn.Linear(self.dim_context,
                                             self.dim_query,
                                             bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = torch.nn.Linear(self.dim_context,
                                                  self.dim_query,
                                                  bias=False)
            self.linear_query = torch.nn.Linear(self.dim_query,
                                                self.dim_query,
                                                bias=True)
            self.v = torch.nn.Linear(self.dim_query, 1, bias=False)
        
    def forward(self, context, queries):
        """
        Args:
          context (FloatTensor): context to be scored ``(batch, ctx_len, ctx_dim)``
          queries (FloatTensor): sequence of queries ``(batch, qrs_len, qrs_dim)``
        Returns:
          FloatTensor: raw attention scores (unnormalized) for each query index
            ``(batch, ctx_len, qrs_len)``
        """

        # Check input sizes
        ctx_batch, ctx_len, ctx_dim = context.size()
        qrs_batch, qrs_len, qrs_dim = queries.size()
        
        aeq(qrs_batch, ctx_batch)
        aeq(qrs_dim, self.dim_query)
        aeq(ctx_dim, self.dim_context)
        
        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                context_ = context.view(ctx_batch * ctx_len, ctx_dim)
                context_ = self.linear_in(context_)
                context = context_.view(ctx_batch, ctx_len, qrs_dim)
            # queries_ = queries.transpose(1, 2)
            # (batch, ctx_len, qrs_dim) x (batch, qrs_dim, qrs_len) --> (batch, ctx_len, qrs_len)
            # return torch.bmm(context, queries_)

            # (batch, qrs_len, qrs_dim) x (batch, qrs_dim, ctx_len)  --> (batch, qrs_len, ctx_len)
            return torch.bmm(queries, context.transpose(1, 2))
        else:
            wq = self.linear_context(context.view(-1, ctx_dim))
            wq = wq.view(ctx_batch, ctx_len, 1, qrs_dim)
            wq = wq.expand(ctx_batch, ctx_len, qrs_len, qrs_dim)

            uh = self.linear_query(queries.view(-1, qrs_dim))
            uh = uh.view(qrs_batch, 1, qrs_len, qrs_dim)
            uh = uh.expand(qrs_batch, ctx_len, qrs_len, qrs_dim)

            # (batch, ctx_len, qrs_len, d)
            wquh = torch.tanh(wq + uh)

        return self.v(wquh.view(-1, qrs_dim)).view(ctx_batch, qrs_len, ctx_len)
    
    

class DualAttention(torch.nn.Module):
    def __init__(self, dim, attn_type="dot", attn_func="softmax"):
        super().__init__()

        # dims shenanigans. memory_bank should be dim[0]
        if isinstance(dim, int):
            dim = dim, dim
        elif not isinstance(dim, tuple):
            raise ValueError('dim should be one or two ints')
            
            
        assert len(dim) == 2
        assert isinstance(dim[0], int)
        assert isinstance(dim[1], int)
        self.main_dim, self.pos_dim = dim
        
        if attn_func == 'softmax':
            self.attn_func = torch.nn.functional.softmax
        elif attn_func == 'sparsemax':
            self.attn_func = sparsemax
        else:
            raise ValueError("Please select a valid attention function.")
            
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type (got {:s}).".format(
                attn_type))
        self.attn_type = attn_type
        
        self.main_scorer = AttentionScorer(self.main_dim, attn_type)
        self.pos_scorer = AttentionScorer((self.main_dim, self.pos_dim),
                                          attn_type)
        
        # mlp wants it with bias, others no
        self.linear_out = torch.nn.Linear(self.main_dim * 2, 
                                          self.main_dim,
                                          bias=(attn_type=="mlp"))
        
        
    def forward(self, source, memory_bank, memory_lengths=None):
        # source is used to query memory_bank
        
        # assert one step input
        assert source.dim() == 2
        
        source = source.unsqueeze(1)
        
        # we unpack memory_bank which also contains fieldname embeddigns
        main_embs, pos_embs =  memory_bank
        main_embs = main_embs.transpose(0, 1)
        pos_embs = pos_embs.transpose(0, 1)
        
        # Checks and balances
        aeq(main_embs.size(0), source.size(0))  # should be same batch size
        aeq(main_embs.size(2), source.size(2))  # should be same dimension
        aeq(self.main_dim, main_embs.size(2))   # self-explanatory
        aeq(self.pos_dim, pos_embs.size(2))     # self-explanatory
        
        # we score both the encoder states and the fieldname embeddings
        main_align = self.main_scorer(main_embs.contiguous(), source)
        pos_align = self.pos_scorer(pos_embs.contiguous(), source)
        
        # we mask <blank> tokens
        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=main_align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            
            main_align.masked_fill_(~mask, -float('inf'))
            pos_align.masked_fill_(~mask, -float('inf'))
        
        # reshape + softmax/sparsemax layer
        batch_size, qrs_len, _ = source.size()
        _, ctx_len, ctx_dim = main_embs.size()
        main_align.view(batch_size * qrs_len, ctx_len)
        
        main_align = self.attn_func(main_align, dim=-1)
        pos_align = self.attn_func(pos_align, dim=-1)
        
        # final attention scores are termwise multiplications
        # we also normalize the scores so they sum to one
        align_vectors = main_align * pos_align
        align_vectors = align_vectors / align_vectors.sum(-1).unsqueeze(-1)
        
        # The context vector c_t to be fed to the generator 
        # is the weighted average over all the encoder hidden states
        c_t = torch.bmm(align_vectors, main_embs)

        # concatenate
        concat_c = torch.cat([c_t, source], 2).view(batch_size * qrs_len, ctx_dim*2)
        attn_h = self.linear_out(concat_c).view(batch_size, qrs_len, ctx_dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = torch.tanh(attn_h)


        attn_h = attn_h.squeeze(1)
        align_vectors = align_vectors.squeeze(1)
    
        ret = {
            '': align_vectors,
            '_main_align': main_align.squeeze(1),
            '_pos_align':pos_align.squeeze(1)
        }

        return attn_h, ret