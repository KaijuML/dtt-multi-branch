"""Define RNN-based encoders."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn import Parameter

from onmt.encoders.encoder import EncoderBase

class StructureAwareLSTM(torch.nn.Module):
    """
    Imitates the torch.nn.LSTM
    Really less efficident, but we can add gates easily
    Doesnt support multi layers
    """
    
    def __init__(self, input_size, field_size, hidden_size, bidirectional):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.field_size = field_size
        self.bidirectional = bidirectional
        
        num_directions = 2 if self.bidirectional else 1
        
        for direction in range(num_directions):
            w_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
            w_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
            w_fh = Parameter(torch.Tensor(2 * hidden_size, field_size))
            b_ih = Parameter(torch.Tensor(4 * hidden_size))
            b_hh = Parameter(torch.Tensor(4 * hidden_size))
            b_fh = Parameter(torch.Tensor(2 * hidden_size))
            layer_params = (w_ih, w_hh, w_fh, b_ih, b_hh, b_fh)

            suffix = '_reverse' if direction == 1 else ''
            param_names = ['weight_ih{}', 'weight_hh{}', 'weight_fh{}',
                           'bias_ih{}', 'bias_hh{}', 'bias_fh{}']
            param_names = [x.format(suffix) for x in param_names]

            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)
                
        self.init_weights()
        
    def extra_repr(self):
        s = '{input_size}-{field_size}, {hidden_size}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)
                
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                torch.nn.init.xavier_uniform_(p.data)
            else:
                torch.nn.init.ones_(p.data)
              
    @staticmethod
    def _cell(input, field, hidden, w_ih, w_hh, w_fh, b_ih, b_hh, b_fh):
        """Classic LSTM op."""
        hx, cx = hidden
        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
        
        fieldgates = F.linear(field, w_fh, b_fh)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        fieldgate, fieldvalue = fieldgates.chunk(2, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        fieldgate = torch.sigmoid(fieldgate)
        fieldvalue = torch.sigmoid(fieldvalue)

        cy = (forgetgate * cx) + (ingate * cellgate) + (fieldgate * fieldvalue)
        hy = outgate * torch.tanh(cy)

        return hy, cy
    
    def _check_inputs(self, inputs, fields):
    
        assert inputs.shape[:2] == fields.shape[:2]
        assert inputs.size(2) == self.input_size
        assert fields.size(2) == self.field_size
    
    def forward(self, inputs, fields, hidden=None):
        
        seq_len, batch_size, input_dim = inputs.shape
        
        # >>> initiate empty outputs, (hn, cn)
        num_directions = 2 if self.bidirectional else 1
        output = inputs.new_zeros(seq_len, batch_size, num_directions * self.hidden_size)
        hn = inputs.new_zeros(num_directions, batch_size, self.hidden_size)
        cn = inputs.new_zeros(num_directions, batch_size, self.hidden_size)
        
        # >>> left to right >>>
        if hidden is None:
            hx = inputs.new_zeros(batch_size, self.hidden_size,
                                 requires_grad=False)
            hx_left = (hx, hx)
        else:
            hx_left = hidden[0][0], hidden[1][0]
            
        for idx, (input, field) in enumerate(zip(inputs.split(1), fields.split(1))):
            hx_left = self._cell(input.squeeze(0), field.squeeze(0), hx_left,
                                 self.weight_ih, self.weight_hh, self.weight_fh,
                                 self.bias_ih, self.bias_hh, self.bias_fh,)
            output[idx, :, :self.hidden_size] = hx_left[0]

        hn[0] = hx_left[0]
        cn[0] = hx_left[1]
            
        # >>> right to left >>>
        if self.bidirectional:
            if hidden is None:
                hx = inputs.new_zeros(batch_size, self.hidden_size,
                                     requires_grad=False)
                hx_right = (hx, hx)
            else:
                hx_right = hidden[0][1], hidden[1][1]
            
            hiddens_right = list()
            for idx, (input, field) in enumerate(zip(inputs.flip(0).split(1), fields.flip(0).split(1))):
                hx_right = self._cell(input.squeeze(0), field.squeeze(0), hx_right,
                                      self.weight_ih_reverse, self.weight_hh_reverse, self.weight_fh_reverse,
                                      self.bias_ih_reverse, self.bias_hh_reverse, self.bias_fh_reverse)
                hiddens_right.append(hx_right)
                output[seq_len - idx - 1, :, self.hidden_size:] = hx_right[0]
                
            hn[1] = hx_right[0]
            cn[1] = hx_right[1]
          
        return output, (hn, cn)
    

class StructureAwareEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (onmt.modules.Embeddings): embedding module to use
    """

    def __init__(self, bidirectional, hidden_size, embeddings=None,
                 use_bridge=False):
        super().__init__()
        assert embeddings is not None
        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        self.total_hidden_dim = hidden_size // num_directions
        hidden_size = hidden_size // num_directions
        
        
        self.embeddings = embeddings

        self.rnn = StructureAwareLSTM(
                        input_size=embeddings.emb_luts[0].embedding_dim,
                        field_size=embeddings.emb_luts[1].embedding_dim,
                        hidden_size=hidden_size,
                        bidirectional=bidirectional)
        
#         self.rnn, self.no_pack_padded_seq = 
#             rnn_factory(rnn_type,
#                         input_size=embeddings.embedding_size,
#                         hidden_size=hidden_size,
#                         num_layers=num_layers,
#                         dropout=dropout,
#                         bidirectional=bidirectional)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self.bridge = nn.ModuleList(
                [nn.Linear(hidden_size, hidden_size, bias=True)
                 for _ in range(2)])
    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.brnn,
            opt.enc_rnn_size,
            embeddings,
            opt.bridge)
    
    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        # we embed separatly all parts of src
        emb_luts = self.embeddings.make_embedding.emb_luts
        inputs_ = [feat.squeeze(2) for feat in src.split(1, dim=2)]
        inputs = [f(x) for f, x in zip(emb_luts, inputs_)]
        
        # we extract the field-names embedding
        fields = inputs[1]
        
        # we concat + mlp all embeddings
        inputs = self.embeddings.make_embedding.mlp(torch.cat(inputs, 2))
        
        memory_bank, encoder_final = self.rnn(inputs, fields)
        
#         packed_emb = emb
#         if lengths is not None and not self.no_pack_padded_seq:
#             # Lengths data is wrapped inside a Tensor.
#             lengths_list = lengths.view(-1).tolist()
#             packed_emb = pack(emb, lengths_list)

#         memory_bank, encoder_final = self.rnn(packed_emb)

#         if lengths is not None and not self.no_pack_padded_seq:
#             memory_bank = unpack(memory_bank)[0]

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)
        return encoder_final, (memory_bank, fields), lengths

    def _bridge(self, hidden):
        """Forward hidden state through bridge."""
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs

    def update_dropout(self, dropout):
        self.rnn.dropout = dropout
