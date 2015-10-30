"""Specifies a particular instance of an encoder-decoder model."""
from gru import GRULayer
from lstm import LSTMLayer
from outputlayer import OutputLayer
from spec import Spec
from vanillarnn import VanillaRNNLayer

class EncoderDecoderSpec(Spec):
  """Abstract class for a specification of an encoder-decoder model.
  
  Concrete subclasses must implement the following method:
  - self.create_rnn_layer(vocab, hidden_size): Create an RNN layer.
  """
  def create_vars(self):
    self.encoder = self.create_rnn_layer(self.in_vocabulary, self.hidden_size, True)
    self.decoder = self.create_rnn_layer(self.out_vocabulary, self.hidden_size, False)
    self.writer = self.create_output_layer(self.out_vocabulary, self.lexicon, self.hidden_size)

  def get_local_params(self):
    return self.encoder.params + self.decoder.params + self.writer.params

  def create_rnn_layer(self, vocab, hidden_size, is_encoder):
    raise NotImplementedError

  def create_output_layer(self, vocab, lexicon, hidden_size):
    return OutputLayer(vocab, lexicon, hidden_size)

  def get_init_state(self):
    return self.encoder.get_init_state()

  def f_enc(self, x_t, h_prev):
    """Returns the next hidden state for encoder."""
    return self.encoder.step(x_t, h_prev)

  def f_dec(self, y_t, h_prev):
    """Returns the next hidden state for decoder."""
    return self.decoder.step(y_t, h_prev)

  def f_write(self, h_t, cur_lex_entries):
    """Gives the softmax output distribution."""
    return self.writer.write(h_t, cur_lex_entries)

class VanillaEncDecSpec(EncoderDecoderSpec):
  """Encoder-decoder model with vanilla RNN recurrent units."""
  def create_rnn_layer(self, vocab, hidden_size, is_encoder):
    return VanillaRNNLayer(vocab, hidden_size,
                           create_init_state=is_encoder)

class GRUEncDecSpec(EncoderDecoderSpec):
  """Encoder-decoder model with GRU recurrent units."""
  def create_rnn_layer(self, vocab, hidden_size, is_encoder):
    return GRULayer(vocab, hidden_size,
                    create_init_state=is_encoder)

class LSTMEncDecSpec(EncoderDecoderSpec):
  """Encoder-decoder model with LSTM recurrent units."""
  def create_rnn_layer(self, vocab, hidden_size, is_encoder):
    return LSTMLayer(vocab, hidden_size,
                     create_init_state=is_encoder)
