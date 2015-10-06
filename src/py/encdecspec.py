"""Specifies a particular instance of an encoder-decoder model."""
from gru import GRULayer
from lstm import LSTMLayer
from spec import Spec
from vanillarnn import VanillaRNNLayer

class EncoderDecoderSpec(Spec):
  """Abstract class for a specification of an encoder-decoder model.
  
  Concrete subclasses must implement the following method:
  - self.create_layer(vocab, hidden_size): Create an RNN layer.
  """
  def create_vars(self):
    self.encoder = self.create_layer(self.in_vocabulary, self.hidden_size, True)
    self.decoder = self.create_layer(self.out_vocabulary, self.hidden_size, False)

  def get_local_params(self):
    return self.encoder.params + self.decoder.params

  def create_layer(self, vocab, hidden_size, is_encoder):
    raise NotImplementedError

  def get_init_state(self):
    return self.encoder.get_init_state()

  def f_enc(self, x_t, h_prev):
    """Returns the next hidden state for encoder."""
    return self.encoder.step(x_t, h_prev)

  def f_dec(self, y_t, h_prev):
    """Returns the next hidden state for decoder."""
    return self.decoder.step(y_t, h_prev)

  def f_write(self, h_t):
    """Gives the softmax output distribution."""
    return self.decoder.write(h_t)

class VanillaEncDecSpec(EncoderDecoderSpec):
  """Encoder-decoder model with vanilla RNN recurrent units."""
  def create_layer(self, vocab, hidden_size, is_encoder):
    return VanillaRNNLayer(vocab, hidden_size,
                           create_init_state=is_encoder,
                           create_output_layer=not is_encoder)

class GRUEncDecSpec(EncoderDecoderSpec):
  """Encoder-decoder model with GRU recurrent units."""
  def create_layer(self, vocab, hidden_size, is_encoder):
    return GRULayer(vocab, hidden_size,
                    create_init_state=is_encoder,
                    create_output_layer=not is_encoder)

class LSTMEncDecSpec(EncoderDecoderSpec):
  """Encoder-decoder model with LSTM recurrent units."""
  def create_layer(self, vocab, hidden_size, is_encoder):
    return LSTMLayer(vocab, hidden_size,
                     create_init_state=is_encoder,
                     create_output_layer=not is_encoder)
