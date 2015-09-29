-- A single LSTM node (one timestep)
--
-- Taken from https://github.com/oxford-cs-ml-2015/practical6
local LSTM = {}

function LSTM.create(opt)
  -- Create an lstm cell (as a gModule).
  --
  -- Options include:
  --   opt.embedding_size: Size of input word embeddings
  --   opt.hidden_size: Size of hidden state
  --   opt.use_attention: Use an attention model (Bahdanau et al. 2014)
  --   opt.hard_attention: Use hard attention model (default = soft)
  --   opt.annotation_size: Size of annotations
  --   opt.attention_hidden_size: Size of attention hidden layer.
  -- 
  -- Standard LSTM  will expect input {x, c_prev, h_prev}
  -- and return {c, h}.
  --
  -- If opt.use_attention, require additional input annotations,
  -- and opt.annotation_size and opt.attention_hidden_size must be provided.
  -- If opt.hard_attention, 
  -- expects annotations to be a single vector rather than matrix.
  --
  -- If opt.location_aware, expects additional input prev_alpha,
  -- and generates additional output alpha.
  -- If opt.location_aware and opt.hard_attention,
  -- expects prev_alpha to be a single index, rather than a distribution.
  --
  -- If opt.read_write, 
  -- output distribution alpha will just be P(read) at current time.
  local x = nn.Identity()()
  local c_prev = nn.Identity()()
  local h_prev = nn.Identity()()
  local context = nn.Identity()()

  function new_input_sum()
    -- transforms input
    local total = {}
    local i2h = nn.Linear(opt.embedding_size, opt.hidden_size)(x)
    table.insert(total, i2h)
    local h2h = nn.Linear(opt.hidden_size, opt.hidden_size)(h_prev)
    table.insert(total, h2h)
    if opt.use_attention then
      -- Contribution from the context vector 
      local c2h = nn.Linear(opt.annotation_size, opt.hidden_size)(context)
      table.insert(total, c2h)
    end
    return nn.CAddTable()(total)
  end

  local in_gate = nn.Sigmoid()(new_input_sum())
  local forget_gate = nn.Sigmoid()(new_input_sum())
  local out_gate = nn.Sigmoid()(new_input_sum())
  local in_transform = nn.Tanh()(new_input_sum())

  local next_c = nn.CAddTable()({
    nn.CMulTable()({forget_gate, c_prev}),
    nn.CMulTable()({in_gate, in_transform})
  })
  local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  local inputs = {x, c_prev, h_prev}
  if opt.use_attention then
    table.insert(inputs, context)
  end

  local outputs = {next_c, next_h}
  return nn.gModule(inputs, outputs)
end

return LSTM
