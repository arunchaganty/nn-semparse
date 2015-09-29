-- A single LSTM cell (one timestep)
--
-- Taken from https://github.com/oxford-cs-ml-2015/practical6
local LSTMCell = {}

function LSTMCell.create(input_size, hidden_size)
  -- Create an lstm cell (as a gModule).
  --
  -- Args:
  --   input_size: Size input vectors (which includes attention if desired)
  --   hidden_size: Size of hidden state
  -- 
  -- LSTMCell will expect input {x, c_prev, h_prev} and return {c, h}.
  local x = nn.Identity()()
  local c_prev = nn.Identity()()
  local h_prev = nn.Identity()()
  local context = nn.Identity()()

  function new_input_sum()
    -- transforms input
    local i2h = nn.Linear(opt.embedding_size, opt.hidden_size)(x)
    local h2h = nn.Linear(opt.hidden_size, opt.hidden_size)(h_prev)
    return nn.CAddTable()({i2h, h2h})
  end

  local in_gate = nn.Sigmoid()(new_input_sum())
  local forget_gate = nn.Sigmoid()(new_input_sum())
  local out_gate = nn.Sigmoid()(new_input_sum())
  local in_transform = nn.Tanh()(new_input_sum())

  local c_next = nn.CAddTable()({
    nn.CMulTable()({forget_gate, c_prev}),
    nn.CMulTable()({in_gate, in_transform})
  })
  local h_next = nn.CMulTable()({out_gate, nn.Tanh()(c_next)})
  return nn.gModule({x, c_prev, h_prev}, {c_next, h_next})
end

return LSTMCell
