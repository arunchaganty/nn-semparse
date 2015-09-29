-- Sequence-to-sequence RNN models
require 'dp'  -- Just for a couple convenience functions
require 'nngraph'  -- Use nngraph for actual model
require 'optim'  -- For adagrad, rmsprop
local ProFi = require 'lib/ProFi'  -- Lua Profiler

local Builder = require 'builder'
local dataset_utils = require 'dataset_utils'
local LSTM = require 'lstm'
local model_utils = require 'model_utils'

local opt = {}  -- Keep command-line options at top-level scope

local function read_opt()
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Train a sequence-to-sequence RNN.')
  cmd:text('Options:')

  --[[ Model choice ]]--
  cmd:option('--useAttention', false, 'Use attention (Bahdanau et al. 2014).')
  cmd:option('--locationAware', false, 'Use location-aware attention (Chorowski et al. 2015).')

  --[[ Learning options ]]--
  cmd:option('--learningRate', 0.1, 'Learning rate at t=0.')
  cmd:option('--maxEpoch', 10, 'Maximum number of epochs to run.')
  cmd:option('--optimizer', 'adagrad', 'Optimization variant: [adagrad, rmsprop, sgd]')

  --[[ Input layer ]]--
  cmd:option('--inputEmbeddingSize', 10, 'Size of input word embeddings.')
  cmd:option('--outputEmbeddingSize', 10, 'Size of output word embeddings.')
  -- cmd:option('--gloveEmbeddings', false, 'Use GloVe embeddings.')
  -- cmd:option('--fixEmbeddings', false, "Don't backprop through word embeddings.")

  --[[ Recurrent layer(s) ]]--
  cmd:option('--encoderHiddenSize', 80, 'Number of hidden units for the encoder.') 
  cmd:option('--decoderHiddenSize', 80, 'Number of hidden units for the decoder.') 
  cmd:option('--attentionHiddenSize', 10,
             'Number of hidden units for attention mechanism.')

  --[[ Output layer]]--
  cmd:option('--maxDecodingLen', 100, 'Maximum length of decoded sentence.')
  
  --[[ Data ]]--
  cmd:option('--trainFile', 'txt/ss_small.txt',
             'Text file to use as dataset. ' ..
             'Each line is a tab-delimited pair of sentences, ' ..
             'with words delimited by spaces.')

  --[[ Debugging/Performance ]]--
  cmd:option('--nngraphDebug', false, 'Use nngraph debugging tools.')
  cmd:option('--profi', false, 'Use ProFi profiler.')
  cmd:option('--numThreads', 1, 'Number of torch threads.')

  --[[ Parse args ]]--
  cmd:text()
  opt = cmd:parse(arg)

  -- TODO(robinjia): Various error checks, such as
  --   For some models, encoderHiddenSize == decoderHiddenSize
  return opt
end

local function do_setup()
  if opt.profi then
    ProFi:start()
  end
  if opt.nngraphDebug then
    nngraph.setDebug(true)
  end
  torch.setnumthreads(opt.numThreads)
end

local function init_input_layer(vocab_size, embedding_size)
  local indices = nn.Identity()()
  local lookup = nn.LookupTable(vocab_size, embedding_size)(indices)
  return nn.gModule({indices}, {lookup})
end

local function init_lstm_state(hidden_size)
  local dummy = nn.Identity()()  -- Should always be 0
  local state = nn.Linear(1, hidden_size)(dummy)
  return nn.gModule({dummy}, {state})
end

local function init_lstm_gen_state(in_size, out_size)
  local input = nn.Identity()()
  local output = nn.Tanh()(nn.Linear(in_size, out_size)(input))
  return nn.gModule({input}, {output})
end

local function init_lstm_cell(embedding_size, hidden_size, opt)
  opt = opt or {}
  local lstm_opt = { 
    embedding_size = embedding_size,
    hidden_size = hidden_size,
  }
  for k, v in pairs(opt) do
    lstm_opt[k] = v
  end
  -- TODO(robinjia): Build deep LSTM
  return LSTM.create(lstm_opt)
end

local function init_attention_h2a(hidden_size, attention_hidden_size)
  local h = nn.Identity()()
  local h2a = nn.Linear(hidden_size, attention_hidden_size)(h)
  return nn.gModule({h}, {h2a})
end

local function init_attention_annotation2a(annotation_size, attention_hidden_size)
  local annotations = nn.Identity()()
  local annotations2a = nn.Linear(annotation_size,
                                  attention_hidden_size)(annotations)
  return nn.gModule({annotations}, {annotations2a})
end

local function init_attention_top_layer(attention_hidden_size)
  local input = nn.Identity()()
  local output = nn.Linear(attention_hidden_size, 1)(input)
  return nn.gModule({input}, {output})
end

local function init_output_layer(hidden_size, vocab_size)
  local h_t = nn.Identity()()
  local predictions = nn.LogSoftMax()(
      nn.Linear(hidden_size, vocab_size)(h_t))
  return nn.gModule({h_t}, {predictions})
end

local function init_modules(dataset)
  -- Create the atomic gModules whose parameters define the network.
  local in_vocab_size = dataset:get_in_vocab_size()
  local out_vocab_size = dataset:get_out_vocab_size()
  local decoder_lstm_opt = {
    use_attention = opt.useAttention,
    annotation_size = 2 * opt.encoderHiddenSize,
    attention_hidden_size = opt.attentionHiddenSize,
  }
  local modules = {
    enc_input_layer = init_input_layer(in_vocab_size, opt.inputEmbeddingSize),
    enc_c_0 = init_lstm_state(opt.encoderHiddenSize),
    enc_h_0 = init_lstm_state(opt.encoderHiddenSize),
    enc_lstm_cell = init_lstm_cell(opt.inputEmbeddingSize, opt.encoderHiddenSize),
    dec_input_layer = init_input_layer(out_vocab_size, opt.outputEmbeddingSize),
    dec_lstm_cell = init_lstm_cell(opt.outputEmbeddingSize, opt.decoderHiddenSize,
                                   decoder_lstm_opt),
    output_layer = init_output_layer(opt.decoderHiddenSize, out_vocab_size),
  }
  if opt.useAttention then
    -- Use a bidirectional LSTM encoder
    modules.enc_bkwd_c_0 = init_lstm_state(opt.encoderHiddenSize)
    modules.enc_bkwd_h_0 = init_lstm_state(opt.encoderHiddenSize)
    modules.enc_bkwd_lstm_cell = init_lstm_cell(opt.inputEmbeddingSize,
                                                opt.encoderHiddenSize)
    -- Setup the decoder initial states
    -- They're computed from the backward encoder's final state
    -- as in Bahdanau et al. 2014.
    modules.dec_c_0 = init_lstm_gen_state(opt.encoderHiddenSize,
                                          opt.decoderHiddenSize)
    modules.dec_h_0 = init_lstm_gen_state(opt.encoderHiddenSize,
                                          opt.decoderHiddenSize)

    -- Initialize the attention module
    modules.dec_h2a = init_attention_h2a(
        opt.decoderHiddenSize, opt.attentionHiddenSize)
    modules.dec_annotation2a = init_attention_annotation2a(
        2 * opt.encoderHiddenSize, opt.attentionHiddenSize)
    modules.dec_attention_top_layer = init_attention_top_layer(
        opt.attentionHiddenSize)
  end
  return modules
end

local function build_net(modules, clones, input_seq, output_seq)
  -- Unrolls the graph.
  --
  -- Note that we use input_seq here to compute the objective.
  -- We also need it passed during forward(), so that we can compute 
  -- the network's predictions.
  --
  -- Args:
  --   modules: A table of gModules that together define the network.
  --   clones: A table of clones of gModules, for recurrent pieces.
  --   input_seq: The input sequence, as a 1-D Tensor.
  --   output_seq: The output sequence, as a 1-D Tensor.
  -- Returns:
  --   a gModule that accepts input_seq as input and returns the objective.
  local input_len = input_seq:size(1)
  local inds_in = nn.Identity()()  -- Value during forward() is Tensor of indices
  local inds_out = nn.Identity()()  -- Value during forward() is Tensor of indices
  local dummy = nn.Identity()()  -- Value during forward() should be 0.
  local x = modules.enc_input_layer(inds_in)
  local y = modules.dec_input_layer(inds_out)

  local dec_h_list
  if opt.useAttention then
    local encoder_f_components = {
      lstm_cell_clones = clones.enc_lstm_cell,
      c_0 = modules.enc_c_0(dummy),
      h_0 = modules.enc_h_0(dummy),
    }
    local encoder_b_components = {
      lstm_cell_clones = clones.enc_bkwd_lstm_cell,
      c_0 = modules.enc_bkwd_c_0(dummy),
      h_0 = modules.enc_bkwd_h_0(dummy),
    }
    local h_fwd, c_fwd = Builder.lstm(encoder_f_components, input_seq, x)
    local h_bkwd, c_bkwd = Builder.lstm(encoder_b_components, input_seq, x,
                                        {backward=true})
    local annotations_table = {}
    for t = 1, input_len do
      annotations_table[t] = nn.Reshape(1, 2 * opt.encoderHiddenSize)(
          nn.JoinTable(1)({h_fwd[t], h_bkwd[t]}))  -- Reshape to matrix
    end
    local annotations = nn.JoinTable(1)(annotations_table)
    local decoder_components = {
      lstm_cell_clones = clones.dec_lstm_cell,
      c_0 = modules.dec_c_0(h_bkwd[1]),  
      h_0 = modules.dec_h_0(h_bkwd[1]),
    }
    local decoder_opt = {
      annotations = annotations,
      h2a_clones = clones.dec_h2a,
      annotation2a_clones = clones.dec_annotation2a,
      attention_top_layer_clones = clones.dec_attention_top_layer,
      annotation_size = opt.annotationSize,
      annotation_seq_len = input_len,
    }
    dec_h_list = Builder.lstm(decoder_components, output_seq, y, decoder_opt)
  else
    local encoder_components = {
      lstm_cell_clones = clones.enc_lstm_cell,
      c_0 = modules.enc_c_0(dummy),
      h_0 = modules.enc_h_0(dummy),
    }
    local enc_h_list, enc_c_list = Builder.lstm(encoder_components, input_seq, x)
    local decoder_components = {
      lstm_cell_clones = clones.dec_lstm_cell,
      c_0 = enc_c_list[input_len],
      h_0 = enc_h_list[input_len]
    }
    dec_h_list = Builder.lstm(decoder_components, output_seq, y)
  end
  local total_nll = Builder.output_layer(clones.output_layer, output_seq,
                                         dec_h_list)
  -- TODO(robinjia): Fix issue where inds_out is unused when output_seq has length 1
  return nn.gModule({inds_in, inds_out, dummy}, {total_nll})
end

local function extract_params(modules)
  local module_names = {}
  for k, v in pairs(modules) do
    table.insert(module_names, k)
  end
  table.sort(module_names)  -- Use a canonical ordering of the modules

  local module_list = {}
  for i, name in ipairs(module_names) do
    table.insert(module_list, modules[name])
  end
  return model_utils.combine_all_parameters(unpack(module_list))
end

local function make_clones(modules, dataset)
  local clones = {}
  local max_input_len = dataset:get_max_input_len()
  local max_output_len = dataset:get_max_output_len()
  local to_clone = {
    enc_lstm_cell = max_input_len, 
    enc_bkwd_lstm_cell = max_input_len,
    dec_lstm_cell = max_output_len, 
    dec_h2a = max_output_len,
    dec_annotation2a = max_output_len,
    dec_attention_top_layer = max_output_len,
    output_layer = max_output_len,
  }
  for key, num_clones in pairs(to_clone) do 
    if modules[key] then
      clones[key] = model_utils.clone_many_times(modules[key], num_clones)
    end
  end
  return clones
end

local function get_optimizer()
  if opt.optimizer == 'adagrad' then
    return optim.adagrad
  elseif opt.optimizer == 'rmsprop' then
    return optim.rmsprop
  elseif opt.optimizer == 'sgd' then
    return optim.sgd
  else
    error(string.format('Unrecognized optimizer "%s".', opt.optimizer))
  end
end

local function train(modules, clones, dataset, params, grad_params)
  local num_examples = dataset:get_num_examples()
  local optim_config = { learningRate = opt.learningRate }
  local optimizer = get_optimizer()

  for epoch = 1, opt.maxEpoch do
    local timer = torch.Timer()
    local epoch_loss = 0
    local perm = torch.randperm(num_examples):totable()
    for _, i in ipairs(perm) do
      local ex = dataset:get(i)
      local input_seq = ex[1]
      local output_seq = ex[2]
      local net = build_net(modules, clones, input_seq, output_seq)

      local function feval(params_)
        net:zeroGradParameters()
        local inputs = {input_seq, output_seq, torch.Tensor({0})}
        local cur_loss = net:forward(inputs)
        net:backward(inputs, torch.Tensor({1}))
        grad_params:clamp(-1, 1)  -- Clip gradients element-wise
        return cur_loss, grad_params
      end

      local _, cur_loss = optimizer(feval, params, optim_config)
      cur_loss = cur_loss[1]:totable()[1]
      epoch_loss = epoch_loss + cur_loss
    end
    local elapsed_time = timer:time().real
    print(string.format('Epoch %d: Loss = %g (time = %.2f s)', 
                        epoch, epoch_loss, elapsed_time))
  end
end

local function build_decoding_cell(modules, input_seq)
  -- Build the decoder cell, for test-time decoding.
  --
  -- A decoder cell accepts {y, c_prev, h_prev} as input.
  -- If attention is on, also requires annotations as 4th input.
  -- If location aware attention is on, also requires alpha_prev as 5th input.
  -- Outputs are {c, h}.
  -- If location aware attention is on, has alpha as 5th output.
  local input_len = input_seq:size(1)
  local y = nn.Identity()()  -- Value during forward() is an embedding vector
  local c_prev = nn.Identity()()
  local h_prev = nn.Identity()()
  local annotations = nn.Identity()()
  local lstm_cur
  if opt.useAttention then
    -- Compute the current context vector for the attention model
    local h2a = nn.Replicate(input_len)(modules.dec_h2a(h_prev))
    local annotation2a = modules.dec_annotation2a(annotations)
    local a = nn.Tanh()(nn.CAddTable()({h2a, annotation2a}))
    local e = modules.dec_attention_top_layer(a)
    local alpha = nn.SoftMax()(e)
    local context = nn.Reshape(opt.annotation_size)(
        nn.MM(true, false)({annotations, nn.Reshape(input_len, 1)(alpha)}))
    lstm_cur = modules.dec_lstm_cell({y, c_prev, h_prev, context})
  else
    lstm_cur = modules.dec_lstm_cell({y, c_prev, h_prev})
  end
  local c = nn.SelectTable(1)(lstm_cur)
  local h = nn.SelectTable(2)(lstm_cur)
  local inputs = {y, c_prev, h_prev}
  if opt.useAttention then
    table.insert(inputs, annotations)
  end
  return nn.gModule(inputs, {c, h})
end

local function build_decoding_net(modules, clones, input_seq)
  -- Build the components needed for decoding test inputs.
  --
  -- When doing decoding, advance one cell at a time, instead of unrolling
  -- the whole graph at the start.
  --
  -- Returns: Table with the following:
  --   c_0: Initial c, as a Tensor
  --   h_0: Initial h, as a Tensor
  --   annotations: If attention is on, the input annotations, as a Tensor.
  --   alpha_0: If attention is on, the initial alpha, as a Tensor
  --   cell: gModule that advances one timestep.
  local input_len = input_seq:size(1)
  local inds_in = nn.Identity()()  -- Value during forward() is Tensor of indices
  local dummy = nn.Identity()()  -- Value during forward() should be 0.
  local zero_tensor = torch.Tensor({0})
  local x = modules.enc_input_layer(inds_in)

  if opt.useAttention then
    foo
  else
    local encoder_components = {
      lstm_cell_clones = clones.enc_lstm_cell,
      c_0 = modules.enc_c_0(dummy),
      h_0 = modules.enc_h_0(dummy),
    }
    local enc_h_list, enc_c_list = Builder.lstm(encoder_components, input_seq, x)
    local enc_c_0 = enc_c_list[input_len]
    local enc_h_0 = enc_h_list[input_len]
    local module = nn.gModule({inds_in, dummy}, {enc_c_0, enc_h_0})
    local vals = module:forward({input_seq, zero_tensor})

    return {
      c_0 = vals[1],
      h_0 = vals[2],
      cell = build_decoding_cell(modules, input_seq)
    }
  end
end

local function decode(modules, clones, input_seq, dataset)
  local net = build_decoding_net(modules, clones, input_seq)
  local c = net.c_0
  local h = net.h_0
  local y = {}
  for t = 1, opt.maxDecodingLen do
    local y_dist = modules.output_layer:forward(h)
    local log_p, index = torch.max(y_dist, 1)
    local y_t = index:totable()[1]
    table.insert(y, y_t)
    if y_t == dataset.output_vocab.EOS then break end
    local y_emb = modules.dec_input_layer:forward(index):select(1, 1)
    local vals = net.cell:forward({y_emb, c, h})
    c = vals[1]
    h = vals[2]
  end
  return torch.Tensor(y)
end

local function test(modules, clones, dataset)
  -- Evaluate performance on the given dataset.
  --
  -- Currently, just use exact string matches.
  local num_correct = 0
  local timer = torch.Timer()
  for i, ex in ipairs(dataset.dataset) do
    local x = ex[1]
    local y = ex[2]
    local y_pred = decode(modules, clones, x, dataset)
    if y:size(1) == y_pred:size(1) and torch.all(torch.eq(y, y_pred)) then
      num_correct = num_correct + 1
    end
    -- Print info
    print(string.format('x      = %s', table.concat(x:totable(), ' ')))
    print(string.format('y      = %s', table.concat(y:totable(), ' ')))
    print(string.format('y_pred = %s', table.concat(y_pred:totable(), ' ')))
  end
  local elapsed_time = timer:time().real
  print(string.format('Accuracy: %d / %d = %0.2f%% (time = %.2f s)',
                      num_correct, dataset:get_num_examples(),
                      100 * num_correct / dataset:get_num_examples(),
                      elapsed_time))
end

local function main()
  print(opt)
  do_setup()
  local train_dataset = dataset_utils.from_file(opt.trainFile)
  -- local test_dataset = dataset_utils.from_file(opt.testFile)
  local modules = init_modules(train_dataset)
  local params, grad_params = extract_params(modules)
  local clones = make_clones(modules, train_dataset)
  print('Training...')
  train(modules, clones, train_dataset, params, grad_params)
  print('Evaluating on training set...')
  test(modules, clones, train_dataset)
  -- test(modules, test_dataset)
end

local function do_exit()
  if opt.profi then
    ProFi:stop()
    ProFi:writeReport('ProFi_report.txt')
  end
end

opt = read_opt()
main()
do_exit()
