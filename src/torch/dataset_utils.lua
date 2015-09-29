-- Utilities for handling datasets
dataset_utils = {}

local class = require 'class'

local Vocabulary = class('Vocabulary')
dataset_utils.Vocabulary = Vocabulary
-- A Vocabulary of words.
--
-- This class exports the following fields:
--   self.word_to_ind: table mapping words to numerical indices
--   self.ind_to_word: table mapping numerical indices to words
function Vocabulary:__init(sentences)
  -- Create the vocabulary
  -- Args:
  --   sentences: Table of sentences, with words separated by spaces.
  self.EOS = 1  -- End-of-sentence token
  self.UNK = 2  -- Unknown word
  local next_ind = 3
  self.word_to_ind = {['EOS'] = self.EOS, ['UNK'] = self.UNK}
  self.ind_to_word = {'EOS', 'UNK'}
  for i, sentence in ipairs(sentences) do
    for w in string.gmatch(sentence, '[^ ]+') do
      if not self.word_to_ind[w] then
        self.ind_to_word[next_ind] = w
        self.word_to_ind[w] = next_ind
        next_ind = next_ind + 1
      end
    end
  end
end

function Vocabulary:get_size()
  return #self.ind_to_word
end

Seq2SeqDataset = class('Seq2SeqDataset')
dataset_utils.Seq2SeqDataset = Seq2SeqDataset
-- A sequence-to-sequence dataset.
--
-- A dataset is a pair of (input, output) sentences.
-- We also have to store a mapping between words in the dataset
-- and numerical indices.
--
-- This class exports the following fields:
--   self.raw_dataset: dataset, as list of sentence strings
--   self.dataset: dataset, as list of torch Tensors
--
-- These fields should be considered read-only.
-- A few helpful utilities are also provided.
function Seq2SeqDataset:__init(raw_data)
  -- Create the sequence-to-sequence dataset.
  --
  -- Args:
  --   raw_data: Table of pairs of sentences, with words separated by spaces.
  self.raw_data = raw_data
  local input_sentences = {}
  local output_sentences = {}
  for i, ex in ipairs(raw_data) do
    table.insert(input_sentences, ex[1])
    table.insert(output_sentences, ex[2])
  end
  self.input_vocab = Vocabulary(input_sentences)
  self.output_vocab = Vocabulary(output_sentences)
  self.dataset = {}
  self.max_input_len = 0
  self.max_output_len = 0
  for i, ex in ipairs(raw_data) do
    local input_sentence = ex[1]
    local output_sentence = ex[2]
    local cur_input_table = {}
    local cur_output_table = {}
    for w in string.gmatch(input_sentence, '[^ ]+') do
      table.insert(cur_input_table, self.input_vocab.word_to_ind[w])
    end
    for w in string.gmatch(output_sentence, '[^ ]+') do
      table.insert(cur_output_table, self.output_vocab.word_to_ind[w])
    end

    -- Add EOS tags
    table.insert(cur_input_table, self.input_vocab.EOS)
    table.insert(cur_output_table, self.output_vocab.EOS)

    if #cur_input_table > self.max_input_len then
      self.max_input_len = #cur_input_table
    end
    if #cur_output_table > self.max_output_len then
      self.max_output_len = #cur_output_table
    end
    local cur_input_torch = torch.Tensor(cur_input_table)
    local cur_output_torch = torch.Tensor(cur_output_table)
    table.insert(self.dataset, {cur_input_torch, cur_output_torch})
  end
end

function Seq2SeqDataset:get(i)
  return self.dataset[i]
end

function Seq2SeqDataset:get_in_vocab_size()
  return self.input_vocab:get_size()
end

function Seq2SeqDataset:get_out_vocab_size()
  return self.output_vocab:get_size()
end

function Seq2SeqDataset:get_max_input_len()
  return self.max_input_len
end

function Seq2SeqDataset:get_max_output_len()
  return self.max_output_len
end

function Seq2SeqDataset:get_num_examples()
  return #self.dataset
end

function Seq2SeqDataset:print_debug()
  for i, ex in ipairs(self.dataset) do
    print(string.format('  %s -> %s', table.concat(ex[1]:totable(), ' '),
                                      table.concat(ex[2]:totable(), ' ')))
  end
end

function dataset_utils.from_file(filename)
  -- Reads a dataset from a file
  --
  -- Returns either LanguageModelDataset or Seq2SeqDataset, depending on
  -- file format.
  --
  -- Expects file to have one example per line.
  -- If seq2seq, input and output should be separated by tab.
  local raw_data = {}
  local is_seq2seq = true
  for line in io.lines(filename) do
    if string.find(line, '\t') then
      is_seq2seq = true
      local ex = {}
      for seq in string.gmatch(line, '[^\t]+') do
        table.insert(ex, seq)
      end
      table.insert(raw_data, ex)
    else
      is_seq2seq = false
      table.insert(raw_data, line)
    end
  end
  return ((is_seq2seq and Seq2SeqDataset(raw_data))
          or LanguageModelDataset(raw_data))
end

return dataset_utils
