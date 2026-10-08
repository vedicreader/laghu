# Release notes

<!-- do not remove -->

## 0.1.0

First release.

- `load` and `LeanModel` read a model2vec static model with `StaticModel`'s `encode`, `encode_as_sequence`, `tokenize`, `tokens`, `dim`, `median_token_length` and `unk_token_id`, and give model2vec's ids and vectors bit for bit.
- The embedding table, and a quantized model's `weights` and `mapping`, are memory-mapped read-only from `model.safetensors`. Processes share one copy in the page cache.
- `Unigram` tokenizes a Unigram tokenizer.json with a Viterbi pass over a dict of pieces instead of `tokenizers`' trie: potion-multilingual-128M takes 135 MB of private memory a process instead of 1,348 MB. Added tokens (`lstrip`, `rstrip`, `normalized`), byte fallback, Metaspace `prepend_scheme`, truncation and padding follow `tokenizers`.
- A cache of segmented pre-tokens makes natural text 1.6 to 1.9 times faster on a first pass for word-splitting tokenizers.
- WordPiece, BPE and WordLevel models, and Unigram files using `single_word` added tokens, tokenize with `tokenizers`; the table is still mapped.
- `load` falls back to model2vec's `StaticModel`, with a logged warning, when laghu cannot read a model; `laghu[fallback]` installs model2vec.
- `same_as_model2vec` compares a model both ways; `laghu.bench` measures memory and speed in fresh processes. All 26 models in the README match model2vec.
- `encode([])` returns an empty `(0, dim)` array where model2vec raises; `use_multiprocessing` is accepted and ignored.
