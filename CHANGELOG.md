# Release notes

<!-- do not remove -->

## 0.3.0

- laghu uses the optional Rust core `laghu-rs` (module `laghu_rs`) when it is installed: `pip install laghu-rs`, or the new extra `laghu[rust]`. For a Unigram model it normalizes, pre-tokenizes and segments whole batches in Rust on all cores with the GIL released. For a float32 table without weights or a token mapping it also averages the vectors in Rust. Ids and vectors are the same as the pure Python path's and model2vec 0.9's, bit for bit.
- potion-multilingual-128M encodes 500-character chunks at 39,750 chunks/s with `laghu-rs` on 14 cores, against 1,736 chunks/s in pure Python and 5,276 chunks/s with model2vec 0.9, with 64 MB of private memory.
- laghu tokenizes in pure Python when `laghu_rs` cannot be imported, when `LAGHU_RUST=0` is set as the model loads, when the model is loaded with `shared=False`, and when tokenizer.json truncates or pads.
- laghu splits text in Python and segments it in Rust when `laghu_rs.TOKENIZERS` differs from the installed Python `tokenizers` version, when the `tokenizers` crate cannot read the tokenizer.json, or when the crate splits any text in `PROBE` differently from Python. The last case logs a warning.
- The vocabulary file is version 2. A byte trie for `laghu_rs` follows the marisa trie, which is unchanged. laghu rebuilds a version 1 file on first load.
- `laghu.bench.speed` measures chunks per second and memory in a fresh process for `how` in `'rust'`, `'phase1'` (Rust segmenting with the Python split), `'python'` and `'model2vec'`. `laghu.bench.vocab_cost` measures the private memory that making a `Unigram` adds.

### laghu-rs 0.1.0

- First release of the Rust core, an abi3 wheel for Python 3.11 and later, for macOS (arm64, x86_64) and manylinux (x86_64, aarch64).
- `laghu_rs.Seg` runs the Unigram Viterbi search over the byte trie in laghu's mapped vocabulary file, and averages rows of a mapped float32 table.
- `laghu_rs.Pre` normalizes, splits out added tokens and pre-tokenizes with the `tokenizers` crate 0.23.2. `laghu_rs.TOKENIZERS` is `"0.23.2"`.
- Building from source needs a Rust toolchain and a C compiler, for the `onig` regex engine.

## 0.2.1

- The `fallback` extra installs model2vec below 0.10. model2vec 0.10 gives different vectors, and its `StaticModel.tokenize` no longer takes `max_length`.
- `same_as_model2vec` raises `ImportError` with model2vec 0.10 instead of failing partway with `TypeError`.

## 0.1.1

- Plainer, shorter prose in the README and notebooks. No code changes.

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
