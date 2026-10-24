# Release notes

<!-- do not remove -->

## 0.4.0

- With `laghu-rs` 0.2.0 installed, laghu tokenizes WordPiece models in Rust when their tokenizer.json has a BertNormalizer, a BertPreTokenizer and a WordPiece model, as the English potion models do. For a float32 table without weights or a token mapping it also averages the vectors in Rust. Ids and vectors are the same as `tokenizers`' and model2vec 0.9's, bit for bit.
- potion-base-8M encodes 500-character chunks at 432,938 chunks/s with `laghu-rs`, against 55,514 chunks/s with `tokenizers` and 55,600 chunks/s with model2vec 0.9, with 81 MB of private memory against model2vec's 140 MB.
- `HFTok` has `rs`, a `laghu_rs.WordPiece` or None, and `threads`. `rs` is None when `laghu_rs` cannot be imported or has no `WordPiece`, when `LAGHU_RUST=0` is set as the model loads, when the model is loaded with `lean=False`, when tokenizer.json truncates or pads, when the installed `tokenizers` has another minor version than `laghu_rs.TOKENIZERS`, for any other pipeline or for added tokens that are normalized or use `lstrip`, `rstrip` or `single_word`, and when its ids differ from `tokenizers`' on any text in `PROBE`. The last case logs a warning.
- `Unigram` and `HFTok` both have `rs_batch` and `rs_encode`, which `LeanModel` calls for either tokenizer.
- The 0.3.0 notes gave model2vec 5,276 chunks/s on potion-multilingual-128M. That figure was measured with `tokenizers` on one core. On all cores model2vec encodes 21,105 chunks/s, and laghu with `laghu-rs` 42,385 chunks/s.

### laghu-rs 0.2.0

- `laghu_rs.WordPiece` gives the ids of a BertNormalizer, BertPreTokenizer and WordPiece pipeline without the offsets of the `tokenizers` crate, on threads with the GIL released, and averages rows of a mapped float32 table. It is a port of `FastWordPiece` from model2vec-rs (MIT, copyright (c) 2025 The Minish Lab, `LICENSE-model2vec-rs`). It takes NFD and the character classes from the crates `tokenizers` uses, and looks pieces up in the model's vocabulary alone.
- `Seg.encode` and `WordPiece.encode` average with the same code. `Seg`'s results are unchanged.

## 0.3.1

- laghu splits text in Rust when the installed Python `tokenizers` has the same major and minor version as `laghu_rs.TOKENIZERS`. Before, the versions had to match exactly, so installs with `tokenizers` 0.23.1 or 0.23.3 split text in Python. The `PROBE` check still decides.
- The `tokenizers` crate dependency in `rust/Cargo.toml` is `0.23.2`, a compatible range. It is no longer the exact pin `=0.23.2`.

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
