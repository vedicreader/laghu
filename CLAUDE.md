# Working in this repo

nbdev 3 on hatchling. The notebooks under `nbs/` are the source; `laghu/*.py` is generated, except `laghu/__init__.py`, which is
written by hand (the version and the three names it exports). Edit the notebook, run `uv run nbdev-prepare` (with a hyphen), never
edit a generated `.py`. `README.md` comes from `nbs/index.ipynb`.

## The one rule

The output must be model2vec's, bit for bit: the same ids from `tokenize`, the same dtype and bits from `encode` and
`encode_as_sequence`. That is why `encode` does exactly model2vec's numpy steps in the same batches (an empty text's float64 zero row
makes its whole batch float64, and that is kept). A faster or tidier way that changes one bit is a bug. Two differences are deliberate
and documented in `02_encoder`: `encode([])` returns an empty array where model2vec raises, and `use_multiprocessing` is ignored.

## Three modules, in order

`core` (files, mapped tensors, the test corpus) has no laghu imports. `unigram` imports `core`. `encoder` imports both. `bench` is
measurement only and is never imported by the others.

## Unigram is a port of Rust

`unigram.Unigram` re-implements `tokenizers`' Unigram model and its added-token pass (`AddedVocabulary::extract_and_normalize`,
`find_matches`), and borrows the normalizer and pre-tokenizer from `tokenizers` itself. When a tokenizer.json uses something it
does not reproduce, it raises `NotImplementedError` and the caller uses `tokenizers` (`HFTok`); the table is mapped either way. Every
new feature gets a variant in `01_unigram`'s tests, compared against `encode_batch` on `corpus()` in batches of 64.

`Unigram.word` caches a pre-token's ids; that is exact only because `viterbi` depends on the pre-token alone. Cutting a pre-token
further (at `▁`, say) changes where the float sums start and breaks exact ties the other way: measured, one text in 4,726 differs.
The spike at the end of `03_bench` records what else was tried (tokie, gigatoken, worker processes) and why none is used.

## The Rust core

`rust/` builds the optional abi3 wheel `laghu-rs` (module `laghu_rs`, Python 3.11+, maturin). laghu imports it if it can and never
requires it. `Unigram.rs` is a `laghu_rs.Seg`: the same Viterbi search over UTF-8 byte offsets, plus mean pooling over the mapped
table in `encode`. It is None without `laghu_rs`, with `LAGHU_RUST=0`, with `shared=False`, and when tokenizer.json truncates or pads.

`Unigram.rs_pre` is a `laghu_rs.Pre`: a `Tokenizer` from the `tokenizers` crate built from `_stub(d)`, the tokenizer.json with a
one-piece model and the added tokens kept. It does added-token extraction, normalization and pre-tokenization, as
`encode_single_sequence` does before the model, and maps the stub's added-token ids back to tokenizer.json's. `_rs_pre` returns
None, and Python's `split` feeds `Seg`, in three cases. `compatible` finds that `laghu_rs.TOKENIZERS` and `tokenizers.__version__`
differ in major or minor version. The crate
raises `ValueError` on the JSON. `Pre.split` differs from `split` on any text in `PROBE` (the `ODD` texts under 10,000 characters),
which logs a warning.

The `tokenizers` dependency in `rust/Cargo.toml` (`0.23.2`, a compatible range), the `TOKENIZERS` constant in `rust/src/lib.rs` and
the Python `tokenizers` version laghu is tested with must share a minor version. A different minor version switches `rs_pre` off
everywhere through the guard. `PROBE` guards patch differences. The crate needs a C compiler to build `onig_sys`.

Every Rust path is tested against the Python path, `tokenizers` and model2vec: `same` in `01_unigram` runs Rust, Rust with the Python
split (`phase1`) and Python; `02_encoder` compares vectors on `corpus(3000)`. Build the extension into `.venv` with
`maturin develop --release -m rust/Cargo.toml` before running the tests, or the Rust cells skip.

## The vocabulary file

`shared_vocab` maps a Unigram vocabulary from a file built once per tokenizer.json, in a child process (`01_unigram`). Version 2 holds,
each section aligned by `_up`: the `_HDR` header (magic `LAGHUVOC`, version, counts, the tokenizer.json's size and mtime_ns and
vocabulary span, the trie's byte length, the lowest score, a CRC-32 of everything after the header); the ids (int32) and float64
scores in the marisa trie's key order; a count of pieces by length; the marisa trie; then the byte trie for `laghu_rs`. The byte
trie is a `_CSR` header (magic `LAGHUCSR`, node count, edge count), then each node's first edge (u32, nodes + 1), each edge's child
node (u32), each node's value (u32, the marisa key index, or `NO_VALUE`) and each edge's byte (u8). A node's edges are in byte
order. `csr_trie` builds it.

`Vocab.csr` gives the four byte-trie arrays as memoryviews of the mmap. `Seg` reads them through the buffer protocol and never copies
them: a copy adds 18.7 MB per process for potion-multilingual-128M. Bump `VOCAB_VERSION` whenever the layout or what goes into it
changes, so old files are built again. `shared=False` keeps the dict path; both are tested against `tokenizers`.

## Tests need models

The tests download models from the Hub through `try_resolve`, which returns None offline, so a cell skips rather than fails. The
bench cell is flagged `slow` (`tst_flags` in pyproject.toml) because it downloads about 5 GB: `uv run nbdev-test --flags slow`.

## model2vec is not a dependency

It is the optional extra `laghu[fallback]` and a dev dependency, never a runtime one. `load` imports it only to fall back, and
`same_as_model2vec` only to compare. A test that compares with it checks `M2V` (`importlib.util.find_spec('model2vec')`) and skips
without it.

## Licence

MIT (`LICENSE`, `license = "MIT"` in pyproject.toml).
