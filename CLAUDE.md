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

## Tests need models

The tests download models from the Hub through `try_resolve`, which returns None offline, so a cell skips rather than fails. The
bench cell is flagged `slow` (`tst_flags` in pyproject.toml) because it downloads about 5 GB: `uv run nbdev-test --flags slow`.

## model2vec is not a dependency

It is the optional extra `laghu[fallback]` and a dev dependency, never a runtime one. `load` imports it only to fall back, and
`same_as_model2vec` only to compare. A test that compares with it checks `M2V` (`importlib.util.find_spec('model2vec')`) and skips
without it.

## Licence

MIT (`LICENSE`, `license = "MIT"` in pyproject.toml).
