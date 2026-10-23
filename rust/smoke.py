"Check an installed laghu-rs wheel: its `tokenizers` version, and laghu's Rust ids against `tokenizers` on a tiny Unigram model."
import os, sys, tempfile
from pathlib import Path
import laghu_rs, tokenizers
from tokenizers import Tokenizer, AddedToken, models, normalizers, pre_tokenizers

from laghu.unigram import compatible
assert compatible(laghu_rs.TOKENIZERS, tokenizers.__version__), (laghu_rs.TOKENIZERS, tokenizers.__version__)

pieces = ['<unk>', '▁', '▁the', '▁cat', '▁sat', 'the', 'cat', 's', 'a', 't', 'c', 'h', 'e', '▁on', 'on', 'o', 'n', 'm']
vocab = [(p, 0.0 if i == 0 else -1.0 - len(p) / 10 - i / 100) for i, p in enumerate(pieces)]
tok = Tokenizer(models.Unigram(vocab, unk_id=0, byte_fallback=False))
tok.normalizer = normalizers.NFKC()
tok.pre_tokenizer = pre_tokenizers.Metaspace()
tok.add_special_tokens([AddedToken('<mask>', normalized=False)])
texts = ['the cat sat on the mat', 'Thé cät <mask> sat', 'ﬁne   spaces\tand\nlines', '', '<mask><mask> x', 'cats on mats ' * 20]

# Windows cannot delete a file that is still mapped; the folder is left to the system there
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    os.environ['LAGHU_CACHE'] = str(Path(d) / 'cache')
    fn = Path(d) / 'tokenizer.json'; tok.save(str(fn))
    from laghu.unigram import Unigram
    u = Unigram(fn)
    assert u.rs is not None and u.rs_pre is not None, (u.rs, u.rs_pre)
    got, want = u(texts), [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]
    assert got == want, (got, want)
    del u
print(f'laghu_rs ok: tokenizers {tokenizers.__version__}, {len(texts)} texts match on {sys.platform}')
