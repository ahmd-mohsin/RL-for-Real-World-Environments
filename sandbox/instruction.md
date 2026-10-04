The repository at `/testbed` is `python-slugify`. Fix the following issue.

---

**Modern algorithm: hard cut can leave a trailing delimiter after a post-stage replacement**

With `algorithm='modern'`, if a `replacement_stage='post'` replacement introduces repeated delimiters and `max_length` cuts through them, the slug can end with a delimiter:

```python
from slugify import slugify

slugify(
    'value',
    algorithm='modern',
    replacements=[('value', 'one--two')],
    replacement_stage='post',
    max_length=5,
)
# Actual:   'one-'
# Expected: 'one'
```

A slug produced by a hard cut should never end in a partial or trailing delimiter. A run of delimiters should only be emitted if at least one character of the following word also fits. This should also hold for multi-character separators, for delimiter runs at the start of the text, and for the hard-cut fallback used by `word_boundary=True`. A delimiter character that occurs inside a word must be preserved.

The default (legacy) algorithm and the public `smart_truncate` function must not change.
