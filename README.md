# rsl-news-loader

Reads [RSL-News](https://github.com/radumicea/RSL-News), the Romanian Sign Language
translation dataset: its manifests and splits, the sentences of every segment, and
any features computed on the segments. It reads a local copy, or the Hugging Face
dataset repo (_link to be added_), fetching only the files used. The videos are not
on Hugging Face: RSL-News's `download.sh` makes them.

```bash
pip install "rsl-news-loader[all] @ git+https://github.com/radumicea/rsl-news-loader"
```

```python
from rsl_news import RSLNews, Windows, SentencePiece
from rsl_news.dataset import Sentences

data = RSLNews('path/to/RSL-News')                           # a local copy
data = RSLNews('data/RSL-News', hf_repo='<user>/RSL-News')   # the same files, fetched on first use
for s in data.sentences('test'):
    s.text, s.start, s.end                                   # seconds from the segment's first frame

test = Sentences(data, 'test', Windows('bsl5k', size=8, stride=2), SentencePiece(data))
test[0]       # {'features': the windows inside the sentence, 'tokens', 'text', 'weights', 'index'}
```

## What a sample is

One sentence: the features between its timestamps, its text, and its token ids.

- **Features** come from a reader, or a dict of readers:
  - `Windows(name, size, stride)`: `segment_<i>.<name>.npz` (RSL-News's `.bsl5k.npz`),
    one row per window of frames; a sentence gets the windows that lie entirely inside it.
  - `Frames(name)`: per-frame data, as the `.pose_mp.npz`, `.face_mp.npz` and
    `.pose_rtmw.npz` landmarks; a sentence gets its frames.
  - Both read RSL-News's `.npz` files, which `rsl_news.files` defines: `load(path, start,
    stop)` decodes only the chunks holding those rows, and `save(path, arrays)` writes one.
  - `Video()`: the frames of `mp4s/.../segment_<i>.mp4`.
- **Timing** (`timing=`) decides which stretch of the segment a sample covers.
  `Extend(before, after, weight)` adds seconds before and after the sentence, a fixed
  amount or `(low, high)` drawn for every sample and again every epoch; `weight`, e.g.
  `ramp(1.5, floor=0.2)`, weights each step by its distance outside the sentence. Any
  function `(sentence, rng) -> Span` works too; a sentence knows its neighbours'
  timestamps.
- **Text**: `tokenizer=` is any function from texts to token ids, e.g.
  `SentencePiece(data)` (the baselines' model, `tokenizer/`) or a Hugging Face
  tokenizer's `lambda texts: tok(texts)['input_ids']`; without one, a sample has only
  the text.

`rsl_news` without `.dataset` needs only numpy and zstandard; the extras add Hugging Face
(`hf`), PyTorch (`torch`), the tokenizer (`sentencepiece`), the landmarks (`poses`)
and the videos (`video`).

## Layout

The same locally and on Hugging Face (`<episode>` is the video id for PrimaTV,
`<yyyy>/<mm>/<dd>` otherwise):

```
manifests/<prima|protv|digi24>_manifest.json     broadcasts, their split, their segments' frames [start, end)
dataset/<Channel>/<episode>/segment_<i>.json     sentences: start, end (seconds), text
dataset/<Channel>/<episode>/segment_<i>.<name>.npz   the segment's features (rsl_news.files)
tokenizer/spm_unigram_cased_16k.model            the baselines' SentencePiece model
mp4s/<Channel>/<episode>/segment_<i>.mp4         the videos (download.sh; not on Hugging Face)
```
