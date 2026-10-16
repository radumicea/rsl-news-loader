"""Read RSL-News: its manifests, sentences, features, poses and videos, from a local copy or from
the Hugging Face dataset repo (which has the same layout and is fetched as it is used).

    from rsl_news import RSLNews, Windows, SentencePiece

    data = RSLNews('../RSL-News')                                # a local copy
    data = RSLNews('data/RSL-News', hf_repo='<user>/RSL-News')   # the same files, fetched on first use
    for sentence in data.sentences('test'):
        sentence.text, sentence.start, sentence.end              # seconds from the segment's first frame

    bsl5k = Windows('bsl5k', size=8, stride=2)
    first, stop = bsl5k.steps(data, sentence.segment, sentence.start, sentence.end)
    x = bsl5k.read(data, sentence.segment, first, stop)         # the windows that lie inside the sentence

`rsl_news.dataset.Sentences` puts it together as a PyTorch dataset. Layout (`<episode>` is the
video id for PrimaTV, `<yyyy>/<mm>/<dd>` otherwise; see README.md):

    manifests/<prima|protv|digi24>_manifest.json   broadcasts, their split and segments (frames [start, end))
    dataset/<Channel>/<episode>/segment_<i>.json   sentences: start, end (seconds), text
    dataset/<Channel>/<episode>/segment_<i>.<name>.npy|.npz   features of the segment
    tokenizer/                                     the SentencePiece model of the baselines
    mp4s/<Channel>/<episode>/segment_<i>.mp4       the videos (RSL-News's download.sh; not on Hugging Face)
"""
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

FPS = 25
# manifest -> channel folder, in the order the segments are listed
CHANNELS = {'digi24': 'Digi24', 'prima': 'PrimaTV', 'protv': 'ProTV'}


@dataclass(frozen=True)
class Segment:
    channel: str        # PrimaTV, ProTV or Digi24
    broadcast: str      # the manifest item's id (PrimaTV) or date
    index: int          # the segment's index in its broadcast
    split: str          # train, val or test
    frames: Optional[int]   # its length in video frames (None if the manifest has no bounds)
    name: str           # <Channel>/<episode>/segment_<i>: dataset/<name>.json, mp4s/<name>.mp4, ...


@dataclass(frozen=True)
class Sentence:
    segment: Segment
    index: int                      # its position in the segment's label file
    start: float                    # seconds from the segment's first frame
    end: float
    text: str
    previous_end: Optional[float]   # the neighbouring sentences in the segment, if any
    next_start: Optional[float]


class RSLNews:
    """The dataset under `root`. With `hf_repo`, files missing under `root` are downloaded from
    that Hugging Face dataset repo (at `revision`) when first needed."""

    def __init__(self, root='.', hf_repo=None, revision=None):
        self.root = Path(root)
        self.hf_repo, self.revision = hf_repo, revision
        self._remote = None

    def path(self, name) -> Path:
        return self.root / name

    def fetch(self, names):
        """Download the files of `names` (paths under root) that are missing and on the hub."""
        missing = [n for n in names if not self.path(n).exists()]
        if not missing or not self.hf_repo:
            return
        from huggingface_hub import HfApi, hf_hub_download
        if self._remote is None:
            api = HfApi()
            self.revision = self.revision or api.dataset_info(self.hf_repo).sha
            self._remote = set(api.list_repo_files(self.hf_repo, repo_type='dataset', revision=self.revision))
        get = lambda n: hf_hub_download(self.hf_repo, n, repo_type='dataset', revision=self.revision, local_dir=self.root)
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(get, [n for n in missing if n in self._remote]))

    def items(self, split=None):
        """The manifests' items (broadcasts), each with its 'channel', of one split or all."""
        self.fetch([f'manifests/{m}_manifest.json' for m in CHANNELS])
        out = []
        for manifest, channel in CHANNELS.items():
            for item in json.loads(self.path(f'manifests/{manifest}_manifest.json').read_text(encoding='utf-8'))['items']:
                if split is None or item['split'] == split:
                    out.append({'channel': channel, **item})
        return out

    def segments(self, split=None):
        out = []
        for item in self.items(split):
            broadcast = item.get('id') or item['date']
            episode = item['id'] if 'id' in item else item['date'].replace('-', '/')
            out += [Segment(item['channel'], broadcast, s['index'], item['split'],
                            s['end'] - s['start'] if 'end' in s else None,
                            f"{item['channel']}/{episode}/segment_{s['index']}") for s in item['segments']]
        return out

    def labels(self, segment):
        """A segment's sentences as stored: [{start, end, text}, ...]"""
        return json.loads(self.path(f'dataset/{segment.name}.json').read_text(encoding='utf-8'))

    def sentences(self, split=None):
        """The sentences of a split (or all), in manifest order; segments without a label file
        (here or on the hub) are left out."""
        segments = self.segments(split)
        self.fetch([f'dataset/{s.name}.json' for s in segments])
        out = []
        for segment in segments:
            if not self.path(f'dataset/{segment.name}.json').exists():
                continue
            rows = self.labels(segment)
            for i, row in enumerate(rows):
                out.append(Sentence(segment, i, row['start'], row['end'], row['text'],
                                    rows[i - 1]['end'] if i else None, rows[i + 1]['start'] if i + 1 < len(rows) else None))
        return out


# ---------------------------------------------------------------- features
#
# A reader knows a kind of input: `files` (what to fetch for a segment), `steps` (which of its time
# steps lie in [start, end], in seconds), `times` (when each step is, in seconds) and `read`.

def frame_range(start, end, frames, fps=FPS):
    """The video frames of [start, end] seconds: [ceil(start * fps), ceil(end * fps)), within the
    segment (rounded first: 0.2 * 25 is 5.000000000000001)."""
    first = math.ceil(round(start * fps, 6))
    stop = math.ceil(round(end * fps, 6))
    return max(first, 0), min(stop, frames)


class Windows:
    """Features computed over windows of frames: dataset/<segment>.<name>.npy, [windows, ...], where
    window k covers frames [k * stride, k * stride + size). A span gets the windows that lie
    entirely inside its frames."""

    def __init__(self, name, size, stride, fps=FPS):
        self.name, self.size, self.stride, self.fps = name, size, stride, fps

    def files(self, segment):
        return [f'dataset/{segment.name}.{self.name}.npy']

    def available(self, data, segment):
        return data.path(self.files(segment)[0]).exists()

    def length(self, data, segment):
        return np.load(data.path(self.files(segment)[0]), mmap_mode='r').shape[0]

    def steps(self, data, segment, start, end, length=None):
        first = math.ceil(round(start * self.fps, 6))
        stop = math.ceil(round(end * self.fps, 6))
        length = self.length(data, segment) if length is None else length
        return max(math.ceil(first / self.stride), 0), min((stop - self.size) // self.stride + 1, length)

    def times(self, first, stop):
        return (np.arange(first, stop) * self.stride + self.size / 2) / self.fps

    def read(self, data, segment, first, stop):
        return np.load(data.path(self.files(segment)[0]), mmap_mode='r')[first:stop]


class Frames:
    """Per-frame data: dataset/<segment>.<name>.npy ([frames, ...]) or .npz. The landmarks'
    .npz files (pose, face, rtmw; RSL-News's extract_poses_*.py) are read in chunks; `arrays`
    picks some of their arrays (one name: that array; a list or None: a dict)."""

    def __init__(self, name, arrays=None, fps=FPS):
        self.name, self.arrays, self.fps = name, arrays, fps

    def _file(self, data, segment):
        npz = f'dataset/{segment.name}.{self.name}.npz'
        return npz if data.path(npz).exists() or not data.path(npz[:-1] + 'y').exists() else npz[:-1] + 'y'

    def files(self, segment):
        return [f'dataset/{segment.name}.{self.name}.npz', f'dataset/{segment.name}.{self.name}.npy']

    def available(self, data, segment):
        return data.path(self._file(data, segment)).exists()

    def length(self, data, segment):
        return segment.frames

    def steps(self, data, segment, start, end, length=None):
        return frame_range(start, end, segment.frames if length is None else length, self.fps)

    def times(self, first, stop):
        return (np.arange(first, stop) + 0.5) / self.fps

    def read(self, data, segment, first, stop):
        path = data.path(self._file(data, segment))
        if path.suffix == '.npy':
            return np.load(path, mmap_mode='r')[first:stop]
        arrays = load(path, first, stop)
        if isinstance(self.arrays, str):
            return arrays[self.arrays]
        return arrays if self.arrays is None else {k: arrays[k] for k in self.arrays}


class Video:
    """The segment's video, mp4s/<segment>.mp4 (RSL-News's download.sh), as RGB frames [T, H, W, 3]."""

    def __init__(self, fps=FPS):
        self.fps = fps

    def files(self, segment):
        return []          # not on Hugging Face

    def available(self, data, segment):
        return data.path(f'mp4s/{segment.name}.mp4').exists()

    def length(self, data, segment):
        return segment.frames

    def steps(self, data, segment, start, end, length=None):
        return frame_range(start, end, segment.frames if length is None else length, self.fps)

    def times(self, first, stop):
        return (np.arange(first, stop) + 0.5) / self.fps

    def read(self, data, segment, first, stop):
        import cv2
        cap = cv2.VideoCapture(str(data.path(f'mp4s/{segment.name}.mp4')))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        frames = []
        for _ in range(stop - first):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame[:, :, ::-1])
        cap.release()
        return np.stack(frames) if frames else np.zeros((0, 0, 0, 3), np.uint8)


def decode(blob, fmt, frames):
    """One chunk of a landmarks .npz (RSL-News's extract_poses_*.py) -> {name: array}.
    The chunk holds the arrays one after another, each with frames on the last axis, as unsigned
    ints of its bit patterns, each frame minus the one before (modulo 2^bits), bytes grouped by
    significance, zstd-compressed."""
    import zstandard
    data, pos, out = zstandard.ZstdDecompressor().decompress(blob), 0, {}
    for k, (dtype, shape) in fmt['arrays'].items():
        n = np.dtype(dtype).itemsize
        size = n * frames * math.prod(shape)
        planes = np.frombuffer(data, np.uint8, size, pos).reshape(n, -1)
        pos += size
        d = np.ascontiguousarray(planes.T).view(f'<u{n}').reshape(*shape, frames)
        out[k] = np.moveaxis(np.cumsum(d, axis=-1, dtype=d.dtype).view(dtype), -1, 0)
    if pos != len(data):
        raise ValueError('chunk size does not match its format')
    return out


def load(path, start=0, stop=None):
    """Frames [start, stop) of a landmarks .npz (default: all) -> {name: array}.
    Only the chunks holding them are read."""
    with np.load(path) as z:
        fmt = json.loads(str(z['format']))
        frames, chunk = fmt['frames'], fmt['chunk']
        stop = frames if stop is None else stop
        if not 0 <= start <= stop <= frames:
            raise ValueError(f'frames [{start}, {stop}) are not within [0, {frames})')
        first = start // chunk
        parts = [decode(z[f'chunk_{i:05d}'], fmt, min(chunk, frames - i * chunk))
                 for i in range(first, -(-stop // chunk))]
        out = {k: np.concatenate([p[k] for p in parts])[start - first * chunk:stop - first * chunk]
               if parts else np.zeros((0, *shape), dtype) for k, (dtype, shape) in fmt['arrays'].items()}
    return out


# ---------------------------------------------------------------- timing

@dataclass(frozen=True)
class Span:
    """The stretch of a segment a sample covers, in seconds, and optionally how much each moment
    counts: weight(times in seconds) -> weights."""
    start: float
    end: float
    weight: Optional[Callable] = None


def exact(sentence, rng=None):
    """The sentence's own timestamps."""
    return Span(sentence.start, sentence.end)


def ramp(fade, floor=0.0, hold=0.0):
    """weight(distance in seconds outside the sentence): 1 up to `hold`, then falling linearly to
    `floor` over `fade` seconds, `floor` beyond."""
    return lambda d: np.clip(1 - (np.asarray(d) - hold) / fade, floor, 1) if fade > 0 else np.where(np.asarray(d) <= hold, 1.0, floor)


class Extend:
    """Extra time before and after each sentence: seconds, or (low, high) for an amount drawn
    uniformly for every sample (again every epoch). `weight(distance)` (e.g. `ramp(1.5, 0.2)`)
    weights each step by how far outside the sentence it is; None: no weights."""

    def __init__(self, before=0.0, after=0.0, weight=None):
        self.before, self.after, self.weight = before, after, weight

    def __call__(self, sentence, rng):
        amount = lambda a: rng.uniform(*a) if isinstance(a, (tuple, list)) else a
        start, end = sentence.start - amount(self.before), sentence.end + amount(self.after)
        if self.weight is None:
            return Span(max(start, 0.0), end)
        core = (sentence.start, sentence.end)
        return Span(max(start, 0.0), end,
                    lambda t: self.weight(np.maximum(np.maximum(core[0] - t, t - core[1]), 0.0)))


# ---------------------------------------------------------------- text

class SentencePiece:
    """A SentencePiece model as a tokenizer: texts -> token ids (lowercased first if `lowercase`).
    `model` is a path under the dataset's root (fetched from the hub if missing) or anywhere."""

    def __init__(self, data, model='tokenizer/spm_unigram_lowercase_16k.model', lowercase=True):
        import sentencepiece
        path = Path(model)
        if not path.is_absolute():
            data.fetch([model])
            path = data.path(model)
        self.processor = sentencepiece.SentencePieceProcessor(model_file=str(path))
        self.lowercase = lowercase

    def __call__(self, texts):
        return self.processor.encode([t.lower() for t in texts] if self.lowercase else list(texts))

    @property
    def pieces(self):
        return [self.processor.id_to_piece(i) for i in range(self.processor.get_piece_size())]
