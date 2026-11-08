"""The files of RSL-News's features and landmarks (.bsl5k.npz, .pose_mp.npz, .face_mp.npz,
.pose_rtmw.npz): written by RSL-News's extract_*.py with save(), read with load().

    from rsl_news.files import load
    pose = load('dataset/ProTV/2021/03/16/segment_1.pose_mp.npz')             # all frames
    face = load('dataset/ProTV/2021/03/16/segment_1.face_mp.npz', 500, 650)   # frames 500..649

A file is an uncompressed .npz: `format`, a JSON string {chunk, frames, arrays: {name: [dtype,
shape of one frame]}} (and "delta": false if the frames are not stored as differences), and
chunk_<i:05d>, rows [i * chunk, (i + 1) * chunk) of every array, encoded (encode()). The rows
are video frames, or windows of frames for features. load() decodes only the chunks it needs.
"""
import json
import math

import numpy as np
import zstandard

CHUNK = 256  # rows compressed together; a clip only decodes the chunks it touches


def encode(arrays, delta=True):
    """One chunk of frames of every array -> one zstd frame (level 19).

    The arrays one after another, each as: frames moved to the last axis; values taken
    as the little-endian unsigned ints of their bit patterns; if `delta`, each frame minus
    the previous one (the chunk's first frame minus zero), modulo 2^bits; then the bytes
    grouped by significance (all lowest bytes first). Consecutive frames of landmarks are
    close, so the differences are small and their high bytes mostly zero, which zstd packs
    well; features that change more from step to step pack better without them."""
    planes = []
    for a in arrays.values():
        n = a.itemsize
        u = np.ascontiguousarray(np.moveaxis(a, 0, -1)).view(f'<u{n}')
        if delta:
            u = np.diff(u, axis=-1, prepend=np.zeros_like(u[..., :1]))
        planes.append(np.ascontiguousarray(u.view(np.uint8).reshape(-1, n).T).tobytes())
    return np.frombuffer(zstandard.ZstdCompressor(level=19).compress(b''.join(planes)), np.uint8)


def decode(blob, fmt, frames):
    """One chunk (encode()) of a file with format `fmt`, holding `frames` rows -> {name: array}."""
    data, pos, out = zstandard.ZstdDecompressor().decompress(blob), 0, {}
    for k, (dtype, shape) in fmt['arrays'].items():
        n = np.dtype(dtype).itemsize
        size = n * frames * math.prod(shape)
        planes = np.frombuffer(data, np.uint8, size, pos).reshape(n, -1)
        pos += size
        d = np.ascontiguousarray(planes.T).view(f'<u{n}').reshape(*shape, frames)
        if fmt.get('delta', True):
            d = np.cumsum(d, axis=-1, dtype=d.dtype)
        out[k] = np.moveaxis(d.view(dtype), -1, 0)
    if pos != len(data):
        raise ValueError('chunk size does not match its format')
    return out


def save(path, arrays, delta=True):
    """{name: array, rows first} -> the file at `path` (.npz). delta: store each row as its
    difference from the previous one (the landmarks; not the features, which change more)."""
    frames = len(next(iter(arrays.values())))
    fmt = {'chunk': CHUNK, 'frames': frames,
           'arrays': {k: [a.dtype.str, list(a.shape[1:])] for k, a in arrays.items()}}
    if not delta:
        fmt['delta'] = False
    chunks = {f'chunk_{i:05d}': encode({k: a[s:s + CHUNK] for k, a in arrays.items()}, delta)
              for i, s in enumerate(range(0, frames, CHUNK))}
    np.savez(path, format=np.array(json.dumps(fmt)), **chunks)


def load(path, start=0, stop=None, clip=False):
    """Rows [start, stop) of a file (default: all) -> {name: array}. Only the chunks holding
    them are read. clip: cut [start, stop) to the file, as slicing an array does, instead of
    raising an error."""
    with np.load(path) as z:
        fmt = json.loads(str(z['format']))
        frames, chunk = fmt['frames'], fmt['chunk']
        stop = frames if stop is None else stop
        if clip:
            start = min(max(start, 0), frames)
            stop = min(max(stop, start), frames)
        if not 0 <= start <= stop <= frames:
            raise ValueError(f'frames [{start}, {stop}) are not within [0, {frames})')
        first = start // chunk
        parts = [decode(z[f'chunk_{i:05d}'], fmt, min(chunk, frames - i * chunk))
                 for i in range(first, -(-stop // chunk))]
        out = {k: np.concatenate([p[k] for p in parts])[start - first * chunk:stop - first * chunk]
               if parts else np.zeros((0, *shape), dtype) for k, (dtype, shape) in fmt['arrays'].items()}
    return out


def length(path):
    """The number of rows (frames or windows) of a file, without decoding it."""
    with np.load(path) as z:
        return json.loads(str(z['format']))['frames']
