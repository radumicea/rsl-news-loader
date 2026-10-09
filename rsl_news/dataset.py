"""RSL-News as a PyTorch dataset: one sample per sentence.

    from rsl_news import RSLNews, Windows, SentencePiece, Extend, ramp
    from rsl_news.dataset import Sentences

    data = RSLNews('../RSL-News')
    train = Sentences(data, 'train', Windows('bsl5k', size=8, stride=2), SentencePiece(data),
                      timing=Extend(after=(1, 5), weight=ramp(1.5, floor=0.2)))
    sample = train[0]        # {'features', 'weights', 'tokens', 'text', 'index'}
"""
import numpy as np
from torch.utils.data import Dataset

from . import exact


class Sentences(Dataset):
    """The sentences of `split` (in manifest order) whose segment has its labels and features;
    the other segments are counted in `num_missing`.

    features: a reader (`Windows`, `Frames`, `Video`), or a dict of them; a sample's 'features'
        (and 'weights') is then a dict too. The first reader sets the lengths.
    tokenizer: texts -> token ids (e.g. `SentencePiece`, or a Hugging Face tokenizer's
        `lambda texts: tok(texts)['input_ids']`), or None: 'tokens' is None.
    timing: (sentence, rng) -> Span: which stretch of the segment a sample covers, and the weight
        of each step. Default: the sentence's timestamps, no weights. Random spans are drawn from
        (seed, epoch, index): call `set_epoch` at each epoch.
    keep: (sentence, steps, tokens) -> bool: which sentences to use. Default: those with a step.

    `lengths` (steps of the first reader, for epoch 0's spans) and the token arrays are flat
    numpy arrays, shared by data loader workers without copies."""

    def __init__(self, data, split, features, tokenizer=None, timing=None, keep=None, seed=0):
        self.data, self.split = data, split
        self.readers = features if isinstance(features, dict) else {None: features}
        self.timing, self.seed, self.epoch = timing or exact, seed, 0
        main = next(iter(self.readers.values()))

        sentences, segments = data.sentences(split), data.segments(split)
        data.fetch([f for s in segments for r in self.readers.values() for f in r.files(s)])
        labelled = {s for s in segments if data.path(f'dataset/{s.name}.json').exists()}
        self.segments = [s for s in segments if s in labelled and all(r.available(data, s) for r in self.readers.values())]
        self.num_missing = len(segments) - len(self.segments)
        index = {s: i for i, s in enumerate(self.segments)}
        length = {s: main.length(data, s) for s in self.segments}

        candidates = [s for s in sentences if s.segment in index]
        tokens = tokenizer([s.text for s in candidates]) if tokenizer else [None] * len(candidates)
        keep = keep or (lambda sentence, steps, ids: steps > 0)
        kept, segment, first, lengths, ids = [], [], [], [], []
        for sentence, toks in zip(candidates, tokens):
            span = self.timing(sentence, self._rng(len(kept)))
            begin, end = main.steps(data, sentence.segment, span.start, span.end, length[sentence.segment])
            if not keep(sentence, end - begin, toks):
                continue
            kept.append(sentence)
            segment.append(index[sentence.segment])
            first.append(begin)
            lengths.append(end - begin)
            ids.append(toks)
        self.num_dropped = len(candidates) - len(kept)
        self.sentences = kept
        self.texts = [s.text for s in kept]
        self.segment = np.array(segment, dtype=np.int32)
        self.sentence = np.array([s.index for s in kept], dtype=np.int32)
        self.first = np.array(first, dtype=np.int64)
        self.lengths = np.array(lengths, dtype=np.int64)
        if tokenizer:
            self.token_offsets = np.cumsum([0] + [len(t) for t in ids], dtype=np.int64)
            self.tokens = np.array([t for toks in ids for t in toks], dtype=np.int32)
        else:
            self.token_offsets = self.tokens = None
        self._fixed = self.timing is exact

    def _rng(self, index):
        return np.random.default_rng([self.seed, self.epoch, index])

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.sentences)

    @property
    def names(self):
        """<segment>:<index of the sentence in its segment>"""
        return [f'{s.segment.name}:{s.index}' for s in self.sentences]

    def tokens_of(self, idx):
        if self.tokens is None:
            return None
        return self.tokens[self.token_offsets[idx]:self.token_offsets[idx + 1]]

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        segment = sentence.segment
        span = None if self._fixed else self.timing(sentence, self._rng(idx))
        features, weights = {}, {}
        for i, (name, reader) in enumerate(self.readers.items()):
            if span is None and i == 0:
                begin, end = self.first[idx], self.first[idx] + self.lengths[idx]
            else:
                s = span or self.timing(sentence, None)
                begin, end = reader.steps(self.data, segment, s.start, s.end)
            features[name] = reader.read(self.data, segment, begin, end)
            weights[name] = None if span is None or span.weight is None else \
                np.asarray(span.weight(reader.times(begin, end)), dtype=np.float32)
        if None in self.readers:
            features, weights = features[None], weights[None]
        return {'features': features, 'weights': weights, 'tokens': self.tokens_of(idx),
                'text': sentence.text, 'index': idx}
