# CircularBuffer

A fixed-size ring buffer over a numpy array, with three different policies for what happens
when a write runs out of space.

The buffer stores rows of a fixed `width` in an array of `depth` slots. Nothing is ever
shifted in memory — a read pointer and a write pointer move forward and wrap around.

It started as a personal programming exercise rather than as a library, which is worth
knowing before depending on it: see [When this is worth using](#when-this-is-worth-using)
for where it earns its keep and where the standard library does the job better.

```python
import numpy as np
from CircularBuffer import CircBuff, CircBuffEmpty, WriteOverFlowMode

buf = CircBuff(width=2, depth=8, data_type=np.int32,
               write_overflow_mode=WriteOverFlowMode.LIMIT)

buf.write_single(np.array([1, 10], dtype=np.int32))
buf.write_batch(np.array([[2, 20], [3, 30]], dtype=np.int32))

item = buf.read_single()        # shape (1, width)
rows = buf.read_batch(2)        # up to 2 rows, fewer if the buffer holds fewer
```

## Requirements

Python 3.13+, numpy. The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync          # create .venv and install dependencies
uv run pytest    # run the test suite
```

## When this is worth using

For a plain queue of Python objects, `collections.deque(maxlen=n)` is the right answer: it
is in the standard library, it is battle-tested, and it gives you FOLLOW semantics for free.
This buffer earns its place only when the data is **fixed-width numeric rows** that you want
to keep in one contiguous numpy array.

The trade-off is where the copying happens. `deque` stores references and copies nothing on
append, so single appends are cheap — but turning its contents into an array costs a full
conversion. This buffer writes into its array up front, so bulk writes and array-shaped
reads are cheap, while item-at-a-time writes carry per-call overhead.

Measured on depth 4096, width 8, 20 000 rows (int32), on this machine:

| | |
|---|---|
| Host CPU | AMD Ryzen 5 PRO 5650G (Zen 3) |
| Virtualisation | QEMU/KVM on Proxmox VE, CPU type `x86-64-v3`, 4 vCPU, inside a Docker container |
| OS | Linux 6.8.0-139-generic, glibc 2.36 |
| Python | 3.13.12 (CPython) |
| NumPy | 2.3.2, using AVX2 and FMA3 |

numpy performs best with the right instruction sets available. On a virtual platform, check
what the guest actually sees — a generic CPU model can hide AVX2 and FMA.

| operation | this buffer | `deque(maxlen=…)` |
|---|---|---|
| writing row by row | 12.7 ms | **0.5 ms** |
| writing in batches of 256 | **0.1 ms** | 1.2 ms |
| reading the whole content as a `(depth, width)` array | **0.8 ms** | 53.6 ms |

The benchmark is in [`main.py`](main.py); `uv run main.py` reproduces it and prints the
machine it ran on alongside the table above. Each case is run once as a warm-up and then five
more times, and the **fastest** of those five is reported. The warm-up is there because the
first call pays for allocation and for the CPU ramping up its clock, and the minimum is used
instead of an average because anything slower than the fastest run is interference from the
operating system rather than a property of the code.

So it fits when you are buffering sampled or vector data — signal frames, sensor readings,
telemetry records — that arrives in chunks and is consumed as arrays: a sliding window over
a sampled signal, a pre-trigger recorder that keeps the last N frames before an event, or a
producer that hands whole blocks to numpy or a plotting routine. Reach for `deque` instead
when items arrive one at a time, are not numeric, or never need to be an array.

Note that this buffer is **not thread-safe**. The usual producer/consumer setup needs
external locking.

## The three overflow modes

The mode is fixed at construction and decides what a write does once the buffer is full.

| mode | on a full buffer | reads |
|---|---|---|
| `LIMIT` | refuses the write, keeps unread data intact | consume; raise when empty |
| `FLOW` | overwrites, never reports itself empty | circle forever, never consume |
| `FOLLOW` | overwrites, drags `read_ptr` along | consume; always start at the oldest survivor |

The same input, a depth-3 buffer written with `1..5` one item at a time:

```
LIMIT   last write -> FULL      array=[1, 2, 3]   reads: 1, 2, 3, then CircBuffEmpty
FLOW    last write -> NONEMPTY  array=[4, 5, 3]   reads: 4, 5, 3, 4, ... (endless)
FOLLOW  last write -> FULL      array=[4, 5, 3]   reads: 3, 4, 5, then CircBuffEmpty
```

`LIMIT` refuses items 4 and 5 outright. `FLOW` and `FOLLOW` both end up with the same bytes
in the array but disagree about where reading starts: `FOLLOW` moved `read_ptr` so that
reads come back in age order (3, 4, 5), while `FLOW` reads from wherever it left off.

## API

| method | notes |
|---|---|
| `write_single(data)` | one row; returns the new state, or `FULL` if `LIMIT` refused it |
| `write_batch(data)` | 2-D array of rows; batches longer than the buffer are handled |
| `read_single()` | returns shape `(1, width)`; raises `CircBuffEmpty` when empty |
| `read_batch(n)` | up to `n` rows; `n >= depth` means "everything" |
| `dump()` | the whole array from `read_ptr`, and empties the buffer |
| `flush()` | resets pointers and state, keeps the allocated array |
| `get_state()` | `EMPTY` / `NONEMPTY` / `FULL` |
| `get_occupancy()` | number of items currently held |
| `is_overflow()` | whether data has been overwritten or the buffer filled up |

Invalid arguments raise rather than returning a status flag: `ValueError` for a bad width or
a malformed constructor argument, `CircBuffEmpty` for reading an empty buffer.

`read_batch(n)` with `n >= depth` drains the buffer in `LIMIT` and `FOLLOW`, but in `FLOW` it
is a snapshot of the whole ring and leaves the pointers untouched — `FLOW` never consumes.

## Why pointers alone are not enough

The obvious implementation of a ring buffer tracks nothing but `read_ptr` and `write_ptr` and
derives everything else from them. That approach breaks, and it breaks quietly. This is the
single most important thing to understand about the code.

### `read_ptr == write_ptr` means two opposite things

There are `depth + 1` possible occupancies — from empty to completely full — but only `depth`
possible distances between two pointers. Two states have to share one representation, and it
is the two that matter most:

```
fresh buffer, depth=4 :  rd=0 wr=0  occupancy 0
after 4 writes        :  rd=0 wr=0  occupancy 4
```

Both are `rd == wr`. So the natural occupancy formula returns the wrong answer for one of them:

```
(wr - rd) % depth  ==  0     for the full buffer, the same as for the empty one
```

That is why `CircBuffFlags` exists and why `buffer_state` is carried alongside the pointers.
`get_occupancy()` checks the state first and only falls back to the modulo:

```python
if self.buffer_state == CircBuffFlags.FULL:
    return self.depth
return (self.write_ptr - self.read_ptr) % self.depth
```

Without that flag there is no way to answer "is it full or empty?" at all.

### Position comparisons are not a substitute

Once you accept the flag, the next temptation is to maintain it by comparing positions —
`write_ptr >= read_ptr`, or "did the pointer just cross the end of the array?". Both fail,
because **after a wrap the write pointer can legitimately sit *before* the read pointer**:

```
depth=5, rd=3, wr=3 (empty), then a batch of 7 rows

  after : rd=3  wr=0   array=[3, 4, 5, 6, 7]

  7 rows into 5 slots - the buffer certainly wrapped, yet
    position test   wr >= rd   ->  0 >= 3  ->  False    misses it
    occupancy test  occ + step >= depth  ->  0 + 7 >= 5  ->  True    catches it
```

The same trap appears from the other side. Testing "did the pointer cross the end of the
array?" detects a wrap of the *array*, not the buffer filling up. With `read_ptr` at 1, a
depth-5 buffer becomes full when `write_ptr` reaches 1 — a step from 0 to 1, which crosses
no boundary at all. A full buffer would go on silently accepting writes and overwriting
unread data.

### What the code does instead

Every overflow decision is derived from the occupancy *before* the write plus the number of
rows written:

```python
occupancy_before = self.get_occupancy()
self.write_ptr += step
...
if occupancy_before + step >= self.depth:      # filled up or overflowed
```

This holds no matter where the pointers happen to sit, how far the write jumped, or whether
the array boundary was crossed once, twice or not at all. Pointer positions are used only for
addressing memory; they are never asked to answer questions about how full the buffer is.

### Why this is worth spelling out

All of this was found by testing, not by reading. The bugs it caused were:

- a full buffer in `LIMIT` reporting `NONEMPTY` and overwriting unread data, whenever
  `read_ptr` was not at 0
- reads returning only the first item of a full buffer, the rest unreachable
- `FOLLOW` handing back the newest item instead of the oldest after an overwrite
- `FLOW` failing to set `overflow_flag` when the write wrapped

Every one of them came from the same root cause, and every one of them was silent — no
exception, no obviously wrong output, just a buffer quietly disagreeing with itself.

### Consequence for bulk reads

Because nothing is ever moved in memory, array order and logical order diverge as soon as a
pointer wraps:

```
depth=4, FOLLOW, wrote 1..6:

  array          = [5, 6, 3, 4]        physical order
  reads give       3, 4, 5, 6          logical order, oldest first
```

Every bulk read therefore has to reassemble its result from two slices — the tail of the
array and then its head. A single slice would silently return short data, which is exactly
what `read_batch` used to do in `LIMIT`.

## Tests

```bash
uv run pytest -q
```

Alongside the hand-written cases there is `test_write_batch_matches_repeated_write_single`,
which checks a property rather than an example: for every mode, every depth from 2 to 8,
every starting state and every batch size, `write_batch(n)` must leave the buffer in exactly
the state that `n` separate `write_single` calls would. That single test found bugs that the
hand-written cases had missed for all three modes — batch and single-item paths drifting
apart is precisely the kind of thing that is hard to aim at by hand.

## AI assistance

Parts of this project were developed with an AI assistant (Claude): bug fixes to the
state tracking, the test suite, code comments and this documentation. The buffer's
design and its three overflow modes are the author's own work.
