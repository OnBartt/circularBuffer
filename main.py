"""Benchmark of CircBuff against collections.deque.

Run it with `uv run main.py`. It prints the machine it ran on and a markdown table
ready to paste into the README, so the numbers there can always be reproduced.

The comparison is deliberately about one task: keep the last `DEPTH` rows around and be
able to hand them over as a contiguous (DEPTH, WIDTH) array. Both containers solve it,
they just pay at different moments - deque stores references and copies nothing on
append, while CircBuff writes into its array right away. That is the whole trade-off,
so measuring writes without also measuring the conversion back to an array would flatter
deque and tell only half the story.
"""

import gc
import platform
import time
from collections import deque

import numpy as np

from CircularBuffer import CircBuff, WriteOverFlowMode

DEPTH = 4096        # buffer capacity in rows
WIDTH = 8           # values per row
ROWS = 20_000       # rows pushed through the buffer per run
BATCH = 256         # rows per write_batch call
READS = 100         # how many times the whole content is converted to an array
REPEAT = 5          # each case runs this many times; the best time is reported


def _measure(fn) -> float:
    """Best wall-clock time of REPEAT runs.

    The minimum is used rather than the mean because anything slower than the fastest run
    is noise from the OS, not from the code - the fastest run is the one least interrupted.
    """
    fn()  # warm-up: first call pays for imports, allocation and CPU frequency ramp-up
    return min(_timed(fn) for _ in range(REPEAT))


def _timed(fn) -> float:
    """One timed run, with the garbage collector held off.

    The deque case allocates hundreds of thousands of short-lived objects, so a collection
    cycle can land in the middle of a run and end up in its time. Collecting first and then
    disabling makes each run start from the same state and measure only the work itself -
    the same thing timeit does. It is re-enabled afterwards, even if the call raises.
    """
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter()
        fn()
        return time.perf_counter() - start
    finally:
        if gc_was_enabled:
            gc.enable()


def _describe_machine() -> list[str]:
    """What ran the benchmark, so the numbers can be read in context."""
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return [
        f"CPU:     {cpu}",
        f"OS:      {platform.platform()}",
        f"Python:  {platform.python_version()} ({platform.python_implementation()})",
        f"NumPy:   {np.__version__}",
    ]


def benchmark() -> dict[str, tuple[float, float]]:
    """Run every case and return {label: (circbuff_seconds, deque_seconds)}."""
    row = np.arange(WIDTH, dtype=np.int32)
    batch = np.tile(row, (BATCH, 1))

    def circbuff_single():
        buf = CircBuff(width=WIDTH, depth=DEPTH, data_type=np.int32,
                       write_overflow_mode=WriteOverFlowMode.FOLLOW)
        for _ in range(ROWS):
            buf.write_single(row)

    def deque_single():
        que = deque(maxlen=DEPTH)
        for _ in range(ROWS):
            que.append(row)

    def circbuff_batch():
        buf = CircBuff(width=WIDTH, depth=DEPTH, data_type=np.int32,
                       write_overflow_mode=WriteOverFlowMode.FOLLOW)
        for _ in range(ROWS // BATCH):
            buf.write_batch(batch)

    def deque_batch():
        que = deque(maxlen=DEPTH)
        for _ in range(ROWS // BATCH):
            que.extend(batch)

    # Both containers are filled once up front, because this case measures reading only.
    full_buf = CircBuff(width=WIDTH, depth=DEPTH, data_type=np.int32,
                        write_overflow_mode=WriteOverFlowMode.FOLLOW)
    full_buf.write_batch(np.tile(row, (DEPTH, 1)))
    full_que = deque((row for _ in range(DEPTH)), maxlen=DEPTH)

    def circbuff_read():
        for _ in range(READS):
            full_buf.dump()

    def deque_read():
        for _ in range(READS):
            np.array(list(full_que))

    cases = {
        "writing row by row": (circbuff_single, deque_single),
        f"writing in batches of {BATCH}": (circbuff_batch, deque_batch),
        f"reading the whole content as a ({DEPTH}, {WIDTH}) array": (circbuff_read, deque_read),
    }
    return {label: (_measure(mine), _measure(theirs)) for label, (mine, theirs) in cases.items()}


def _as_markdown(results: dict[str, tuple[float, float]]) -> list[str]:
    """Format as the README table, with the winner of each row in bold."""
    lines = ["| operation | this buffer | `deque(maxlen=…)` |", "|---|---|---|"]
    for label, (mine, theirs) in results.items():
        mine_txt, theirs_txt = f"{mine * 1000:.1f} ms", f"{theirs * 1000:.1f} ms"
        if mine < theirs:
            mine_txt = f"**{mine_txt}**"
        else:
            theirs_txt = f"**{theirs_txt}**"
        lines.append(f"| {label} | {mine_txt} | {theirs_txt} |")
    return lines


def main():
    rows = f"{ROWS:,}".replace(",", " ")      # thin space, not a comma, for readability
    print(f"CircBuff vs collections.deque - depth {DEPTH}, width {WIDTH}, "
          f"{rows} rows (int32), best of {REPEAT}\n")
    for line in _describe_machine():
        print(f"  {line}")
    print()

    results = benchmark()
    for line in _as_markdown(results):
        print(line)


if __name__ == "__main__":
    main()
