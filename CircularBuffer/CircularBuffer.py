from enum import Enum

import numpy as np


class CircBuffFlags(Enum):
    """Buffer fill state.

    This is tracked explicitly rather than derived from the pointers, because read_ptr ==
    write_ptr is ambiguous in a ring buffer: it holds both when the buffer is empty and
    when it is completely full. The pointers alone cannot tell those two apart.
    """

    EMPTY = 0
    FULL = 1
    NONEMPTY = 2


class WriteOverFlowMode(Enum):
    """What a write does once the buffer runs out of free space.

    LIMIT  - refuses the write and keeps unread data intact; the only mode that can reject
             a write, which is why it is the only one with an early return in write_single.
    FLOW   - keeps overwriting and never reports itself empty, so reads circle forever;
             suits a continuously sampled signal where only the recent window matters.
    FOLLOW - overwrites too, but drags read_ptr along so that reads still start at the
             oldest surviving item instead of landing in the middle of the rewritten data.
    """

    LIMIT = 0
    FLOW = 1
    FOLLOW = 2


class CircBuffEmpty(Exception):
    """Raised when reading from an empty buffer."""


class CircBuff:
    """Fixed-size ring buffer over a numpy array of shape (depth, width).

    Both pointers move forward only and wrap modulo depth; nothing is ever shifted in memory.
    Because of that, the array order and the logical order diverge as soon as a pointer wraps,
    which is why every bulk read has to reassemble its result from two slices.

    Occupancy and fullness are derived from buffer_state plus the pointers, never from the
    pointers alone - see CircBuffFlags for why that distinction matters.
    """

    instance_count = 0

    def __init__(self, width: int, depth: int, data_type: np.dtype, write_overflow_mode: WriteOverFlowMode):
        self.width = width
        self.depth = depth
        self.data_type = np.dtype(data_type)
        # Validate before allocating: numpy would reject some of these on its own, but with
        # messages about array dimensions rather than about the argument the caller passed.
        # bool is excluded explicitly because it is a subclass of int, so depth=True would
        # silently pass as depth=1.
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise ValueError(f"width must be an integer >= 1, got {width!r}")
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
            raise ValueError(f"depth must be an integer >= 1, got {depth!r}")
        if not isinstance(write_overflow_mode, WriteOverFlowMode):
            raise TypeError(f"write_overflow_mode must be a WriteOverFlowMode, got {write_overflow_mode!r}")

        CircBuff.instance_count += 1
        self.data = np.zeros((self.depth, self.width), dtype=self.data_type)
        self.read_ptr = 0
        self.write_ptr = 0
        self.write_overflow_mode = write_overflow_mode
        self.buffer_state = CircBuffFlags.EMPTY
        self.overflow_flag = False

    def _move_write_ptr(self, step) -> CircBuffFlags:
        occupancy_before = self.get_occupancy()
        self.write_ptr += step
        if self.write_ptr >= self.depth:
            self.write_ptr %= self.depth

        match self.write_overflow_mode:
            case WriteOverFlowMode.LIMIT:
                # after a write, equal pointers cannot mean empty -> the buffer is full
                if self.write_ptr == self.read_ptr:
                    self.buffer_state = CircBuffFlags.FULL
                else:
                    self.buffer_state = CircBuffFlags.NONEMPTY

            case WriteOverFlowMode.FLOW:
                # same as FOLLOW: after wrapping, write_ptr may end up before read_ptr,
                # so detect the wrap from the occupancy rather than by comparing positions
                if occupancy_before + step >= self.depth:
                    self.overflow_flag = True
                self.buffer_state = CircBuffFlags.NONEMPTY

            case WriteOverFlowMode.FOLLOW:
                # filled up or overflowed -> the read pointer steps aside to the write pointer.
                # Deriving this from the occupancy is more reliable than comparing pointer
                # positions: after wrapping, write_ptr may end up before read_ptr.
                if occupancy_before + step >= self.depth:
                    self.buffer_state = CircBuffFlags.FULL
                    self.overflow_flag = True
                    self.read_ptr = self.write_ptr
                else:
                    self.buffer_state = CircBuffFlags.NONEMPTY

            case _:
                raise ValueError(f"unknown write_overflow_mode: {self.write_overflow_mode!r}")

        return self.buffer_state

    def _move_read_ptr(self, step) -> CircBuffFlags:
        self.read_ptr += step
        if self.read_ptr >= self.depth:
            self.read_ptr %= self.depth

        match self.write_overflow_mode:
            case WriteOverFlowMode.LIMIT:
                if self.read_ptr == self.write_ptr:
                    self.buffer_state = CircBuffFlags.EMPTY
                else:
                    self.buffer_state = CircBuffFlags.NONEMPTY

            case WriteOverFlowMode.FLOW:
                # FLOW never reports itself empty: reading does not consume, it just advances
                # around the ring, so there is always something to hand back.
                self.buffer_state = CircBuffFlags.NONEMPTY

            case WriteOverFlowMode.FOLLOW:
                # same as LIMIT: after a read, equal pointers mean empty, while
                # rd > wr is the ordinary state of a wrapped buffer that still holds data
                if self.read_ptr == self.write_ptr:
                    self.buffer_state = CircBuffFlags.EMPTY
                else:
                    self.buffer_state = CircBuffFlags.NONEMPTY

                # The caller has now seen the data, so the "you lost something" warning has
                # served its purpose and is cleared. FLOW keeps its flag until dump or flush.
                self.overflow_flag = False

            case _:
                raise ValueError(f"unknown write_overflow_mode: {self.write_overflow_mode!r}")

        return self.buffer_state

    def _trim_to_depth(self, data: np.ndarray) -> np.ndarray:
        """Keep only the last `depth` rows of an oversized batch, already ordered as they land in the array.

        Rows older than the last `depth` would be overwritten before anyone could read them,
        so copying them into the array at all is wasted work. The surviving rows are rotated
        here rather than written in several passes, which keeps the caller down to one pass.
        """
        if data.shape[0] <= self.depth:
            return data
        if (res := data.shape[0] % self.depth) == 0:
            return data[-self.depth:, :]
        return np.vstack((data[data.shape[0]-res:data.shape[0], :],
                          data[data.shape[0]-(self.depth-res)-res:data.shape[0]-res, :]))

    def get_state(self) -> CircBuffFlags:
        return self.buffer_state

    def is_overflow(self) -> bool:
        return self.overflow_flag

    def get_occupancy(self) -> int:
        # The FULL case has to be special-cased: for a full buffer read_ptr == write_ptr, so
        # the modulo below would compute 0 - the same answer it gives for an empty buffer.
        if self.buffer_state == CircBuffFlags.FULL:
            return self.depth
        else:
            return (self.write_ptr - self.read_ptr) % self.depth

    def flush(self) -> CircBuffFlags:
        self.write_ptr = 0
        self.read_ptr = 0
        self.overflow_flag = False
        self.buffer_state = CircBuffFlags.EMPTY

        return CircBuffFlags.EMPTY

    def write_single(self, data: np.ndarray) -> CircBuffFlags:
        if data.size != self.width:
            raise ValueError(f"item has width {data.size}, buffer expects {self.width}")

        match self.write_overflow_mode:
            case WriteOverFlowMode.LIMIT:
                # LIMIT is the only mode that may refuse a write; the other two always accept
                # and differ only in how _move_write_ptr treats the pointers afterwards.
                if self.buffer_state == CircBuffFlags.FULL:
                    return CircBuffFlags.FULL
                else:
                    self.data[self.write_ptr, :] = data
                    return self._move_write_ptr(1)

            case WriteOverFlowMode.FLOW | WriteOverFlowMode.FOLLOW:
                self.data[self.write_ptr, :] = data
                return self._move_write_ptr(1)

            case _:
                raise ValueError(f"unknown write_overflow_mode: {self.write_overflow_mode!r}")

    def read_single(self) -> np.ndarray:
        if self.buffer_state == CircBuffFlags.EMPTY:
            raise CircBuffEmpty("cannot read from an empty buffer")

        # All three modes read identically; what differs between them is where read_ptr was
        # left by the preceding writes. The match only rejects an unknown mode.
        match self.write_overflow_mode:
            case WriteOverFlowMode.LIMIT | WriteOverFlowMode.FLOW | WriteOverFlowMode.FOLLOW:
                # .copy() so the caller cannot mutate the buffer through the returned view
                data = self.data[self.read_ptr, :].copy()
                self._move_read_ptr(1)
                return data[None, :]

            case _:
                raise ValueError(f"unknown write_overflow_mode: {self.write_overflow_mode!r}")

    def _contents_from_read_ptr(self) -> np.ndarray:
        """The whole array rearranged so that it starts at read_ptr.

        Deliberately returns every slot, including ones never written and ones already read,
        so the result is always `depth` rows long regardless of occupancy. Callers that want
        only the live items should read them instead of dumping.
        """
        return np.vstack((self.data[self.read_ptr:, :], self.data[0:self.read_ptr, :])).copy()

    def dump(self) -> np.ndarray:
        self.overflow_flag = False
        self.buffer_state = CircBuffFlags.EMPTY
        return self._contents_from_read_ptr()

    def write_batch(self, data: np.ndarray) -> CircBuffFlags:
        if data.shape[1] != self.width:
            raise ValueError(f"batch has width {data.shape[1]}, buffer expects {self.width}")

        match self.write_overflow_mode:
            case WriteOverFlowMode.LIMIT:
                if self.buffer_state == CircBuffFlags.FULL:
                    return CircBuffFlags.FULL
                # LIMIT must not touch unread data, so anything past the free space is dropped
                # rather than wrapped around - this mirrors write_single refusing once FULL.
                free = self.depth - self.get_occupancy()
                n = min(data.shape[0], free)           # how many rows actually get written
                tail = self.depth - self.write_ptr     # how many fit before the end of the array

                if n > tail:
                    self.data[self.write_ptr:self.depth, :] = data[0:tail, :]
                    self.data[0:n-tail, :] = data[tail:n, :]
                else:
                    self.data[self.write_ptr:self.write_ptr+n, :] = data[0:n, :]

                self._move_write_ptr(n)
                return self.buffer_state

            case WriteOverFlowMode.FLOW | WriteOverFlowMode.FOLLOW:
                data_tmp = self._trim_to_depth(data)
                tail = self.depth - self.write_ptr
                if data_tmp.shape[0] > tail:
                    self.data[self.write_ptr:self.depth, :] = data_tmp[0:tail, :]
                    self.data[0:data_tmp.shape[0]-tail, :] = data_tmp[tail:, :]
                else:
                    self.data[self.write_ptr:self.write_ptr+data_tmp.shape[0], :] = data_tmp
                # Deliberately data.shape[0], not data_tmp.shape[0]: the pointer has to end up
                # where it would after writing every row one by one. Passing the trimmed count
                # would land it a whole lap short for batches longer than the buffer.
                self._move_write_ptr(data.shape[0])
                return self.buffer_state

            case _:
                raise ValueError(f"unknown write_overflow_mode: {self.write_overflow_mode!r}")

    def read_batch(self, depth: int) -> np.ndarray:
        """Read up to `depth` items. Returns fewer if the buffer holds fewer.

        Asking for at least the buffer's own depth means "give me everything"; in LIMIT and
        FOLLOW that drains the buffer, while FLOW treats it as a snapshot and leaves it alone.
        """
        if self.buffer_state == CircBuffFlags.EMPTY:
            raise CircBuffEmpty("cannot read from an empty buffer")

        match self.write_overflow_mode:
            case WriteOverFlowMode.LIMIT:
                if depth >= self.depth:
                    return self.dump()

                # Live items are not necessarily contiguous: once read_ptr has wrapped they sit
                # at the end of the array and continue at the start, so they need reassembling.
                count = min(depth, self.get_occupancy())
                if count > self.depth - self.read_ptr:
                    data = np.vstack((self.data[self.read_ptr:self.depth, :],
                                      self.data[0:count-(self.depth - self.read_ptr), :]))
                else:
                    data = self.data[self.read_ptr:self.read_ptr+count, :]
                self._move_read_ptr(count)

                return data.copy()

            case WriteOverFlowMode.FLOW:
                if depth >= self.depth:
                    # in FLOW, "take everything" is a snapshot of the whole ring: unlike
                    # LIMIT and FOLLOW the buffer is not drained and read_ptr stays put
                    return self._contents_from_read_ptr()

                if depth > self.depth - self.read_ptr:
                    tmp_data = np.vstack((self.data[self.read_ptr:self.depth, :],
                                          self.data[0:depth-(self.depth - self.read_ptr), :]))
                else:
                    tmp_data = self.data[self.read_ptr:self.read_ptr+depth, :]

                self._move_read_ptr(depth)
                return tmp_data.copy()

            case WriteOverFlowMode.FOLLOW:
                if depth >= self.depth:
                    return self.dump()

                if depth > self.depth - self.read_ptr:
                    tmp_data = np.vstack((self.data[self.read_ptr:self.depth, :],
                                          self.data[0:depth-(self.depth - self.read_ptr), :]))
                else:
                    tmp_data = self.data[self.read_ptr:self.read_ptr+depth, :]

                self._move_read_ptr(depth)
                return tmp_data.copy()

            case _:
                raise ValueError(f"unknown write_overflow_mode: {self.write_overflow_mode!r}")
