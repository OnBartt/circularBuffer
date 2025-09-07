from enum import Enum

import numpy as np


class CircBuffFlags(Enum):
    EMPTY = 0
    FULL = 1
    NONEMPTY = 2
    WIDTH_ERROR = 3


class WriteOverFlowMode(Enum):
    """Modes for write overflow handling
        1) LIMIT - When overflow occurs, new data write is prohibited.
        2) FLOW - When overflow occurs, new data write is permited."""

    LIMIT = 0
    FLOW = 1


class CircBuff:
    instance_count = 0

    def __init__(self, width: int, depth: int, data_type: np.dtype, write_overflow_mode: WriteOverFlowMode):
        CircBuff.instance_count += 1
        self.width = width
        self.depth = depth
        self.data = np.zeros((self.depth, self.width), dtype=data_type)
        self.read_ptr = 0
        self.write_ptr = 0
        self.write_overflow_mode = write_overflow_mode
        self.buffer_state = CircBuffFlags.EMPTY

    def __inc_write_ptr(self):
        self.write_ptr += 1
        if self.write_ptr >= self.depth:
            self.write_ptr = 0

        if self.write_ptr == self.read_ptr:
            """If write_ptr catches up the read_ptr, buffer is full."""
            self.buffer_state = CircBuffFlags.FULL
        else:
            self.buffer_state = CircBuffFlags.NONEMPTY

        return self.buffer_state

    def __inc_read_ptr(self):
        self.read_ptr += 1
        if self.read_ptr >= self.depth:
            self.read_ptr = 0

        if self.read_ptr == self.write_ptr:
            """If read_ptr catches up the write_ptr and the buffer was not overflowed, buffer is empty.
               If it was overflowed, the buffer is only non-empty"""
            if self.buffer_state == CircBuffFlags.FULL:
                self.buffer_state = CircBuffFlags.NONEMPTY
            else:
                self.buffer_state = CircBuffFlags.EMPTY

        return self.buffer_state

    def gey_state(self):
        return self.buffer_state

    def get_occupancy(self):
        if self.buffer_state == CircBuffFlags.FULL:
            return self.depth
        else:
            return (self.write_ptr - self.read_ptr) % self.depth

    def flush(self):
        self.write_ptr = 0
        self.read_ptr = 0
        self.buffer_state = CircBuffFlags.EMPTY

        return CircBuffFlags.EMPTY

    def write_single(self, data: np.ndarray):
        if data.size > self.width or data.size < self.width:
            return CircBuffFlags.WIDTH_ERROR

        if self.buffer_state == CircBuffFlags.FULL and self.write_overflow_mode == WriteOverFlowMode.LIMIT:
            return CircBuffFlags.FULL
        else:
            self.data[self.write_ptr, :] = data
            return self.__inc_write_ptr()

    def read_single(self):
        if self.buffer_state == CircBuffFlags.EMPTY:
            return CircBuffFlags.EMPTY
        else:
            data = self.data[self.read_ptr, :].copy()
            self.__inc_read_ptr()
            return data
