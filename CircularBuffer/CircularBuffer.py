from enum import Enum

import numpy as np


class CircBuffFlags(Enum):
    EMPTY = 0
    FULL = 1
    NONEMPTY = 2
    WIDTH_ERROR = 3


class WriteOverFlowMode(Enum):
    LIMIT = 0
    FLOW = 1
    FOLLOW = 2


class CircBuff:
    instance_count = 0

    def __init__(self, width: int, depth: int, data_type: np.dtype, write_overflow_mode: WriteOverFlowMode):
        CircBuff.instance_count += 1
        self.width = width
        self.depth = depth
        self.data_type = np.dtype(data_type)
        self.data = np.zeros((self.depth, self.width), dtype=self.data_type)
        self.read_ptr = 0
        self.write_ptr = 0
        self.write_overflow_mode = write_overflow_mode
        self.buffer_state = CircBuffFlags.EMPTY
        self.overflow_flag = False

    def __inc_write_ptr(self):
        self.write_ptr += 1
        self.buffer_state = CircBuffFlags.NONEMPTY
        if self.write_ptr >= self.depth:
            self.write_ptr = 0
            self.buffer_state = CircBuffFlags.FULL

        match self.write_overflow_mode:
            case WriteOverFlowMode.FLOW:
                if self.write_ptr == self.read_ptr:
                    self.overflow_flag = True

            case WriteOverFlowMode.FOLLOW:
                if self.buffer_state == CircBuffFlags.FULL:
                    self.buffer_state = CircBuffFlags.FULL
                elif self.write_ptr == self.read_ptr:
                    self.buffer_state = CircBuffFlags.FULL
                else:
                    self.buffer_state = CircBuffFlags.NONEMPTY

        return self.buffer_state

    def __inc_read_ptr(self):
        self.read_ptr += 1
        if self.read_ptr >= self.depth:
            self.read_ptr = 0

        match self.write_overflow_mode:
            case WriteOverFlowMode.LIMIT:
                if self.read_ptr == self.write_ptr:
                    self.buffer_state = CircBuffFlags.EMPTY
                else:
                    self.buffer_state = CircBuffFlags.NONEMPTY

            case WriteOverFlowMode.FLOW:
                self.buffer_state = CircBuffFlags.NONEMPTY

            case WriteOverFlowMode.FOLLOW:
                if self.read_ptr == self.write_ptr:
                    if self.buffer_state == CircBuffFlags.FULL:
                        self.buffer_state = CircBuffFlags.NONEMPTY
                    else:
                        self.buffer_state = CircBuffFlags.EMPTY
                else:
                    self.buffer_state = CircBuffFlags.NONEMPTY

        return self.buffer_state

    def get_state(self):
        return self.buffer_state

    def is_overflow(self):
        return self.overflow_flag

    def get_occupancy(self):
        if self.buffer_state == CircBuffFlags.FULL:
            return self.depth
        else:
            return (self.write_ptr - self.read_ptr) % self.depth

    def flush(self):
        self.write_ptr = 0
        self.read_ptr = 0
        self.overflow_flag = False
        self.buffer_state = CircBuffFlags.EMPTY

        return CircBuffFlags.EMPTY

    def write_single(self, data: np.ndarray):
        if data.size > self.width or data.size < self.width:
            return CircBuffFlags.WIDTH_ERROR

        match self.write_overflow_mode:
            case WriteOverFlowMode.LIMIT:
                if self.buffer_state == CircBuffFlags.FULL:
                    return CircBuffFlags.FULL
                else:
                    self.data[self.write_ptr, :] = data
                    return self.__inc_write_ptr()

            case WriteOverFlowMode.FLOW:
                self.data[self.write_ptr, :] = data
                return self.__inc_write_ptr()

            case WriteOverFlowMode.FOLLOW:
                if self.buffer_state == CircBuffFlags.FULL:
                    self.data[self.write_ptr, :] = data
                    self.__inc_read_ptr()
                    return self.__inc_write_ptr()
                else:
                    self.data[self.write_ptr, :] = data
                    return self.__inc_write_ptr()

    def read_single(self):
        match self.write_overflow_mode:
            case WriteOverFlowMode.LIMIT:
                if self.buffer_state == CircBuffFlags.EMPTY:
                    return CircBuffFlags.EMPTY
                else:
                    data = self.data[self.read_ptr, :].copy()
                    self.__inc_read_ptr()
                    return data

            case WriteOverFlowMode.FLOW:
                data = self.data[self.read_ptr, :].copy()
                self.__inc_read_ptr()
                return data

            case WriteOverFlowMode.FOLLOW:
                if self.buffer_state == CircBuffFlags.EMPTY:
                    return CircBuffFlags.EMPTY
                else:
                    data = self.data[self.read_ptr, :].copy()
                    self.__inc_read_ptr()
                    return data

    def dump(self) -> np.ndarray:
        if self.write_overflow_mode == WriteOverFlowMode.FLOW:
            self.overflow_flag = False

        return np.vstack((self.data[self.read_ptr:self.depth+1, :], self.data[0:self.read_ptr, :])).copy()

    def write_batch(self, data: np.ndarray):
        ...

    def read_batch(self, depth: int) -> np.ndarray:
        ...
