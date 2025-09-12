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

    def __inc_write_ptr(self) -> CircBuffFlags:
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

    def __inc_read_ptr(self) -> CircBuffFlags:
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

    def get_state(self) -> CircBuffFlags:
        return self.buffer_state

    def is_overflow(self) -> bool:
        return self.overflow_flag

    def get_occupancy(self) -> int:
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

    def read_single(self) -> np.ndarray | CircBuffFlags:
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
        read_bu = self.read_ptr
        self.overflow_flag = False
        if self.write_overflow_mode != WriteOverFlowMode.FLOW:
            self.read_ptr = 0
            self.write_ptr = 0
            self.buffer_state = CircBuffFlags.EMPTY

        return np.vstack((self.data[read_bu:self.depth+1, :], self.data[0:read_bu, :])).copy()

    def write_batch(self, data: np.ndarray):
        if data.shape[1] > self.width or data.shape[1] < self.width:
            return CircBuffFlags.WIDTH_ERROR

        match self.write_overflow_mode:
            case WriteOverFlowMode.LIMIT:
                if data.shape[0] > self.depth-self.read_ptr:
                    self.data[self.read_ptr:self.depth, :] = data[0:self.depth-self.read_ptr, :]
                    self.write_ptr = 0
                    self.buffer_state = CircBuffFlags.FULL
                    return self.buffer_state
                else:
                    self.data[self.read_ptr:self.read_ptr+data.shape[0], :] = data.copy()
                    self.write_ptr += data.shape[0]
                    self.buffer_state = CircBuffFlags.NONEMPTY
                    return self.buffer_state

            case WriteOverFlowMode.FLOW:
                if data.shape[0] > self.depth:
                    data_tmp = data[-self.depth:, :]
                else:
                    data_tmp = data

                if data_tmp.shape[0] >= self.depth - self.write_ptr:
                    self.data[self.write_ptr:self.depth, :] = data_tmp[0:self.depth-self.write_ptr, :].copy()
                    self.data[0:-(self.depth-self.write_ptr)] = data_tmp[self.depth-self.write_ptr-1, :].copy()
                    self.write_ptr = (self.write_ptr + data_tmp.shape[0]) % self.depth
                else:
                    self.data[self.write_ptr:data_tmp.shape[0], :] = data_tmp.copy()
                    self.write_ptr += data_tmp.shape[0]

                self.buffer_state = CircBuffFlags.NONEMPTY
                return self.buffer_state

            case WriteOverFlowMode.FOLLOW:
                ...

    def read_batch(self, depth: int) -> np.ndarray | CircBuffFlags:
        if self.buffer_state == CircBuffFlags.EMPTY:
            return CircBuffFlags.EMPTY

        match self.write_overflow_mode:
            case WriteOverFlowMode.LIMIT:
                if depth > (occupancy := self.get_occupancy()):
                    data = self.data[self.read_ptr:self.read_ptr+occupancy, :]
                    self.read_ptr += occupancy
                elif depth >= self.depth - self.read_ptr:
                    data = self.data[self.read_ptr:self.depth, :]
                    self.read_ptr = 0
                    self.buffer_state = CircBuffFlags.EMPTY
                else:
                    data = self.data[self.read_ptr:self.read_ptr+depth, :]
                    self.read_ptr += depth
                    self.buffer_state = CircBuffFlags.NONEMPTY

                return data.copy()

            case WriteOverFlowMode.FLOW:
                if depth >= self.depth:
                    return self.dump()

                if depth > self.depth - self.read_ptr:
                    tmp_data = np.vstack((self.data[self.read_ptr:self.depth, :],
                                          self.data[0:depth-(self.depth - self.read_ptr), :]))
                    self.read_ptr = (self.read_ptr + depth) % self.depth
                    # self.buffer_state = CircBuffFlags.NONEMPTY
                else:
                    tmp_data = self.data[self.read_ptr:self.read_ptr+depth, :]
                    self.read_ptr += depth

                return tmp_data.copy()

            case WriteOverFlowMode.FOLLOW:
                return CircBuffFlags.NONEMPTY
