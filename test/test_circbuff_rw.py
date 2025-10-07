import pytest
import numpy as np
import sys
import os

# přidej kořenový adresář projektu do sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CircularBuffer.CircularBuffer import CircBuff, CircBuffFlags, WriteOverFlowMode  # noqa: E402


# --- Fixtury ---
@pytest.fixture
def make_buffer():
    def _make(mode=WriteOverFlowMode.LIMIT, width=2, depth=10, dtype=np.int32) -> CircBuff:
        return CircBuff(width=width, depth=depth, data_type=np.dtype(dtype), write_overflow_mode=mode)
    return _make


@pytest.fixture
def sample_data():
    """Několik řádků dat pro zápisy a dávky."""
    return [
        np.array([1, 10], dtype=np.int32),
        np.array([2, 20], dtype=np.int32),
        np.array([3, 30], dtype=np.int32),
        np.array([4, 40], dtype=np.int32),
        np.array([5, 50], dtype=np.int32),
        np.array([6, 60], dtype=np.int32),
        np.array([7, 70], dtype=np.int32),
        np.array([8, 80], dtype=np.int32),
        np.array([9, 90], dtype=np.int32),
        np.array([10, 100], dtype=np.int32),
    ]


# --- init tests ---
@pytest.mark.parametrize("mode", [WriteOverFlowMode.LIMIT, WriteOverFlowMode.FLOW, WriteOverFlowMode.FOLLOW])
def test_initial_state(make_buffer, mode):
    buf = make_buffer(mode, width=2, depth=4)
    assert buf.get_state() == CircBuffFlags.EMPTY
    assert buf.get_occupancy() == 0
    assert not buf.is_overflow()


# --- flush tests ---
@pytest.mark.parametrize("mode", [WriteOverFlowMode.LIMIT, WriteOverFlowMode.FLOW, WriteOverFlowMode.FOLLOW])
def test_flush(make_buffer, sample_data, mode):
    buf = make_buffer(mode, width=2, depth=4)
    buf.data[:2] = np.vstack(sample_data[:2])
    buf.write_ptr = 2
    buf.buffer_state = CircBuffFlags.NONEMPTY
    buf.flush()
    assert buf.buffer_state == CircBuffFlags.EMPTY
    assert buf.write_ptr == 0
    assert buf.read_ptr == 0
    assert buf.overflow_flag is False


# --- write tests ---
@pytest.mark.parametrize("mode", [WriteOverFlowMode.LIMIT, WriteOverFlowMode.FLOW, WriteOverFlowMode.FOLLOW])
def test_write_width_error(make_buffer, sample_data, mode):
    buf = make_buffer(mode, width=1, depth=4)
    state = buf.write_single(sample_data[0])
    assert state == CircBuffFlags.WIDTH_ERROR


# --- write with LIMIT mode ---
def test_write_limit_basic(make_buffer, sample_data):
    buf = make_buffer(mode=WriteOverFlowMode.LIMIT, width=2, depth=4)
    state = buf.write_single(sample_data[0])
    assert buf.get_state() == CircBuffFlags.NONEMPTY
    assert state == CircBuffFlags.NONEMPTY
    assert buf.overflow_flag is False
    assert buf.get_occupancy() == 1
    assert buf.write_ptr == 1
    np.testing.assert_array_equal(buf.data[0], sample_data[0])


def test_write_limit_full(make_buffer, sample_data):
    buf = make_buffer(mode=WriteOverFlowMode.LIMIT, width=2, depth=4)
    state = buf.write_batch(np.vstack(sample_data[:10]))
    assert buf.get_state() == CircBuffFlags.FULL
    assert state == CircBuffFlags.FULL
    assert buf.overflow_flag is False
    assert buf.get_occupancy() == 4
    assert buf.write_ptr == 0
    np.testing.assert_array_equal(buf.data[:4], np.vstack(sample_data[:4]))


def test_write_limit_per_partes(make_buffer, sample_data):
    buf = make_buffer(mode=WriteOverFlowMode.LIMIT, width=2, depth=4)
    state = buf.write_batch(np.vstack(sample_data[:1]))
    assert buf.get_state() == CircBuffFlags.NONEMPTY
    assert state == CircBuffFlags.NONEMPTY
    assert buf.overflow_flag is False
    assert buf.get_occupancy() == 1
    assert buf.write_ptr == 1
    np.testing.assert_array_equal(buf.data[:1], np.vstack(sample_data[:1]))
    state = buf.write_batch(np.vstack(sample_data[:3]))
    assert buf.get_state() == CircBuffFlags.FULL
    assert state == CircBuffFlags.FULL
    assert buf.overflow_flag is False
    assert buf.get_occupancy() == 4
    assert buf.write_ptr == 0
    np.testing.assert_array_equal(buf.data[:4], np.vstack((sample_data[:1], sample_data[:3])))


def test_write_limit_overflow(make_buffer, sample_data):
    buf = make_buffer(mode=WriteOverFlowMode.LIMIT, width=2, depth=4)
    state = buf.write_batch(np.vstack(sample_data[:5]))
    assert buf.get_state() == CircBuffFlags.FULL
    assert state == CircBuffFlags.FULL
    assert buf.overflow_flag is False
    assert buf.get_occupancy() == 4
    assert buf.write_ptr == 0
    np.testing.assert_array_equal(buf.data[:4], np.vstack(sample_data[:4]))


# --- write with FLOW mode ---
def test_write_flow_basic(make_buffer, sample_data):
    buf = make_buffer(mode=WriteOverFlowMode.FLOW, width=2, depth=4)
    state = buf.write_single(sample_data[0])
    assert buf.get_state() == CircBuffFlags.NONEMPTY
    assert state == CircBuffFlags.NONEMPTY
    assert buf.overflow_flag is False
    assert buf.get_occupancy() == 1
    assert buf.write_ptr == 1
    np.testing.assert_array_equal(buf.data[0], sample_data[0])


def test_write_flow_full(make_buffer, sample_data):
    buf = make_buffer(mode=WriteOverFlowMode.FLOW, width=2, depth=4)
    state = buf.write_batch(np.vstack(sample_data[:4]))
    assert buf.get_state() == CircBuffFlags.NONEMPTY
    assert state == CircBuffFlags.NONEMPTY
    assert buf.overflow_flag is True
    assert buf.get_occupancy() == 0
    assert buf.write_ptr == 0
    np.testing.assert_array_equal(buf.data[:4], np.vstack(sample_data[:4]))


def test_write_flow_overflow(make_buffer, sample_data):
    buf = make_buffer(mode=WriteOverFlowMode.FLOW, width=2, depth=4)
    state = buf.write_batch(np.vstack(sample_data[:5]))
    assert buf.get_state() == CircBuffFlags.NONEMPTY
    assert state == CircBuffFlags.NONEMPTY
    assert buf.overflow_flag is True
    assert buf.get_occupancy() == 1
    assert buf.write_ptr == 1
    np.testing.assert_array_equal(buf.data[:], np.vstack((np.array([5, 50]), np.array([2, 20]),
                                                          np.array([3, 30]), np.array([4, 40]),)))


def test_write_flow_over_overflow(make_buffer, sample_data):
    buf = make_buffer(mode=WriteOverFlowMode.FLOW, width=2, depth=4)
    state = buf.write_batch(np.vstack(sample_data[:]))
    assert buf.get_state() == CircBuffFlags.NONEMPTY
    assert state == CircBuffFlags.NONEMPTY
    assert buf.read_ptr == 0
    assert buf.write_ptr == 2
    assert buf.get_occupancy() == 2
    assert buf.overflow_flag is True
    np.testing.assert_array_equal(buf.data[:4], np.vstack((np.array([9, 90]), np.array([10, 100]),
                                                           np.array([7, 70]), np.array([8, 80]))))


# --- write with FOLLOW mode ---
def test_write_follow_basic(make_buffer, sample_data):
    buf = make_buffer(mode=WriteOverFlowMode.FOLLOW, width=2, depth=4)
    state = buf.write_single(sample_data[0])
    assert buf.get_state() == CircBuffFlags.NONEMPTY
    assert state == CircBuffFlags.NONEMPTY
    assert buf.overflow_flag is False
    assert buf.get_occupancy() == 1
    assert buf.write_ptr == 1
    np.testing.assert_array_equal(buf.data[0], sample_data[0])


def test_write_follow_full(make_buffer, sample_data):
    buf = make_buffer(mode=WriteOverFlowMode.FOLLOW, width=2, depth=4)
    state = buf.write_batch(np.vstack(sample_data[:4]))
    assert buf.get_state() == CircBuffFlags.FULL
    assert state == CircBuffFlags.FULL
    assert buf.overflow_flag is True
    assert buf.get_occupancy() == 4
    assert buf.write_ptr == 0
    np.testing.assert_array_equal(buf.data[:4], np.vstack(sample_data[:4]))


def test_write_follow_overflow(make_buffer, sample_data):
    buf = make_buffer(mode=WriteOverFlowMode.FOLLOW, width=2, depth=4)
    state = buf.write_batch(np.vstack(sample_data[:5]))
    assert buf.get_state() == CircBuffFlags.FULL
    assert state == CircBuffFlags.FULL
    assert buf.overflow_flag is True
    assert buf.get_occupancy() == 4
    assert buf.write_ptr == 1
    assert buf.read_ptr == 1
    np.testing.assert_array_equal(buf.data[:], np.vstack((np.array([5, 50]), np.array([2, 20]),
                                                          np.array([3, 30]), np.array([4, 40]),)))


def test_write_follow_over_overflow(make_buffer, sample_data):
    buf = make_buffer(mode=WriteOverFlowMode.FOLLOW, width=2, depth=4)
    state = buf.write_batch(np.vstack(sample_data[:]))
    assert buf.get_state() == CircBuffFlags.FULL
    assert state == CircBuffFlags.FULL
    assert buf.read_ptr == 2
    assert buf.write_ptr == 2
    assert buf.get_occupancy() == 4
    assert buf.overflow_flag is True
    np.testing.assert_array_equal(buf.data[:4], np.vstack((np.array([9, 90]), np.array([10, 100]),
                                                           np.array([7, 70]), np.array([8, 80]))))


# --- read tests ---
@pytest.mark.parametrize("mode", [WriteOverFlowMode.LIMIT, WriteOverFlowMode.FLOW, WriteOverFlowMode.FOLLOW])
def test_full_dump(make_buffer, sample_data, mode):
    buf = make_buffer(mode, width=2, depth=4)
    buf.data = np.vstack(sample_data[:4])
    if mode == WriteOverFlowMode.FLOW:
        buf.buffer_state = CircBuffFlags.NONEMPTY
    else:
        buf.buffer_state = CircBuffFlags.FULL

    data = buf.dump()
    assert buf.buffer_state == CircBuffFlags.EMPTY
    np.testing.assert_array_equal(data, np.vstack(sample_data[:4]))


@pytest.mark.parametrize("mode", [WriteOverFlowMode.LIMIT, WriteOverFlowMode.FLOW, WriteOverFlowMode.FOLLOW])
def test_partial_dump(make_buffer, sample_data, mode):
    buf = make_buffer(mode, width=2, depth=4)
    buf.data[:2] = np.vstack(sample_data[:2])
    buf.buffer_state = CircBuffFlags.NONEMPTY
    buf.write_ptr = 2
    data = buf.dump()
    assert buf.buffer_state == CircBuffFlags.EMPTY
    assert buf.write_ptr == 2
    np.testing.assert_array_equal(data, np.vstack((sample_data[:2], np.array([0, 0]), np.array([0, 0]))))


# --- read with LIMIT mode ---
def test_read_limit_baisc(make_buffer, sample_data):
    buf = make_buffer(WriteOverFlowMode.LIMIT, width=2, depth=4)
    buf.data[:2] = np.vstack(sample_data[:2])
    buf.buffer_state = CircBuffFlags.NONEMPTY
    buf.write_ptr = 2
    data = buf.read_single()
    assert buf.buffer_state == CircBuffFlags.NONEMPTY
    assert buf.get_state() == CircBuffFlags.NONEMPTY
    assert buf.read_ptr == 1
    np.testing.assert_array_equal(data, np.vstack((sample_data[:1])))
    data = buf.read_single()
    assert buf.buffer_state == CircBuffFlags.EMPTY
    assert buf.get_state() == CircBuffFlags.EMPTY
    assert buf.read_ptr == 2
    np.testing.assert_array_equal(data, np.vstack(sample_data[1:2]))


def test_read_limit_batch(make_buffer, sample_data):
    buf = make_buffer(WriteOverFlowMode.LIMIT, width=2, depth=4)
    buf.data[:3] = np.vstack(sample_data[:3])
    buf.buffer_state = CircBuffFlags.NONEMPTY
    buf.write_ptr = 3
    data = buf.read_batch(1)
    assert buf.buffer_state == CircBuffFlags.NONEMPTY
    assert buf.get_state() == CircBuffFlags.NONEMPTY
    assert buf.read_ptr == 1
    np.testing.assert_array_equal(data, np.vstack((sample_data[:1])))
    data = buf.read_batch(2)
    assert buf.buffer_state == CircBuffFlags.EMPTY
    assert buf.get_state() == CircBuffFlags.EMPTY
    assert buf.read_ptr == 3
    np.testing.assert_array_equal(data, np.vstack((sample_data[1:3])))


# read with FLOW mode ---
def test_read_flow_basic(make_buffer, sample_data):
    buf = make_buffer(WriteOverFlowMode.FLOW, width=2, depth=4)
    buf.data[:2] = np.vstack(sample_data[:2])
    buf.buffer_state = CircBuffFlags.NONEMPTY
    buf.write_ptr = 2
    data = buf.read_single()
    assert buf.buffer_state == CircBuffFlags.NONEMPTY
    assert buf.get_state() == CircBuffFlags.NONEMPTY
    assert buf.read_ptr == 1
    np.testing.assert_array_equal(data, np.vstack((sample_data[:1])))
    data = buf.read_single()
    # with FLOW the buffer is never empty (only after flush or dump)
    assert buf.buffer_state == CircBuffFlags.NONEMPTY
    assert buf.get_state() == CircBuffFlags.NONEMPTY
    assert buf.read_ptr == 2
    np.testing.assert_array_equal(data, np.vstack(sample_data[1:2]))


def test_read_flow_batch(make_buffer, sample_data):
    buf = make_buffer(WriteOverFlowMode.FLOW, width=2, depth=4)
    buf.data[:3] = np.vstack(sample_data[:3])
    buf.buffer_state = CircBuffFlags.NONEMPTY
    buf.write_ptr = 3
    data = buf.read_batch(1)
    assert buf.buffer_state == CircBuffFlags.NONEMPTY
    assert buf.get_state() == CircBuffFlags.NONEMPTY
    assert buf.read_ptr == 1
    np.testing.assert_array_equal(data, np.vstack((sample_data[:1])))
    data = buf.read_batch(2)
    assert buf.buffer_state == CircBuffFlags.NONEMPTY
    assert buf.get_state() == CircBuffFlags.NONEMPTY
    assert buf.read_ptr == 3
    np.testing.assert_array_equal(data, np.vstack((sample_data[1:3])))


def test_read_flow_all(make_buffer, sample_data):
    buf = make_buffer(WriteOverFlowMode.FLOW, width=2, depth=4)
    buf.data[:4] = np.vstack(sample_data[:4])
    buf.buffer_state = CircBuffFlags.NONEMPTY
    buf.read_ptr = 3
    data = buf.read_batch(4)
    # with FLOW mode only flush empties the buffer
    assert buf.buffer_state == CircBuffFlags.NONEMPTY
    assert buf.get_state() == CircBuffFlags.NONEMPTY
    assert buf.read_ptr == 3
    np.testing.assert_array_equal(data, np.vstack((sample_data[3:4], sample_data[:3])))


def test_read_flow_wraparound(make_buffer, sample_data):
    buf = make_buffer(WriteOverFlowMode.FLOW, width=2, depth=4)
    buf.data[:4] = np.vstack(sample_data[:4])
    buf.buffer_state = CircBuffFlags.NONEMPTY
    buf.read_ptr = 3
    data = buf.read_batch(2)
    assert buf.buffer_state == CircBuffFlags.NONEMPTY
    assert buf.get_state() == CircBuffFlags.NONEMPTY
    assert buf.read_ptr == 1
    np.testing.assert_array_equal(data, np.vstack((sample_data[3:4], sample_data[:1])))


# read with FOLLOW mode ---
def test_read_follow_basic(make_buffer, sample_data):
    buf = make_buffer(WriteOverFlowMode.FOLLOW, width=2, depth=4)
    buf.data[:2] = np.vstack(sample_data[:2])
    buf.buffer_state = CircBuffFlags.NONEMPTY
    buf.write_ptr = 2
    data = buf.read_single()
    assert buf.buffer_state == CircBuffFlags.NONEMPTY
    assert buf.get_state() == CircBuffFlags.NONEMPTY
    assert buf.read_ptr == 1
    np.testing.assert_array_equal(data, np.vstack((sample_data[:1])))
    data = buf.read_single()
    assert buf.buffer_state == CircBuffFlags.EMPTY
    assert buf.get_state() == CircBuffFlags.EMPTY
    assert buf.read_ptr == 2
    np.testing.assert_array_equal(data, np.vstack(sample_data[1:2]))


def test_read_follow_batch(make_buffer, sample_data):
    buf = make_buffer(WriteOverFlowMode.FOLLOW, width=2, depth=4)
    buf.data[:3] = np.vstack(sample_data[:3])
    buf.buffer_state = CircBuffFlags.NONEMPTY
    buf.write_ptr = 3
    data = buf.read_batch(1)
    assert buf.buffer_state == CircBuffFlags.NONEMPTY
    assert buf.get_state() == CircBuffFlags.NONEMPTY
    assert buf.read_ptr == 1
    np.testing.assert_array_equal(data, np.vstack((sample_data[:1])))
    data = buf.read_batch(2)
    assert buf.buffer_state == CircBuffFlags.EMPTY
    assert buf.get_state() == CircBuffFlags.EMPTY
    assert buf.read_ptr == 3
    np.testing.assert_array_equal(data, np.vstack((sample_data[1:3])))


def test_read_follow_wraparound(make_buffer, sample_data):
    buf = make_buffer(WriteOverFlowMode.FOLLOW, width=2, depth=4)
    buf.data[:4] = np.vstack(sample_data[:4])
    buf.buffer_state = CircBuffFlags.FULL
    buf.read_ptr = 3
    data = buf.read_batch(2)
    assert buf.buffer_state == CircBuffFlags.NONEMPTY
    assert buf.get_state() == CircBuffFlags.NONEMPTY
    assert buf.read_ptr == 1
    np.testing.assert_array_equal(data, np.vstack((sample_data[3:4], sample_data[:1])))


# --- write/read tests ---
# --- write/read with LIMIT
def test_read_write_limit_basic(make_buffer, sample_data):
    buf = make_buffer(WriteOverFlowMode.LIMIT, width=2, depth=4)
    buf_state = buf.write_single(sample_data[2])
    assert buf.get_occupancy() == 1
    data = buf.read_single()
    assert buf.write_ptr == 1
    assert buf.read_ptr == 1
    assert buf_state == CircBuffFlags.NONEMPTY
    assert buf.get_state() == CircBuffFlags.EMPTY
    assert buf.get_occupancy() == 0
    np.testing.assert_array_equal(data, [sample_data[2]])
    buf_state = buf.write_batch(np.vstack(sample_data[:4]))
    assert buf.get_occupancy() == 4
    data = buf.read_batch(4)
    assert buf_state == CircBuffFlags.FULL
    assert buf.write_ptr == 1
    assert buf.read_ptr == 1
    assert buf.get_state() == CircBuffFlags.EMPTY
    np.testing.assert_array_equal(data, np.vstack((sample_data[:4])))


def test_read_write_limit_write_over_limit(make_buffer, sample_data):
    buf = make_buffer(WriteOverFlowMode.LIMIT, width=2, depth=4)
    # Zapisu dva prvky
    buf_state = buf.write_batch(np.vstack(sample_data[:2]))
    # Zapisu dalsi tri prvky. meli by se zapsat jen prvni dva
    buf_state = buf.write_batch(np.vstack(sample_data[:3]))
    assert buf_state == CircBuffFlags.FULL
    assert buf.write_ptr == 0
    # Zkusim zapsat dalsi jeden prvek. Nic by se nemelo stat.
    buf_state = buf.write_batch(np.vstack(sample_data[:1]))
    np.testing.assert_array_equal(buf.data, np.vstack((sample_data[:2], sample_data[:2])))
    data = buf.read_batch(4)
    assert buf.get_state() == CircBuffFlags.EMPTY
    np.testing.assert_array_equal(data, np.vstack((sample_data[:2], sample_data[:2])))


# --- write/read with FLOW
def test_read_write_flow_basic(make_buffer, sample_data):
    buf = make_buffer(WriteOverFlowMode.FLOW, width=2, depth=4)
    buf_state = buf.write_single(sample_data[2])
    assert buf.get_occupancy() == 1
    data = buf.read_single()
    assert buf.write_ptr == 1
    assert buf.read_ptr == 1
    assert buf_state == CircBuffFlags.NONEMPTY
    assert buf.get_state() == CircBuffFlags.NONEMPTY
    assert buf.get_occupancy() == 0
    np.testing.assert_array_equal(data, [sample_data[2]])
    buf_state = buf.write_batch(np.vstack(sample_data[:4]))
    assert buf.get_occupancy() == 0
    data = buf.read_batch(4)
    assert buf_state == CircBuffFlags.NONEMPTY
    assert buf.write_ptr == 1
    assert buf.read_ptr == 1
    assert buf.get_state() == CircBuffFlags.NONEMPTY
    np.testing.assert_array_equal(data, np.vstack((sample_data[:4])))


def test_read_write_flow_overlap(make_buffer, sample_data):
    buf = make_buffer(WriteOverFlowMode.FLOW, width=2, depth=4)
    buf_state = buf.write_batch(np.vstack(sample_data[:6]))
    assert buf_state == CircBuffFlags.NONEMPTY
    assert buf.is_overflow() is True
    assert buf.write_ptr == 2
    assert buf.read_ptr == 0
    data = buf.read_batch(4)
    np.testing.assert_array_equal(data, np.vstack((sample_data[4:6], sample_data[2:4])))


# --- write/read with FOLLOW
def test_read_write_follow_basic(make_buffer, sample_data):
    buf = make_buffer(WriteOverFlowMode.FOLLOW, width=2, depth=4)
    buf_state = buf.write_batch(np.vstack((sample_data[:2])))
    assert buf_state == CircBuffFlags.NONEMPTY
    assert buf.write_ptr == 2
    buf_state = buf.write_batch(np.vstack((sample_data[:2])))
    assert buf_state == CircBuffFlags.FULL
    assert buf.is_overflow() is True
    assert buf.read_ptr == buf.write_ptr
    assert buf.write_ptr == 0
    data = buf.read_batch(4)
    assert buf.get_state() == CircBuffFlags.EMPTY
    np.testing.assert_array_equal(data, np.vstack((sample_data[0:2], sample_data[0:2])))


def test_read_write_follow_following(make_buffer, sample_data):
    buf = make_buffer(WriteOverFlowMode.FOLLOW, width=2, depth=4)
    buf_state = buf.write_batch(np.vstack((sample_data[:5])))
    assert buf_state == CircBuffFlags.FULL
    assert buf.is_overflow() is True
    assert buf.write_ptr == 1
    assert buf.read_ptr == buf.write_ptr
    data = buf.read_batch(3)
    assert buf.get_state() == CircBuffFlags.NONEMPTY
    np.testing.assert_array_equal(data, np.vstack((sample_data[1:4])))
    assert buf.is_overflow() is False


def test_read_write_follow_following_full(make_buffer, sample_data):
    buf = make_buffer(WriteOverFlowMode.FOLLOW, width=2, depth=4)
    buf_state = buf.write_batch(np.vstack((sample_data[:6])))
    assert buf_state == CircBuffFlags.FULL
    assert buf.is_overflow() is True
    assert buf.write_ptr == 2
    assert buf.read_ptr == buf.write_ptr
    data = buf.read_batch(4)
    assert buf.get_state() == CircBuffFlags.EMPTY
    np.testing.assert_array_equal(data, np.vstack((sample_data[2:6])))
    assert buf.is_overflow() is False
