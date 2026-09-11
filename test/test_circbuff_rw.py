import pytest
import numpy as np
import sys
import os

# add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CircularBuffer import CircBuff, CircBuffEmpty, CircBuffFlags, WriteOverFlowMode  # noqa: E402


# --- Fixtures ---
@pytest.fixture
def make_buffer():
    def _make(mode=WriteOverFlowMode.LIMIT, width=2, depth=10, dtype=np.int32) -> CircBuff:
        return CircBuff(width=width, depth=depth, data_type=np.dtype(dtype), write_overflow_mode=mode)
    return _make


@pytest.fixture
def sample_data():
    """Some dummy data."""
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


def test_flush_and_reuse(make_buffer, sample_data):
    for mode in WriteOverFlowMode:
        buf = make_buffer(mode, width=2, depth=4)
        buf.write_batch(np.vstack(sample_data[:3]))
        buf.flush()
        buf.write_batch(np.vstack(sample_data[5:8]))
        data = buf.read_batch(3)
        np.testing.assert_array_equal(data, np.vstack(sample_data[5:8]))


# --- write tests ---
@pytest.mark.parametrize("mode", [WriteOverFlowMode.LIMIT, WriteOverFlowMode.FLOW, WriteOverFlowMode.FOLLOW])
def test_write_width_error(make_buffer, sample_data, mode):
    buf = make_buffer(mode, width=1, depth=4)
    # writing width-two data into a width-one buffer.
    # match= matters: without the width check numpy would raise a ValueError of its own
    # about the mismatched shape and the test would pass without verifying anything.
    with pytest.raises(ValueError, match="expects"):
        buf.write_single(sample_data[0])
    with pytest.raises(ValueError, match="expects"):
        buf.write_batch(np.vstack(sample_data[:2]))


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
    # only the first 4 items get written
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


def test_write_batch_multiple_of_depth(make_buffer, sample_data):
    buf = make_buffer(WriteOverFlowMode.FLOW, width=2, depth=4)
    repeated = np.vstack(sample_data[:4] * 2)
    buf.write_batch(repeated)
    np.testing.assert_array_equal(buf.data, np.vstack(sample_data[:4]))


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


def test_dump_with_nonzero_read_ptr(make_buffer, sample_data):
    buf = make_buffer(WriteOverFlowMode.LIMIT, width=2, depth=5)
    buf.write_batch(np.vstack(sample_data[:5]))
    buf.read_ptr = 2
    buf.buffer_state = CircBuffFlags.FULL

    data = buf.dump()
    expected = np.vstack((sample_data[2:5], sample_data[:2]))
    np.testing.assert_array_equal(data, expected)


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


# --- read with FLOW mode ---
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


# --- read with FOLLOW mode ---
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


def test_read_follow_batch_cross_boundary(make_buffer, sample_data):
    buf = make_buffer(WriteOverFlowMode.FOLLOW, width=2, depth=4)
    buf.write_batch(np.vstack(sample_data[:4]))
    buf.read_ptr = 3
    data = buf.read_batch(2)
    expected = np.vstack((sample_data[3:4], sample_data[:1]))
    np.testing.assert_array_equal(data, expected)


def test_read_follow_overflow_flag_resets_after_read(make_buffer, sample_data):
    buf = make_buffer(WriteOverFlowMode.FOLLOW, width=2, depth=4)
    buf.write_batch(np.vstack(sample_data[:6]))
    assert buf.is_overflow() is True
    _ = buf.read_batch(2)
    assert buf.is_overflow() is False


# --- write/read tests ---
# --- write/read with LIMIT ---
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
    buf_state = buf.write_batch(np.vstack(sample_data[:2]))
    buf_state = buf.write_batch(np.vstack(sample_data[:3]))
    assert buf_state == CircBuffFlags.FULL
    assert buf.write_ptr == 0
    buf_state = buf.write_batch(np.vstack(sample_data[:1]))
    np.testing.assert_array_equal(buf.data, np.vstack((sample_data[:2], sample_data[:2])))
    data = buf.read_batch(4)
    assert buf.get_state() == CircBuffFlags.EMPTY
    np.testing.assert_array_equal(data, np.vstack((sample_data[:2], sample_data[:2])))


# --- write/read with FLOW ---
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


# --- write/read with FOLLOW ---
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


# --- regression tests ---
@pytest.mark.parametrize("mode", [WriteOverFlowMode.FLOW, WriteOverFlowMode.FOLLOW])
def test_write_batch_partial_wrap(make_buffer, sample_data, mode):
    """A batch wrapping past the end of the array must continue at the start in full, not as one repeated row."""
    buf = make_buffer(mode, width=2, depth=4)
    buf.write_batch(np.vstack((sample_data[:2])))
    assert buf.write_ptr == 2

    buf.write_batch(np.vstack((sample_data[2:5])))

    assert buf.write_ptr == 1
    np.testing.assert_array_equal(buf.data, np.vstack((sample_data[4], sample_data[1],
                                                       sample_data[2], sample_data[3])))


def test_read_limit_batch_wraparound(make_buffer, sample_data):
    """A LIMIT read must also collect content split across the end of the array."""
    buf = make_buffer(WriteOverFlowMode.LIMIT, width=2, depth=4)
    buf.data[3] = sample_data[0]
    buf.data[0] = sample_data[1]
    buf.read_ptr = 3
    buf.write_ptr = 1
    buf.buffer_state = CircBuffFlags.NONEMPTY
    assert buf.get_occupancy() == 2

    data = buf.read_batch(3)

    np.testing.assert_array_equal(data, np.vstack((sample_data[:2])))
    assert buf.read_ptr == 1
    assert buf.get_state() == CircBuffFlags.EMPTY


def test_read_limit_full_buffer_readable(make_buffer, sample_data):
    """All items must be readable from a full buffer, not just the first (rd == wr also means FULL)."""
    buf = make_buffer(WriteOverFlowMode.LIMIT, width=2, depth=4)
    for item in sample_data[:4]:
        buf.write_single(item)
    assert buf.get_state() == CircBuffFlags.FULL
    assert buf.read_ptr == buf.write_ptr

    for i in range(4):
        data = buf.read_single()
        np.testing.assert_array_equal(data, np.vstack(([sample_data[i]])))

    assert buf.get_state() == CircBuffFlags.EMPTY
    with pytest.raises(CircBuffEmpty):
        buf.read_single()


@pytest.mark.parametrize("kw", [{"width": 0}, {"depth": 0}, {"width": -1}, {"depth": -1}])
def test_init_rejects_invalid_dimensions(kw):
    with pytest.raises(ValueError):
        CircBuff(**{"width": 2, "depth": 4, "data_type": np.int32,
                    "write_overflow_mode": WriteOverFlowMode.LIMIT, **kw})


def test_invalid_mode_raises(make_buffer):
    buf = make_buffer(WriteOverFlowMode.LIMIT, width=2, depth=4)
    buf.write_overflow_mode = "LIMIT"          # a string, not the enum
    with pytest.raises(ValueError):
        buf.write_single(np.array([1, 10], dtype=np.int32))


@pytest.mark.parametrize("mode", [WriteOverFlowMode.LIMIT, WriteOverFlowMode.FLOW, WriteOverFlowMode.FOLLOW])
def test_read_from_empty_raises(make_buffer, mode):
    """Reading an empty buffer raises, rather than returning a flag mistakable for data."""
    buf = make_buffer(mode, width=2, depth=4)
    assert buf.get_state() == CircBuffFlags.EMPTY

    with pytest.raises(CircBuffEmpty):
        buf.read_single()
    with pytest.raises(CircBuffEmpty):
        buf.read_batch(2)


@pytest.mark.parametrize("shift", [0, 1, 2, 3])
def test_limit_detects_full_at_any_read_ptr(make_buffer, sample_data, shift):
    """LIMIT must detect a full buffer even when read_ptr is not zero, and refuse further writes."""
    buf = make_buffer(WriteOverFlowMode.LIMIT, width=2, depth=4)
    for i in range(shift):                       # move read_ptr without changing occupancy
        buf.write_single(sample_data[i])
        buf.read_single()
    assert buf.get_occupancy() == 0

    for i in range(4):                           # exactly depth items -> full
        assert buf.write_single(sample_data[i]) != CircBuffFlags.FULL or i == 3
    assert buf.get_state() == CircBuffFlags.FULL
    assert buf.get_occupancy() == 4

    snapshot = buf.data.copy()
    assert buf.write_single(sample_data[9]) == CircBuffFlags.FULL
    np.testing.assert_array_equal(buf.data, snapshot)


def test_follow_single_write_moves_read_ptr_on_overwrite(make_buffer, sample_data):
    """In FOLLOW read_ptr must step aside on single writes too, or reads return the newest instead of the oldest."""
    buf = make_buffer(WriteOverFlowMode.FOLLOW, width=2, depth=4)
    for item in sample_data[:4]:                 # fill it up
        buf.write_single(item)
    assert buf.get_state() == CircBuffFlags.FULL
    assert buf.read_ptr == 0

    buf.write_single(sample_data[4])             # overwrite the oldest item
    assert buf.read_ptr == 1, "read_ptr must follow write_ptr on single writes too"
    assert buf.get_state() == CircBuffFlags.FULL

    received = [buf.read_single() for _ in range(4)]
    np.testing.assert_array_equal(np.vstack(received),
                                  np.vstack((sample_data[1], sample_data[2],
                                             sample_data[3], sample_data[4])))


def test_follow_single_matches_batch(make_buffer, sample_data):
    """Writing one by one and in a batch must end in the same state."""
    one_by_one = make_buffer(WriteOverFlowMode.FOLLOW, width=2, depth=4)
    for item in sample_data[:6]:
        one_by_one.write_single(item)

    batched = make_buffer(WriteOverFlowMode.FOLLOW, width=2, depth=4)
    batched.write_batch(np.vstack(sample_data[:6]))

    assert (one_by_one.read_ptr, one_by_one.write_ptr) == (batched.read_ptr, batched.write_ptr)
    assert one_by_one.get_state() == batched.get_state()
    np.testing.assert_array_equal(one_by_one.data, batched.data)


# --- model-based test: batch write against repeated single writes ---
def _buffer_in_state(make_buffer, mode, depth, writes, reads):
    """Bring the buffer into the given state using individual operations."""
    buf = make_buffer(mode, width=1, depth=depth)
    for i in range(writes):
        buf.write_single(np.array([100 + i], dtype=np.int32))
    for _ in range(reads):
        try:
            buf.read_single()
        except CircBuffEmpty:
            pass
    return buf


@pytest.mark.parametrize("mode", list(WriteOverFlowMode))
@pytest.mark.parametrize("depth", [2, 3, 4, 5, 6, 7, 8])
def test_write_batch_matches_repeated_write_single(make_buffer, mode, depth):
    """write_batch(n) must end in the same state as n x write_single.

    Walks every combination of starting buffer state and batch size, including batches
    longer than the buffer. This kind of comparison is what exposed the bugs at the array
    boundary, which hand-written cases are hard to aim at.
    """
    for writes in range(depth + 1):
        for reads in range(depth + 1):
            for n in range(1, 2 * depth + 1):
                batch = np.arange(1, n + 1, dtype=np.int32).reshape(-1, 1)

                batched = _buffer_in_state(make_buffer, mode, depth, writes, reads)
                one_by_one = _buffer_in_state(make_buffer, mode, depth, writes, reads)

                batched_state = batched.write_batch(batch)
                single_state = None
                for row in batch:
                    single_state = one_by_one.write_single(row)

                context = (
                    f"\nmode={mode.name} depth={depth} writes={writes} reads={reads} batch={n}"
                    f"\n  batched   : rd={batched.read_ptr} wr={batched.write_ptr} "
                    f"data={batched.data.ravel()} state={batched_state} overflow={batched.is_overflow()}"
                    f"\n  one by one: rd={one_by_one.read_ptr} wr={one_by_one.write_ptr} "
                    f"data={one_by_one.data.ravel()} state={single_state} overflow={one_by_one.is_overflow()}"
                )
                assert (batched.read_ptr, batched.write_ptr) == (one_by_one.read_ptr, one_by_one.write_ptr), context
                assert batched_state == single_state, context
                assert batched.is_overflow() == one_by_one.is_overflow(), context
                np.testing.assert_array_equal(batched.data, one_by_one.data, err_msg=context)
