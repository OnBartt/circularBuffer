import numpy as np
import pytest
import sys
import os

# přidej kořenový adresář projektu do sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import pytest
from CircularBuffer.CircularBuffer import CircBuff, CircBuffFlags, WriteOverFlowMode  # noqa: E402


@pytest.fixture
def make_buffer():
    def _make(mode=WriteOverFlowMode.LIMIT, width=1, depth=2, dtype=np.int32):
        return CircBuff(width=width, depth=depth, data_type=np.dtype(dtype), write_overflow_mode=mode)
    return _make


@pytest.mark.parametrize("mode", [WriteOverFlowMode.LIMIT, WriteOverFlowMode.FLOW, WriteOverFlowMode.FOLLOW])
def test_initial_state(make_buffer, mode):
    buf = make_buffer(mode, width=2, depth=4)
    assert buf.get_state() == CircBuffFlags.EMPTY
    assert buf.get_occupancy() == 0
    assert not buf.is_overflow()


def test_width_error(make_buffer):
    buf = make_buffer(mode=WriteOverFlowMode.LIMIT, width=2, depth=4)
    wrong_data = np.array([1, 2, 3], dtype=np.int32)  # špatná šířka
    assert buf.write_single(wrong_data) == CircBuffFlags.WIDTH_ERROR


def test_limit_mode_write_and_read(make_buffer):
    buf = make_buffer(mode=WriteOverFlowMode.LIMIT, width=1, depth=2)

    # zapsání prvního prvku
    assert buf.write_single(np.array([10], dtype=np.int32)) == CircBuffFlags.NONEMPTY
    assert buf.get_occupancy() == 1

    # zapsání druhého prvku (buffer se naplní)
    assert buf.write_single(np.array([20], dtype=np.int32)) == CircBuffFlags.FULL
    assert buf.get_occupancy() == 2

    # další zápis je blokován (LIMIT)
    assert buf.write_single(np.array([30], dtype=np.int32)) == CircBuffFlags.FULL

    # čtení prvků
    assert np.array_equal(buf.read_single(), np.array([10], dtype=np.int32))
    assert buf.get_state() == CircBuffFlags.NONEMPTY

    assert np.array_equal(buf.read_single(), np.array([20], dtype=np.int32))
    assert buf.get_state() == CircBuffFlags.EMPTY

    # čtení z prázdného bufferu
    assert buf.read_single() == CircBuffFlags.EMPTY


def test_flow_mode_overwrite_and_overflow_flag(make_buffer):
    buf = make_buffer(mode=WriteOverFlowMode.FLOW, width=1, depth=2)

    buf.write_single(np.array([1], dtype=np.int32))
    buf.write_single(np.array([2], dtype=np.int32))

    # při zápisu třetího prvku dojde k přepsání a overflow_flag se nastaví
    buf.write_single(np.array([3], dtype=np.int32))
    assert buf.is_overflow()

    # čtení po jednom
    v1 = buf.read_single()
    v2 = buf.read_single()
    assert isinstance(v1, np.ndarray)
    assert isinstance(v2, np.ndarray)


def test_follow_mode_overwrite_oldest(make_buffer):
    buf = make_buffer(mode=WriteOverFlowMode.FOLLOW, width=1, depth=2)

    buf.write_single(np.array([1], dtype=np.int32))
    buf.write_single(np.array([2], dtype=np.int32))
    assert buf.get_state() == CircBuffFlags.FULL

    # zápis třetího prvku posune read_ptr dopředu (nejstarší prvek se "zahodí")
    buf.write_single(np.array([3], dtype=np.int32))
    assert buf.get_state() in (CircBuffFlags.FULL, CircBuffFlags.NONEMPTY)

    # čtení prvního (nejstaršího dostupného)
    val = buf.read_single()
    assert isinstance(val, np.ndarray)
    assert val.shape == (1,)


def test_flush_resets_buffer(make_buffer):
    buf = make_buffer(mode=WriteOverFlowMode.LIMIT, width=1, depth=3)
    buf.write_single(np.array([42], dtype=np.int32))
    buf.flush()

    assert buf.get_state() == CircBuffFlags.EMPTY
    assert buf.get_occupancy() == 0
    assert not buf.is_overflow()
    assert buf.read_single() == CircBuffFlags.EMPTY
