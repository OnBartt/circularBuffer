import pytest
import numpy as np
import sys
import os

# přidej kořenový adresář projektu do sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CircularBuffer.CircularBuffer import CircBuff, CircBuffFlags, WriteOverFlowMode  # noqa: E402


# --- Fixtury ---
@pytest.fixture(params=[WriteOverFlowMode.LIMIT, WriteOverFlowMode.FLOW, WriteOverFlowMode.FOLLOW])
def buffer(request):
    """Fixture pro všechny režimy zápisu."""
    width = 2
    depth = 4
    return CircBuff(width=width, depth=depth, data_type=np.dtype(np.float32), write_overflow_mode=request.param)


@pytest.fixture
def sample_data():
    """Několik řádků dat pro zápisy a dávky."""
    return [
        np.array([1, 10], dtype=np.float32),
        np.array([2, 20], dtype=np.float32),
        np.array([3, 30], dtype=np.float32),
        np.array([4, 40], dtype=np.float32),
        np.array([5, 50], dtype=np.float32),
        np.array([6, 60], dtype=np.float32),
    ]


# --- Testy dump ---
def test_dump_initially_empty(buffer):
    dump = buffer.dump()
    assert dump.shape == (buffer.depth, buffer.width)
    np.testing.assert_array_equal(dump, np.zeros_like(dump))


def test_dump_after_writes(buffer, sample_data):
    buffer.write_batch(np.vstack(sample_data[:3]))
    dump = buffer.dump()
    # dump by měl obsahovat aspoň ta tři data, ostatní mohou zůstat nulová
    assert np.any(dump[:3] != 0)


# --- Testy write_batch ---
def test_write_batch_fills_and_overflows_limit(sample_data):
    buf = CircBuff(width=2, depth=3, data_type=np.dtype(np.float32), write_overflow_mode=WriteOverFlowMode.LIMIT)
    result = buf.write_batch(np.vstack(sample_data[:5]))
    # V LIMIT se vejdou jen 3 záznamy
    assert buf.get_occupancy() == 3
    assert result == CircBuffFlags.FULL
    # read_batch musí vrátit jen ty první tři
    out = buf.read_batch(3)
    np.testing.assert_array_equal(out, np.vstack(sample_data[:3]))


def test_write_batch_overwrites_flow(sample_data):
    buf = CircBuff(width=2, depth=3, data_type=np.dtype(np.float32), write_overflow_mode=WriteOverFlowMode.FLOW)
    buf.write_batch(np.vstack(sample_data[:5]))
    # Do FLOW se vejdou poslední tři
    out = buf.read_batch(3)
    np.testing.assert_array_equal(out, np.vstack(sample_data[2:5]))


def test_write_batch_follow_moves_read_ptr(sample_data):
    buf = CircBuff(width=2, depth=3, data_type=np.dtype(np.float32), write_overflow_mode=WriteOverFlowMode.FOLLOW)
    buf.write_batch(np.vstack(sample_data[:5]))
    # V FOLLOW musí read_ptr "dohonit" write_ptr → opět zůstanou poslední tři
    out = buf.read_batch(3)
    np.testing.assert_array_equal(out, np.vstack(sample_data[2:5]))


# --- Testy read_batch ---
def test_read_batch_from_empty_limit_follow():
    for mode in [WriteOverFlowMode.LIMIT, WriteOverFlowMode.FOLLOW]:
        buf = CircBuff(width=2, depth=3, data_type=np.dtype(np.float32), write_overflow_mode=mode)
        result = buf.read_batch(2)
        assert result == CircBuffFlags.EMPTY


def test_read_batch_partial_read(sample_data):
    buf = CircBuff(width=2, depth=4, data_type=np.dtype(np.float32), write_overflow_mode=WriteOverFlowMode.LIMIT)
    buf.write_batch(np.vstack(sample_data[:3]))
    out = buf.read_batch(2)
    np.testing.assert_array_equal(out, np.vstack(sample_data[:2]))
    # buffer by měl zůstat s 1 záznamem
    assert buf.get_occupancy() == 1


def test_read_batch_wraparound(sample_data):
    buf = CircBuff(width=2, depth=3, data_type=np.dtype(np.float32), write_overflow_mode=WriteOverFlowMode.FLOW)
    buf.write_batch(np.vstack(sample_data[:3]))
    _ = buf.read_batch(2)
    # read_ptr už není na začátku, tak zapis další batch přes konec
    buf.write_batch(np.vstack(sample_data[3:6]))
    out = buf.read_batch(3)
    # ve FLOW dostaneme kontinuálně poslední 3
    np.testing.assert_array_equal(out, np.vstack(sample_data[3:6]))


# --- Kritické scénáře přetečení a wrap-around --- #

def test_overflow_exactly_one_past_limit():
    """Zápis přesně o 1 prvek více, než je hloubka, v LIMIT režimu."""
    buf = CircBuff(width=2, depth=3, data_type=np.dtype(np.float32), write_overflow_mode=WriteOverFlowMode.LIMIT)

    data = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float32)
    result = buf.write_batch(data)

    # poslední zápis (4. prvek) se nevejde
    assert result == CircBuffFlags.FULL
    assert buf.get_occupancy() == 3

    out = buf.read_batch(3)
    np.testing.assert_array_equal(out, data[:3])  # jen první tři se uchovaly


def test_overflow_flow_keeps_latest():
    """FLOW přetečení musí zahodit nejstarší a ponechat poslední záznamy."""
    buf = CircBuff(width=2, depth=3, data_type=np.dtype(np.float32), write_overflow_mode=WriteOverFlowMode.FLOW)

    data = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float32)
    buf.write_batch(data)

    # ve FLOW zůstávají poslední 3
    out = buf.read_batch(3)
    np.testing.assert_array_equal(out, data[1:])  # 2,3,4


def test_follow_moves_read_pointer_on_overflow():
    """FOLLOW musí posouvat read_ptr spolu s write_ptr při přetečení."""
    buf = CircBuff(width=2, depth=3, data_type=np.dtype(np.float32), write_overflow_mode=WriteOverFlowMode.FOLLOW)

    data = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float32)
    buf.write_batch(data)

    # stejně jako FLOW – musí zůstat poslední 3
    out = buf.read_batch(3)
    np.testing.assert_array_equal(out, data[1:])


def test_read_batch_larger_than_occupancy_limit():
    """Čtení větší dávky, než je v bufferu (LIMIT)."""
    buf = CircBuff(width=2, depth=4, data_type=np.dtype(np.float32), write_overflow_mode=WriteOverFlowMode.LIMIT)

    data = np.array([[1, 10], [2, 20]], dtype=np.float32)
    buf.write_batch(data)

    out = buf.read_batch(4)
    # vrátí se jen dva validní řádky
    np.testing.assert_array_equal(out, data)


def test_read_batch_wraparound_with_partial_fill():
    """FLOW wrap-around při čtení v dávce."""
    buf = CircBuff(width=2, depth=3, data_type=np.dtype(np.float32), write_overflow_mode=WriteOverFlowMode.FLOW)

    # zaplníme a přeplníme
    buf.write_batch(np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float32))
    _ = buf.read_batch(2)  # posuneme read_ptr doprostřed
    buf.write_batch(np.array([[4, 40], [5, 50]], dtype=np.float32))

    # dump by měl být konzistentní
    dump = buf.dump()
    assert dump.shape == (3, 2)

    # čtení celé dávky po wrapu → poslední 3 hodnoty
    out = buf.read_batch(3)
    np.testing.assert_array_equal(out, np.array([[3, 30], [4, 40], [5, 50]], dtype=np.float32))


def test_multiple_full_overflows_flow():
    """Opakované přetečení ve FLOW režimu stále uchovává poslední N prvků."""
    buf = CircBuff(width=2, depth=3, data_type=np.dtype(np.float32), write_overflow_mode=WriteOverFlowMode.FLOW)

    data = np.array([[i, i*10] for i in range(1, 10)], dtype=np.float32)
    buf.write_batch(data)

    # zůstávají poslední 3 (7,8,9)
    out = buf.read_batch(3)
    np.testing.assert_array_equal(out, data[-3:])


def test_follow_continuous_read_after_overflow():
    """FOLLOW: read_ptr sleduje write_ptr, takže se dají číst jen poslední N hodnot."""
    buf = CircBuff(width=2, depth=3, data_type=np.dtype(np.float32), write_overflow_mode=WriteOverFlowMode.FOLLOW)

    data = np.array([[i, i*10] for i in range(1, 7)], dtype=np.float32)
    buf.write_batch(data)

    # první čtení po přetečení – dostaneme poslední 3
    out1 = buf.read_batch(3)
    np.testing.assert_array_equal(out1, data[-3:])

    # další zápis a čtení – zas poslední 3
    buf.write_batch(np.array([[7, 70], [8, 80]], dtype=np.float32))
    out2 = buf.read_batch(3)
    np.testing.assert_array_equal(out2, np.array([[6, 60], [7, 70], [8, 80]], dtype=np.float32))
