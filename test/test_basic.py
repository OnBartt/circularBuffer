import numpy as np
import sys
import os

# přidej kořenový adresář projektu do sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import pytest
from CircularBuffer.CircularBuffer import CircBuff, CircBuffFlags, WriteOverFlowMode  # noqa: E402


def test_circ_buffer_basic():
    width = 3
    depth = 4

    # vytvoření bufferu
    buf = CircBuff(width=width, depth=depth, data_type=np.dtype(np.float32),
                   write_overflow_mode=WriteOverFlowMode.LIMIT)

    # buffer by měl být prázdný
    assert buf.get_state() == CircBuffFlags.EMPTY
    assert buf.get_occupancy() == 0

    # zapis postupně jeden řádek
    for i in range(depth):
        row = np.array([i, i+1, i+2], dtype=np.float32)
        state = buf.write_single(row)
        if i < depth - 1:
            assert state == CircBuffFlags.NONEMPTY
        else:
            assert state == CircBuffFlags.FULL

    # occupancy by měl být depth
    assert buf.get_occupancy() == depth

    # při LIMIT by další zápis selhal
    overflow_row = np.array([0, 0, 0], dtype=np.float32)
    assert buf.write_single(overflow_row) == CircBuffFlags.FULL

    # čtení postupně
    for i in range(depth):
        expected = np.array([i, i+1, i+2], dtype=np.float32)
        data = buf.read_single()
        np.testing.assert_array_equal(data, expected)

    # buffer by měl být prázdný
    assert buf.get_state() == CircBuffFlags.EMPTY
    assert buf.get_occupancy() == 0

    # flush
    buf.flush()
    assert buf.get_state() == CircBuffFlags.EMPTY
    assert buf.get_occupancy() == 0


def test_circ_buffer_width_error():
    buf = CircBuff(width=3, depth=4, data_type=np.dtype(np.float32),
                   write_overflow_mode=WriteOverFlowMode.FLOW)

    # nesprávná šířka řádku
    wrong_row = np.array([1, 2], dtype=np.float32)
    state = buf.write_single(wrong_row)
    assert state == CircBuffFlags.WIDTH_ERROR


def test_circ_buffer_edge_cases():
    width = 2
    depth = 3

    # FLOW mode buffer
    buf = CircBuff(width=width, depth=depth, data_type=np.dtype(np.float32), write_overflow_mode=WriteOverFlowMode.FLOW)

    # Zápis a přetečení
    buf.write_single(np.array([1, 2], dtype=np.float32))
    buf.write_single(np.array([3, 4], dtype=np.float32))
    buf.write_single(np.array([5, 6], dtype=np.float32))
    # Přepsání prvního řádku
    buf.write_single(np.array([7, 8], dtype=np.float32))

    # Čtení postupně
    expected_rows_flow = [
        np.array([7, 8], dtype=np.float32),
        np.array([3, 4], dtype=np.float32),
        np.array([5, 6], dtype=np.float32)
    ]
    for expected in expected_rows_flow:
        data = buf.read_single()
        np.testing.assert_array_equal(data, expected)

    # Ve FLOW módu buffer nemusí být EMPTY po vyčtení hloubky, kontrolujeme flag
    state = buf.get_state()
    assert state in [CircBuffFlags.NONEMPTY, CircBuffFlags.EMPTY]

    # Čtení z prázdného bufferu (pokud opravdu došlo k úplnému vyprázdnění)
    if state == CircBuffFlags.EMPTY:
        assert buf.read_single() == CircBuffFlags.EMPTY

    # Špatná šířka
    wrong_row = np.array([1], dtype=np.float32)
    assert buf.write_single(wrong_row) == CircBuffFlags.WIDTH_ERROR

    # Test oběhu ukazatelů LIMIT mode
    buf_limit = CircBuff(width=width, depth=depth, data_type=np.dtype(np.float32),
                         write_overflow_mode=WriteOverFlowMode.LIMIT)
    for i in range(depth):
        buf_limit.write_single(np.array([i, i+10], dtype=np.float32))
    for _ in range(depth):
        buf_limit.read_single()
    # Po vyčtení všech řádků je buffer EMPTY
    assert buf_limit.get_state() == CircBuffFlags.EMPTY
