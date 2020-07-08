"""Tests for alpacka.utils.os."""

import pytest

from alpacka.utils import os as alpacka_os


def test_single_write(tmp_path):
    path = tmp_path / 'tmp'
    with alpacka_os.atomic_dump((path,)) as (dump_path,):
        with open(dump_path, 'w') as f:
            f.write('foo')

    with open(path, 'r') as f:
        assert f.read() == 'foo'


def test_double_write(tmp_path):
    path = tmp_path / 'tmp'

    # Write the first time.
    with open(path, 'w') as f:
        f.write('foo')

    # Write the second time.
    with alpacka_os.atomic_dump((path,)) as (dump_path,):
        with open(dump_path, 'w') as f:
            f.write('bar')

    # The content should be overwritten.
    with open(path, 'r') as f:
        assert f.read() == 'bar'


def test_partial_write(tmp_path):
    path = tmp_path / 'tmp'

    # Write the first time.
    with open(path, 'w') as f:
        f.write('foo')

    # Write the second time, with interruption.
    with pytest.raises(Exception):
        with alpacka_os.atomic_dump((path,)) as (dump_path,):
            with open(dump_path, 'w') as f:
                f.write('bar')
                raise Exception

    # The content should not be overwritten.
    with open(path, 'r') as f:
        assert f.read() == 'foo'


def test_write_to_multiple_files(tmp_path):
    path1 = tmp_path / 'tmp1'
    path2 = tmp_path / 'tmp2'
    with alpacka_os.atomic_dump((path1, path2)) as (dump_path1, dump_path2):
        with open(dump_path1, 'w') as f:
            f.write('foo')
        with open(dump_path2, 'w') as f:
            f.write('bar')

    with open(path1, 'r') as f:
        assert f.read() == 'foo'
    with open(path2, 'r') as f:
        assert f.read() == 'bar'
