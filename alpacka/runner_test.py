"""Tests for alpacka.runner."""

import functools

from alpacka import runner
from alpacka import runner_callbacks


def test_smoke(tmpdir, capsys):
    n_epochs = 3
    runner.Runner(
        output_dir=tmpdir,
        n_envs=2,
        n_epochs=n_epochs,
    ).run()

    # Check that metrics were printed in each epoch.
    captured = capsys.readouterr()
    assert captured.out.count('return_mean') == n_epochs


def test_smoke_with_eval(tmpdir, capsys):
    n_epochs = 3
    runner.Runner(
        output_dir=tmpdir,
        n_envs=2,
        n_epochs=n_epochs,
        callback_classes=(functools.partial(
            runner_callbacks.EvaluationCallback, eval_period=1
        ),),
    ).run()

    # Check that eval metrics were printed in each epoch.
    captured = capsys.readouterr()
    assert captured.out.count('eval_episode/return_mean') == n_epochs


def test_restarts(tmpdir, capsys):
    runner.Runner(
        output_dir=tmpdir,
        n_envs=2,
        n_epochs=1,
    ).run()

    # Check that one epoch has been run.
    captured = capsys.readouterr()
    assert captured.out.count('0 |') > 0
    assert captured.out.count('1 |') == 0

    runner.Runner(
        output_dir=tmpdir,
        n_envs=2,
        n_epochs=2,
    ).run()

    # Check that another epoch has been run.
    captured = capsys.readouterr()
    assert captured.out.count('0 |') == 0
    assert captured.out.count('1 |') > 0
    assert captured.out.count('2 |') == 0
