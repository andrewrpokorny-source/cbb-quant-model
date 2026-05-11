"""Regression tests for the dashboard's per-thread stdout capture.

The loading panel installs a process-global ThreadDispatchStdout on
sys.stdout. Two overlapping Streamlit sessions used to wrap each other's
dispatchers, which - combined with a thread-local stream that's shared
across dispatcher hops - produced infinite recursion on the first worker
write. These tests pin the invariants that prevent that:

  * Both helpers always fall back to sys.__stdout__, never to whatever
    sys.stdout currently points at.
  * install_stdout_dispatcher() is idempotent across reruns and does not
    chain dispatchers.
  * Concurrent workers each see their own CapturedStream and do not bleed
    output into each other.
"""

from __future__ import annotations

import io
import sys
import threading
import time

import pytest

from dashboard_helpers import (
    CapturedStream,
    THREAD_BUFFERS,
    ThreadDispatchStdout,
    install_stdout_dispatcher,
    silenced_stdout,
)


@pytest.fixture
def fresh_stdout():
    """Save and restore sys.stdout around tests that mutate it."""
    original = sys.stdout
    try:
        yield
    finally:
        sys.stdout = original
        THREAD_BUFFERS.stream = None


def test_captured_stream_lines_finalize_on_newline(fresh_stdout):
    s = CapturedStream()
    s.write("hello")
    new, offset = s.consume_new(0)
    assert new == []  # nothing finalized yet -- no \n
    assert offset == 0

    s.write(" world\n")
    new, offset = s.consume_new(0)
    assert new == ["hello world"]
    assert offset == 1


def test_captured_stream_handles_multi_line_chunks(fresh_stdout):
    s = CapturedStream()
    s.write("a\nb\nc")
    new, offset = s.consume_new(0)
    assert new == ["a", "b"]  # "c" still partial
    assert offset == 2

    s.write("d\n")  # finalizes "cd"
    more, offset2 = s.consume_new(offset)
    assert more == ["cd"]
    assert offset2 == 3


def test_captured_stream_strips_whitespace_only_lines(fresh_stdout):
    s = CapturedStream()
    s.write("real\n   \n  \nother\n")
    new, _ = s.consume_new(0)
    assert new == ["real", "other"]


def test_install_dispatcher_is_idempotent(fresh_stdout):
    first = install_stdout_dispatcher()
    second = install_stdout_dispatcher()
    third = install_stdout_dispatcher()
    assert first is second is third
    assert sys.stdout is first
    assert isinstance(sys.stdout, ThreadDispatchStdout)


def test_dispatcher_falls_back_to_original_stdout_not_previous_dispatcher(
    fresh_stdout,
):
    """Critical recursion-safety invariant.

    If a dispatcher's fallback were `sys.stdout` at install time -- which
    might already be a dispatcher from an earlier overlapping session --
    a worker write would route DispatchB -> stream -> DispatchA -> stream
    -> DispatchA... until the stack blew up. We pin that the helpers
    always defer to sys.__stdout__ instead, so the chain has length 1.
    """
    # Pretend an earlier session installed dispatcher A.
    install_stdout_dispatcher()
    a = sys.stdout

    # Simulate the previously-buggy second install: even if we forcibly
    # construct a fresh ThreadDispatchStdout instance (e.g. via test or
    # cold-start race), its writes must still target sys.__stdout__, not
    # whatever sys.stdout is at construction time.
    b = ThreadDispatchStdout()
    sys.stdout = b

    captured = CapturedStream()
    THREAD_BUFFERS.stream = captured
    try:
        # A single print() must not recurse. If b.write routed through
        # a.write (and both checked the same thread-local), we'd loop until
        # RecursionError. Bound the recursion check by setting a low limit.
        prev_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(50)
        try:
            print("ping")  # routes through b.write -> captured.write -> sys.__stdout__
        finally:
            sys.setrecursionlimit(prev_limit)
    finally:
        THREAD_BUFFERS.stream = None
        sys.stdout = a

    new, _ = captured.consume_new(0)
    assert new == ["ping"]


def test_main_thread_writes_pass_through_when_no_capture(fresh_stdout):
    """Without a thread-local stream set, dispatcher writes must go to
    sys.__stdout__ and not affect any CapturedStream."""
    install_stdout_dispatcher()
    captured = CapturedStream()
    # NOTE: do *not* set THREAD_BUFFERS.stream here -- we're simulating the
    # main thread, which never opts in.
    print("not-captured-on-main")
    new, _ = captured.consume_new(0)
    assert new == []  # captured saw nothing because it was never registered


def test_worker_threads_isolate_captures(fresh_stdout):
    """Two threads each register their own CapturedStream; their writes
    must not bleed into each other or recurse."""
    install_stdout_dispatcher()

    streams = {"a": CapturedStream(), "b": CapturedStream()}
    barrier = threading.Barrier(3)

    def _worker(tag: str) -> None:
        THREAD_BUFFERS.stream = streams[tag]
        try:
            barrier.wait()  # release all three together
            for i in range(20):
                print(f"{tag}-{i}")
        finally:
            THREAD_BUFFERS.stream = None

    threads = [
        threading.Thread(target=_worker, args=("a",)),
        threading.Thread(target=_worker, args=("b",)),
    ]
    for t in threads:
        t.start()
    barrier.wait()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive()

    a_lines, _ = streams["a"].consume_new(0)
    b_lines, _ = streams["b"].consume_new(0)
    assert a_lines == [f"a-{i}" for i in range(20)]
    assert b_lines == [f"b-{i}" for i in range(20)]


def test_silenced_stdout_only_affects_calling_thread(fresh_stdout):
    """The main thread can suppress its own stdout (e.g. while running the
    backtest) without breaking a concurrent worker's capture in another
    Streamlit session.

    Pins the invariant that ruled out contextlib.redirect_stdout: a global
    stdout swap would have routed *every* thread's writes into the
    silencing sink, so the worker's CapturedStream would have collected
    nothing.
    """
    install_stdout_dispatcher()

    worker_stream = CapturedStream()
    worker_started = threading.Event()
    main_inside_silenced = threading.Event()
    worker_done = threading.Event()
    worker_lines: list[str] = []

    def _worker() -> None:
        THREAD_BUFFERS.stream = worker_stream
        try:
            worker_started.set()
            # Hold here until the main thread is inside silenced_stdout(),
            # then emit -- this is exactly the scenario the finding flagged.
            assert main_inside_silenced.wait(timeout=2)
            for i in range(10):
                print(f"worker-{i}")
        finally:
            THREAD_BUFFERS.stream = None
            worker_done.set()

    t = threading.Thread(target=_worker)
    t.start()
    assert worker_started.wait(timeout=2)

    with silenced_stdout():
        main_inside_silenced.set()
        # Anything we print on the main thread now must be swallowed.
        print("main-should-be-silenced")
        assert worker_done.wait(timeout=5)

    t.join(timeout=5)
    assert not t.is_alive()

    worker_lines, _ = worker_stream.consume_new(0)
    assert worker_lines == [f"worker-{i}" for i in range(10)]


def test_silenced_stdout_restores_previous_thread_local(fresh_stdout):
    """Nested or re-entrant silencing must not clobber a prior thread-local
    stream set by an outer caller (e.g. a worker that already opted in)."""
    install_stdout_dispatcher()
    outer = CapturedStream()
    THREAD_BUFFERS.stream = outer
    try:
        with silenced_stdout():
            print("swallowed")
        # After the context, the outer capture must be back in place.
        assert THREAD_BUFFERS.stream is outer
        print("after-silenced")
        new, _ = outer.consume_new(0)
        assert new == ["after-silenced"]
    finally:
        THREAD_BUFFERS.stream = None


def test_captured_stream_concurrent_writer_and_reader(fresh_stdout):
    """consume_new() called concurrently with write() must not crash or
    deadlock, and every line emitted must eventually be readable in order."""
    s = CapturedStream()
    n_lines = 500
    done = threading.Event()

    def _writer() -> None:
        for i in range(n_lines):
            s.write(f"line-{i}\n")
        done.set()

    seen: list[str] = []
    offset = 0

    def _reader() -> None:
        nonlocal offset
        while not done.is_set() or offset < n_lines:
            new, offset_new = s.consume_new(offset)
            seen.extend(new)
            offset = offset_new
            time.sleep(0.001)

    t1 = threading.Thread(target=_writer)
    t2 = threading.Thread(target=_reader)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert not t1.is_alive() and not t2.is_alive()
    assert seen == [f"line-{i}" for i in range(n_lines)]
