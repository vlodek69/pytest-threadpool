"""Parallel test runner orchestration."""

import contextlib
import io
import logging
import os
import queue
import sys
import threading
from collections import OrderedDict
from datetime import UTC, datetime

from _pytest.logging import LogCaptureHandler, caplog_handler_key, caplog_records_key
from _pytest.runner import CallInfo, call_and_report, show_test_item

from pytest_threadpool._fixtures import FixtureManager
from pytest_threadpool._grouping import GroupKeyBuilder


def _is_teamcity(config, env: dict[str, str] | None = None) -> bool:
    """Detect TeamCity mode via CLI flag or TEAMCITY_VERSION env var."""
    _env = env if env is not None else os.environ
    return bool(config.getoption("teamcity", 0)) or bool(_env.get("TEAMCITY_VERSION"))


def _tc_escape(text: str) -> str:
    """Escape text for TeamCity service message values."""
    return (
        text.replace("|", "||")
        .replace("'", "|'")
        .replace("\n", "|n")
        .replace("\r", "|r")
        .replace("[", "|[")
        .replace("]", "|]")
    )


class _ThreadLocalStream:
    """Thread-local stream proxy for suppressing worker stdout/stderr.

    Installed once on sys.stdout/sys.stderr before parallel workers start.
    Worker threads call ``activate()`` to redirect their writes to a
    per-thread StringIO buffer, and ``deactivate()`` to stop.  Writes
    from non-worker threads (e.g. the main thread, live reporter) pass
    through to the real stream.
    """

    def __init__(self, real: object):
        self._real = real
        self._local = threading.local()

    def activate(self) -> None:
        self._local.buf = io.StringIO()

    def deactivate(self) -> str:
        """Stop buffering and return captured content."""
        buf = getattr(self._local, "buf", None)
        self._local.buf = None
        return buf.getvalue() if buf else ""

    def write(self, s: str) -> int:
        buf = getattr(self._local, "buf", None)
        return buf.write(s) if buf is not None else self._real.write(s)  # type: ignore[union-attr]

    def flush(self) -> None:
        buf = getattr(self._local, "buf", None)
        if buf is not None:
            buf.flush()
        else:
            self._real.flush()  # type: ignore[union-attr]  # pragma: no cover -- flush only called on activated buffers in tests

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


class _ThreadLocalLogHandler(logging.Handler):
    """Thread-local logging handler for capturing log records per worker.

    Installed once on the root logger before parallel workers start.
    Worker threads call ``activate()`` to collect records into a
    per-thread list, and ``deactivate()`` to retrieve them.  Records
    emitted by non-worker threads pass through to other handlers.
    """

    def __init__(self, level: int = logging.NOTSET, formatter: logging.Formatter | None = None):
        super().__init__(level)
        self._local = threading.local()
        if formatter is not None:
            self.setFormatter(formatter)

    def activate(self, caplog_handler: logging.Handler | None = None) -> None:
        self._local.records = []
        self._local.output = io.StringIO()
        self._local.caplog_handler = caplog_handler

    def deactivate(self) -> tuple[list[logging.LogRecord], str]:
        """Stop capturing and return (records, formatted_text)."""
        records = getattr(self._local, "records", None) or []
        output = getattr(self._local, "output", None)
        self._local.records = None
        self._local.output = None
        self._local.caplog_handler = None
        return records, output.getvalue() if output else ""

    def set_caplog_handler(self, handler: logging.Handler | None) -> None:
        """Set or clear the caplog handler for the current worker thread."""
        self._local.caplog_handler = handler

    def emit(self, record: logging.LogRecord) -> None:
        records = getattr(self._local, "records", None)
        if records is None:
            return  # pragma: no cover -- non-worker thread, let other handlers deal with it

        # Forward to caplog handler first.  Use handle() (not emit())
        # so the handler's own level filter — set by caplog.at_level()
        # or caplog.set_level() — is respected.
        caplog_h = getattr(self._local, "caplog_handler", None)
        if caplog_h is not None:
            caplog_h.handle(record)

        records.append(record)
        output = getattr(self._local, "output", None)
        if output is not None:
            try:
                msg = self.format(record)
                output.write(msg + "\n")
            except Exception:  # pragma: no cover
                pass


# Test slot states
_SCHEDULED = "scheduled"
_RUNNING = "running"
_DONE = "done"

# ANSI color codes for terminal output
_GREEN = "\033[32m"
_RED_BOLD = "\033[31;1m"
_YELLOW = "\033[33m"
_CYAN_BOLD = "\033[36;1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _outcome_key(report) -> str:
    """Return a short outcome key for a test report."""
    if hasattr(report, "wasxfail"):
        return "xfail" if report.skipped else "xpass"
    if report.passed:
        return "passed"
    if report.failed:
        return "error" if getattr(report, "when", "call") != "call" else "failed"
    if report.skipped:
        return "skipped"
    return ""


def _format_test_output(item, report, call_rep) -> list[str]:
    """Build display lines for a single test's output.

    Includes: nodeid, outcome, captured stdout/stderr, and traceback.
    """
    lines: list[str] = []
    # Header: nodeid + outcome.
    outcome = report.outcome.upper()
    if hasattr(report, "wasxfail"):
        outcome = "XFAIL" if report.skipped else "XPASS"
    elif getattr(report, "when", "call") != "call" and report.failed:
        outcome = "ERROR"
    color = {
        "PASSED": _GREEN,
        "FAILED": _RED_BOLD,
        "ERROR": _RED_BOLD,
        "SKIPPED": _YELLOW,
        "XFAIL": _YELLOW,
        "XPASS": _RED_BOLD,
    }.get(outcome, "")
    reset = _RESET if color else ""
    lines.append(f"{color}{outcome}{reset}  {item.nodeid}")

    # Sections (captured stdout, stderr, log).
    source = call_rep if call_rep is not None else report
    for section_name, section_content in getattr(source, "sections", []):
        lines.append(f"{_DIM}--- {section_name} ---{_RESET}")
        for line in section_content.rstrip("\n").splitlines():
            lines.append(line)

    # Traceback / longrepr — use toterminal() for colored output.
    longrepr = getattr(source, "longrepr", None)
    if longrepr is not None:
        lines.append(f"{_DIM}--- traceback ---{_RESET}")
        if hasattr(longrepr, "toterminal"):
            import io as _io

            from _pytest._io import TerminalWriter as _TW

            buf = _io.StringIO()
            tw = _TW(buf)
            tw.hasmarkup = True
            longrepr.toterminal(tw)
            for line in buf.getvalue().splitlines():
                lines.append(line)
        else:
            for line in str(longrepr).splitlines():
                lines.append(line)

    # Skip / xfail reason.
    wasxfail = getattr(report, "wasxfail", None)
    if wasxfail:
        lines.append(f"{_DIM}reason: {wasxfail}{_RESET}")

    return lines


class _LiveReporter:
    """Reports parallel results with live per-file line updates.

    Pre-prints all collected file lines before execution starts.
    Each test slot shows one of three states:
      - scheduled: dim dot (waiting to run)
      - running:   bright spinning indicator
      - done:      colored result letter (./F/s)

    Falls back to plain immediate reporting when stdout is not a terminal.
    In passive mode (``-s`` without a live TTY), delegates entirely to
    pytest's standard terminal reporter so IDE runners receive the
    output they rely on for result detection.

    When a ``ViewManager`` is provided, all terminal writes route through
    its ``"main"`` channel instead of directly to the file handle.
    """

    def __init__(self, session, items, view_manager=None):
        tr = session.config.pluginmanager.get_plugin("terminalreporter")
        self._tr = tr
        # noinspection PyProtectedMember
        # No public accessor for TerminalWriter; needed for hasmarkup,
        # raw file handle (ANSI cursor movement), and write suppression.
        self._tw = tr._tw if tr else None  # pyright: ignore[reportPrivateUsage]
        self._total = len(items)
        self._reported = 0
        self._startpath = session.config.rootpath
        self._lock = threading.Lock()
        self._view_manager = view_manager

        # Build file→items mapping in collection order
        self._file_order = []
        self._file_idx = {}
        self._file_items = OrderedDict()
        for item in items:
            fspath = item.fspath
            if fspath not in self._file_idx:
                self._file_idx[fspath] = len(self._file_order)
                self._file_order.append(fspath)
                self._file_items[fspath] = []
            self._file_items[fspath].append(item)

        # Per-item state: maps item → (_SCHEDULED | _RUNNING | _DONE, letter, color)
        self._item_state = {}
        for item in items:
            self._item_state[item] = (_SCHEDULED, ".", "")

        # detect capability
        self._live = (
            tr is not None
            and self._tw is not None
            and hasattr(tr, "_tw")
            and self._tw.hasmarkup
            and session.config.get_verbosity() <= 0
        )

        # Passive mode: when capture is disabled (``-s``) and we're not
        # on a live TTY, let pytest's standard reporter handle output.
        # This keeps IDE test runners (PyCharm, VS Code, etc.) working
        # because they rely on the terminal reporter's output to detect
        # test results.
        capture = session.config.getoption("capture", "fd")
        self._passive = not self._live and capture == "no"
        self._tc = _is_teamcity(session.config)

        self._width = getattr(self._tw, "fullwidth", 80) if self._tw else 80
        # noinspection PyProtectedMember
        # No public accessor on TerminalWriter for the underlying file handle.
        # We need it for direct ANSI escape writes (cursor movement, colors)
        # that bypass TerminalWriter formatting.
        raw_file = self._tw._file if self._tw else None  # pyright: ignore[reportPrivateUsage]
        self._file = raw_file

        # When a ViewManager is active, allocate rows in its shared
        # ScreenBuffer for this group's file lines + progress line.
        self._vm_row_offset = -1
        if view_manager is not None and self._live:
            nlines = len(self._file_order) + 1  # file lines + progress
            self._vm_row_offset = view_manager.allocate_lines(nlines)
            view_manager.add_test_items([item.nodeid for item in items])

        # When True, the next dumb-mode file line prefixes with \n
        # to separate from preceding sequential output.
        self._needs_leading_newline = False

        # Pre-compute colors for worker-thread display updates
        markup = self._tw.hasmarkup if self._tw else False
        self._pass_color = _GREEN if markup else ""
        self._fail_color = _RED_BOLD if markup else ""

    @property
    def live(self):
        return self._live  # pragma: no cover -- only used by external callers, not in test suite

    @property
    def passive(self):
        return self._passive

    def passive_color(self, report) -> str:
        """Return ANSI color for the result word in passive mode."""
        if not self._tw or not self._tw.hasmarkup:
            return ""
        if report.passed:
            return _GREEN
        if report.failed:
            return _RED_BOLD
        if report.skipped:
            return _YELLOW
        return ""

    # -- suppression helpers --------------------------------------------------

    def suppress(self):
        """Temporarily suppress terminal reporter output.

        Only suppresses ``_tw.write``/``_tw.line``.  IDE reporters
        (TeamCity / PyCharm) write directly to ``sys.stdout``, so
        their output passes through unaffected.
        """
        if self._tw:
            self._orig_write = self._tw.write
            self._orig_line = self._tw.line
            self._tw.write = lambda *a, **kw: None
            self._tw.line = lambda *a, **kw: None

    def restore(self):
        """Restore terminal reporter output."""
        if self._tw:
            self._tw.write = self._orig_write
            self._tw.line = self._orig_line

    # -- pre-print all files --------------------------------------------------

    def pre_print(self):
        """Print all collected file lines with dim scheduled dots and a progress line."""
        if not self._live:
            return
        if self._vm_row_offset >= 0 and self._view_manager is not None:
            # Vim-style: populate our rows in the shared buffer, redraw.
            off = self._vm_row_offset
            for i, fspath in enumerate(self._file_order):
                self._view_manager.set_line(off + i, self._build_line_live(fspath))
            self._view_manager.set_line(off + len(self._file_order), self._build_progress_line())
            self._view_manager.redraw()
            return
        assert self._file is not None
        f = self._file
        # Start on a fresh line so we don't overwrite any preceding
        # output (e.g. sequential test results between parallel groups).
        f.write("\n")
        for fspath in self._file_order:
            self._write_line_live(fspath)
            f.write("\n")
        self._write_progress_line()
        f.flush()

    # -- state transitions ----------------------------------------------------

    def mark_running(self, item):
        """Mark a test as currently running and update its file line."""
        with self._lock:
            self._item_state[item] = (_RUNNING, "●", "")
            if self._live:
                self._update_file_line(item.fspath)

    def mark_done(self, item, report):
        """Mark a test as completed and update its file line.

        In live mode, updates in-place with cursor movement.
        In dumb mode, writes the full file line once all tests in
        that file have completed (no cursor movement needed).

        If already marked done by mark_call_done, corrects the letter
        if the final report differs (e.g. skip detected as failure).
        """
        with self._lock:
            letter = self._letter_for(report)
            color = self._color_for(report)
            already_done = self._item_state[item][0] == _DONE
            if already_done and self._item_state[item][1] == letter:
                return
            if not already_done:
                self._reported += 1
            self._item_state[item] = (_DONE, letter, color)
            if self._live:
                self._update_file_line(
                    item.fspath
                )  # pragma: no cover -- live mode requires a real TTY
            elif not self._passive and not self._tc:
                self._maybe_flush_file(item.fspath)

    def mark_call_done(self, item, excinfo):
        """Mark a test call as completed from the worker thread.

        Uses excinfo to determine pass/fail immediately, so the display
        updates as soon as the call finishes rather than waiting for the
        main thread to process hooks.
        """
        with self._lock:
            self._reported += 1
            if excinfo is None:
                letter, color = ".", self._pass_color
            else:
                letter, color = "F", self._fail_color
            self._item_state[item] = (_DONE, letter, color)
            if self._live:
                self._update_file_line(item.fspath)
            elif not self._passive and not self._tc:
                self._maybe_flush_file(item.fspath)

    def finish(self):
        """Reset terminal reporter state after live output."""
        if self._passive:
            return
        if self._vm_row_offset >= 0 and self._view_manager is not None:
            # Alt screen stays up.  ViewManager.wait_and_leave() in
            # pytest_unconfigure will wait for Ctrl+C, then leave alt
            # screen and dump buffer contents to normal stdout.
            # Redirect terminal reporter output into the ViewManager
            # buffer so the failure summary appears on the alt screen
            # and gets dumped when leaving.
            if self._tr:
                self._tr.currentfspath = None
                self._tr._write_progress_information_filling_space = lambda: None  # pyright: ignore[reportPrivateUsage]
                if self._tw:
                    vm = self._view_manager
                    _pending: list[str] = []

                    def _redirect_write(*a, **kw) -> None:
                        text = a[0] if a else ""
                        if not text:
                            return
                        _pending.append(text)
                        # Flush complete lines into the buffer.
                        joined = "".join(_pending)
                        while "\n" in joined:
                            line, joined = joined.split("\n", 1)
                            vm.add_content(line)
                        _pending.clear()
                        if joined:
                            _pending.append(joined)

                    def _redirect_line(*a, **kw) -> None:
                        text = a[0] if a else ""
                        _pending.append(text)
                        joined = "".join(_pending)
                        for part in joined.split("\n"):
                            vm.add_content(part)
                        _pending.clear()

                    self._tw.write = _redirect_write  # pyright: ignore[reportAttributeAccessIssue]
                    self._tw.line = _redirect_line  # pyright: ignore[reportAttributeAccessIssue]
            return
        if self._live and self._tw and self._file:
            # Non-alternate-screen fallback.
            self._file.write("\n\033[A")
            self._file.flush()
        if self._tr:
            self._tr.currentfspath = None
            self._tr._write_progress_information_filling_space = lambda: None  # pyright: ignore[reportPrivateUsage]

    # -- internals ------------------------------------------------------------

    def _update_file_line(self, fspath):
        """Rewrite a single file line and progress using cursor movement (live mode)."""
        if self._vm_row_offset >= 0 and self._view_manager is not None:
            off = self._vm_row_offset
            idx = self._file_idx[fspath]
            self._view_manager.set_line(off + idx, self._build_line_live(fspath))
            self._view_manager.set_line(off + len(self._file_order), self._build_progress_line())
            self._view_manager.redraw()
            return
        assert self._file is not None
        f = self._file
        idx = self._file_idx[fspath]
        bottom = len(self._file_order)
        lines_up = bottom - idx
        f.write(f"\033[{lines_up}A")
        self._write_line_live(fspath)
        f.write(f"\033[{lines_up}B")
        f.write("\r")
        self._write_progress_line()
        f.flush()

    def _build_line_live(self, fspath) -> str:
        """Build a file line string with ANSI formatting (live terminal mode).

        Returns a string of visible text + ANSI codes.  The ScreenBuffer
        handles width clamping and padding.
        """
        items = self._file_items[fspath]
        rel = self._rel_path(fspath)
        width = self._width

        # Budget: path + space + one char per item must fit in width.
        usable = width - 1
        needed = len(rel) + 1 + len(items)

        if needed > usable:
            max_path = usable - 1 - len(items)
            if max_path >= 4:
                rel = rel[: max_path - 1] + "…"
            elif max_path >= 1:
                rel = "…"
            else:
                rel = ""

        parts: list[str] = []
        col = 0
        if rel:
            parts.append(rel + " ")
            col = len(rel) + 1

        for item in items:
            if col >= usable:
                break
            state, _letter, color = self._item_state[item]
            if state == _SCHEDULED:
                parts.append(f"{_DIM}·{_RESET}")
            elif state == _RUNNING:
                parts.append(f"{_CYAN_BOLD}●{_RESET}")
            else:
                if color:
                    parts.append(f"{color}{_letter}{_RESET}")
                else:
                    parts.append(_letter)
            col += 1

        return "".join(parts)

    def _write_line_live(self, fspath):
        """Write a file line with ANSI formatting (live terminal mode).

        Truncates the path and/or test indicators if the line would
        exceed the terminal width, preventing wrap-induced cursor
        miscalculation.
        """
        assert self._file is not None
        f = self._file
        f.write("\r\033[K")
        f.write(self._build_line_live(fspath))

    def _maybe_flush_file(self, fspath):
        """In dumb mode, write a file line once all its tests are done."""
        if not self._tw or not self._file:
            return  # pragma: no cover -- terminal writer is always available in test suite
        file_items = self._file_items[fspath]
        if all(self._item_state[it][0] == _DONE for it in file_items):
            self._write_line_plain(fspath)

    def _write_line_plain(self, fspath):
        """Write a file line without ANSI codes (dumb/pipe mode).

        Truncates the path and/or test indicators if the full line
        would exceed terminal width.
        """
        assert self._file is not None
        f = self._file
        rel = self._rel_path(fspath)
        items = self._file_items[fspath]
        progress = f" [{100 * self._reported // self._total:3d}%]"

        letters = "".join(self._item_state[item][1] for item in items)

        # path + space + letters + progress
        usable = self._width - 1
        line_len = len(rel) + 1 + len(letters) + len(progress)
        if line_len > usable:
            # Try truncating path first.
            max_path = usable - 1 - len(letters) - len(progress)
            if max_path >= 4:
                rel = rel[: max_path - 1] + "…"
            elif max_path >= 1:
                rel = "…"
            else:
                # Path can't shrink enough — truncate letters too.
                rel = "…"
                max_letters = usable - 1 - 1 - len(progress)  # "… " + letters + progress
                letters = letters[:max_letters] if max_letters > 0 else ""

        prefix = "\n" if self._needs_leading_newline else ""
        self._needs_leading_newline = False
        f.write(f"{prefix}{rel} {letters}{progress}\n")
        f.flush()

    def _build_progress_line(self) -> str:
        """Build the progress line string.

        The ScreenBuffer handles width clamping and padding.
        """
        pct = 100 * self._reported // self._total
        return f"{self._reported}/{self._total} [{pct:3d}%]"

    def _write_progress_line(self):
        """Write/update the progress line at the bottom.

        Truncated to terminal width (unlikely to overflow but
        enforced for consistency).
        """
        assert self._file is not None
        f = self._file
        text = self._build_progress_line()
        usable = self._width - 1
        if len(text) > usable:
            text = text[:usable]
        f.write(f"\r\033[K{text}")

    def _rel_path(self, fspath):
        try:
            return os.path.relpath(str(fspath), str(self._startpath))
        except ValueError:  # pragma: no cover -- only on Windows cross-drive paths
            return str(fspath)

    @staticmethod
    def _letter_for(report):
        if hasattr(report, "wasxfail"):
            if report.skipped:
                return "x"
            if report.passed:
                return "X"
        if report.passed:
            return "."
        if report.failed:
            if getattr(report, "when", "call") != "call":
                return "E"
            return "F"
        if report.skipped:
            return "s"
        return "?"  # pragma: no cover -- unknown report status; never produced by pytest

    def _color_for(self, report):
        if not self._tw or not self._tw.hasmarkup:
            return ""
        if hasattr(report, "wasxfail"):
            if report.skipped:
                return _YELLOW
            if report.passed:
                return _RED_BOLD
        if report.passed:
            return _GREEN
        if report.failed:
            return _RED_BOLD
        if report.skipped:
            return _YELLOW
        return ""  # pragma: no cover -- unknown report status fallback


class ParallelRunner:
    """Orchestrates parallel test execution.

    Groups consecutive items by parallel group key and runs each group
    either sequentially (key is None or single item) or in parallel.
    """

    def __init__(self, session, nthreads: int, view_manager=None):
        self._session = session
        self._nthreads = nthreads
        self._view_manager = view_manager

    def run_all(self) -> bool:
        """Main entry: group items and run each group."""
        session = self._session

        if (
            session.testsfailed and not session.config.option.continue_on_collection_errors
        ):  # pragma: no cover -- collection errors abort before reaching runner
            raise session.Interrupted(
                f"{session.testsfailed} error"
                f"{'s' if session.testsfailed != 1 else ''} during collection"
            )

        if session.config.option.collectonly:
            return True

        groups = GroupKeyBuilder.build_groups(session.items)
        has_parallel = any(
            k is not None and len(gi) > 1 and self._nthreads > 1 for k, gi in groups
        )

        needs_sep = False
        for group_idx, (group_key, items) in enumerate(groups):
            if session.shouldfail:
                raise session.Failed(session.shouldfail)
            if session.shouldstop:  # pragma: no cover -- requires --maxfail mid-group timing
                raise session.Interrupted(session.shouldstop)

            # First item of the next group — tells teardown_exact which
            # session/module/class nodes to keep alive across groups.
            next_group_first = None
            for _, future_items in groups[group_idx + 1 :]:
                if future_items:
                    next_group_first = future_items[0]
                    break

            if group_key is None or len(items) <= 1 or self._nthreads <= 1:
                for i, item in enumerate(items):
                    nextitem = items[i + 1] if i + 1 < len(items) else next_group_first
                    if has_parallel:
                        self._run_sequential_nodeid(item, nextitem)
                    else:
                        self._run_sequential(item, nextitem)
                needs_sep = bool(items)
            else:
                self._run_parallel(
                    items,
                    after_sequential=needs_sep,
                    next_group_first=next_group_first,
                )
                needs_sep = True

        return True

    def _run_sequential(self, item, nextitem) -> None:
        """Run the full default protocol for a single item."""
        ihook = item.ihook
        ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)

        # noinspection PyProtectedMember
        # No public API for request lifecycle management; mirrors pytest's
        # own runner.py (_pytest.runner.runtestprotocol).
        if hasattr(item, "_request") and not item._request:  # pyright: ignore[reportPrivateUsage]
            item._initrequest()  # pyright: ignore[reportPrivateUsage]  # pragma: no cover -- request always initialized before runner

        rep_setup = call_and_report(item, "setup", log=True)
        if rep_setup.passed:
            if item.config.getoption("setupshow", False):
                show_test_item(item)  # pragma: no cover -- only with --showfixtures/--setup-show
            if not item.config.getoption("setuponly", False):
                call_and_report(item, "call", log=True)
        call_and_report(item, "teardown", log=True, nextitem=nextitem)

        if hasattr(item, "_request"):
            item._request = False  # pyright: ignore[reportPrivateUsage]
            item.funcargs = None

        ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)

    def _run_sequential_nodeid(self, item, nextitem) -> None:
        """Run a single item sequentially with nodeid-style reporting.

        Used for sequential items in mixed parallel/sequential sessions
        so the output (``file::test PASSED``) visually distinguishes
        them from parallel groups (``file .....``).
        """
        tr = self._session.config.pluginmanager.get_plugin("terminalreporter")
        # noinspection PyProtectedMember
        tw = tr._tw if tr and hasattr(tr, "_tw") else None  # pyright: ignore[reportPrivateUsage]
        # noinspection PyProtectedMember
        f = getattr(tw, "_file", None) if tw else None  # pyright: ignore[reportPrivateUsage]

        # Suppress terminal reporter output so we can write our own format
        orig_write = None
        orig_line = None
        if tw:
            orig_write = tw.write
            orig_line = tw.line
            tw.write = lambda *a, **kw: None
            tw.line = lambda *a, **kw: None

        try:
            ihook = item.ihook
            ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)

            # noinspection PyProtectedMember
            if hasattr(item, "_request") and not item._request:  # pyright: ignore[reportPrivateUsage]
                item._initrequest()  # pyright: ignore[reportPrivateUsage]  # pragma: no cover -- request always initialized before runner

            rep_setup = call_and_report(item, "setup", log=True)
            call_rep = None
            if rep_setup.passed:
                if item.config.getoption("setupshow", False):
                    show_test_item(
                        item
                    )  # pragma: no cover -- only with --showfixtures/--setup-show
                if not item.config.getoption("setuponly", False):
                    call_rep = call_and_report(item, "call", log=True)
            call_and_report(item, "teardown", log=True, nextitem=nextitem)

            if hasattr(item, "_request"):
                item._request = False  # pyright: ignore[reportPrivateUsage]
                item.funcargs = None

            ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
        finally:
            if tw and orig_write is not None:
                tw.write = orig_write
                tw.line = orig_line

        # Write nodeid-style result line
        if f:
            report = call_rep if call_rep is not None else rep_setup
            markup = tw and tw.hasmarkup
            if hasattr(report, "wasxfail"):
                if report.skipped:
                    word = "XFAIL"
                    color = _YELLOW if markup else ""
                else:
                    word = "XPASS"
                    color = _RED_BOLD if markup else ""
            elif report.passed:
                word = "PASSED"
                color = _GREEN if markup else ""
            elif report.failed:
                word = "ERROR" if getattr(report, "when", "call") != "call" else "FAILED"
                color = _RED_BOLD if markup else ""
            elif report.skipped:
                word = "SKIPPED"
                color = _YELLOW if markup else ""
            else:  # pragma: no cover -- unknown report status; never produced by pytest
                word = "?"
                color = ""
            reset = _RESET if color else ""
            f.write(f"\n{item.nodeid} {color}{word}{reset}")
            f.flush()

        if self._view_manager is not None:
            report = call_rep if call_rep is not None else rep_setup
            self._view_manager.set_test_output(
                item.nodeid,
                _format_test_output(item, report, call_rep),
                outcome=_outcome_key(report),
            )

    def _run_parallel(self, items, after_sequential: bool = False, next_group_first=None) -> None:
        """Run a group's tests with parallel fixture setup and calls.

        Function-scoped FixtureDefs are cloned per-item so their setup can
        run concurrently alongside the test call.  Shared fixtures
        (module/class/session scope) are set up once via the first item,
        then served from cache to all workers.

        1. Sequential: set up first item (populates shared fixture caches).
        2. Sequential: prepare remaining items (_initrequest + clone FixtureDefs).
        3. Parallel:   workers run setup (cloned fixtures) + call per item.
        4. Sequential: report, teardown.
        """
        session = self._session
        setup_passed = {}
        setup_reports = {}
        per_item_fixture_fins = {}
        saved_collector_fins = []

        # noinspection PyProtectedMember
        # item._request/_initrequest and session._setupstate: no public API
        # for request lifecycle or setup state management.  Mirrors pytest's
        # own runner.py (_pytest.runner.runtestprotocol / SetupState).
        def _setup_first_item(item):
            """Full sequential setup for the first item (caches shared fixtures)."""
            if hasattr(item, "_request") and not item._request:  # pyright: ignore[reportPrivateUsage]
                item._initrequest()  # pyright: ignore[reportPrivateUsage]  # pragma: no cover -- request always initialized before runner

            needed = set(item.listchain())
            if any(node not in needed for node in session._setupstate.stack):  # pyright: ignore[reportPrivateUsage]
                saved_collector_fins.extend(  # pragma: no cover -- setuponly cross-module
                    FixtureManager.save_collector_finalizers(session, item)
                )
                session._setupstate.teardown_exact(nextitem=item)  # pyright: ignore[reportPrivateUsage]  # pragma: no cover -- setuponly cross-module

            rep = call_and_report(item, "setup", log=False)
            setup_reports[item] = rep
            setup_passed[item] = rep.passed

            if rep.passed:
                per_item_fixture_fins[item] = FixtureManager.save_and_clear_function_fixtures(item)
            else:
                FixtureManager.clear_function_fixture_caches(
                    item
                )  # pragma: no cover -- first-item setup failure in setuponly mode

            if item in session._setupstate.stack:  # pyright: ignore[reportPrivateUsage]
                session._setupstate.stack.pop(item)  # pyright: ignore[reportPrivateUsage]

        if session.config.getoption("setuponly", False):
            for item in items:
                _setup_first_item(item)
            for item in items:
                ihook = item.ihook
                ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
                ihook.pytest_runtest_logreport(report=setup_reports[item])
                if setup_passed[item] and session.config.getoption("setupshow", False):
                    show_test_item(item)
            self._teardown_all(
                items, per_item_fixture_fins, {}, saved_collector_fins, next_group_first
            )
            return

        # Phase 1+2: fire hooks for all items, populate shared fixture caches,
        # and clone function-scoped FixtureDefs.
        #
        # The first item that passes hook evaluation resolves shared fixtures
        # (module/class/session scope) to populate FixtureDef caches.  All
        # subsequent items get cache hits for shared fixtures.  Function-scoped
        # fixtures are NOT created here — they are deferred to parallel workers.
        #
        # noinspection PyProtectedMember
        parallel_items = []
        shared_populated_modules: set = set()

        # Shared-scope fixture output (setup_session, setup_module, etc.)
        # is produced during phase 1+2 ``call_and_report`` for the first
        # item per module.  That hook_rep is later discarded (workers
        # create fresh setup reports), so the captured sections are lost.
        #
        # With default capture, pytest stores the output in hook_rep.sections.
        # With ``-s``, pytest's capture is off and output leaks to raw stdout.
        #
        # We preserve the sections (default mode) and manually redirect
        # stdout (``-s`` mode) so setup output can be re-attached to the
        # setup report in ``_report_item``.
        capture_option = session.config.getoption("capture", "fd")
        capture_setup_redirect = capture_option == "no" and self._nthreads > 1 and len(items) > 1
        setup_captured_sections: dict[object, list] = {}

        for item in items:
            if hasattr(item, "_request") and not item._request:  # pyright: ignore[reportPrivateUsage]
                item._initrequest()  # pyright: ignore[reportPrivateUsage]  # pragma: no cover -- request always initialized before runner

            # Handle collector transitions (needed for PKG_CHILDREN groups
            # that span multiple modules).
            needed = set(item.listchain())
            if any(node not in needed for node in session._setupstate.stack):  # pyright: ignore[reportPrivateUsage]
                saved_collector_fins.extend(
                    FixtureManager.save_collector_finalizers(session, item)
                )
                session._setupstate.teardown_exact(nextitem=item)  # pyright: ignore[reportPrivateUsage]

            # Fire setup hooks with a custom setup function:
            # - First eligible item per module: resolve shared fixtures (populates caches)
            # - Remaining items in same module: no-op (shared caches already populated)
            # This evaluates skip/xfail markers and initializes capture/logging
            # without creating function-scoped fixtures.
            # Per-module tracking ensures cross-module PKG_CHILDREN groups
            # correctly populate shared fixtures for each module.
            original_setup = item.setup
            if item.module not in shared_populated_modules:
                item.setup = lambda i=item: FixtureManager.populate_shared_fixtures(i)
            else:
                item.setup = lambda: None

            # In ``-s`` mode, temporarily redirect stdout/stderr to capture
            # shared-scope fixture output that would otherwise leak to raw
            # stdout.  In default capture mode, pytest captures into
            # hook_rep.sections which we preserve below.
            saved_out = saved_err = None
            buf_out = buf_err = None
            if capture_setup_redirect:
                saved_out, saved_err = sys.stdout, sys.stderr
                buf_out, buf_err = io.StringIO(), io.StringIO()
                sys.stdout, sys.stderr = buf_out, buf_err  # type: ignore[assignment]
            try:
                hook_rep = call_and_report(item, "setup", log=False)
            finally:
                item.setup = original_setup
                if capture_setup_redirect:
                    sys.stdout, sys.stderr = saved_out, saved_err  # type: ignore[assignment]

            # Preserve captured sections from hook_rep (filled by pytest's
            # capture in default mode).  In ``-s`` mode, add our manually
            # redirected output as sections instead.
            sections = list(hook_rep.sections) if hook_rep.sections else []
            if buf_out is not None:
                setup_out = buf_out.getvalue()
                setup_err = buf_err.getvalue()  # type: ignore[union-attr]
                if setup_out:
                    sections.append(("Captured stdout setup", setup_out))
                if setup_err:
                    sections.append(("Captured stderr setup", setup_err))
            if sections:
                setup_captured_sections[item] = sections

            # Pop item from setupstate (pushed by _setupstate.setup inside the
            # hook) so the next item's hook doesn't fail the stack assertion.
            if item in session._setupstate.stack:  # pyright: ignore[reportPrivateUsage]
                session._setupstate.stack.pop(item)  # pyright: ignore[reportPrivateUsage]

            if not hook_rep.passed:
                # Hooks raised (skip, skipif, xfail NOTRUN, etc.)
                setup_reports[item] = hook_rep
                setup_passed[item] = False
                continue

            shared_populated_modules.add(item.module)

            FixtureManager.clone_function_fixturedefs(item)
            parallel_items.append(item)

        # Push all parallel items to setupstate stack so addfinalizer
        # assertions pass during fixture resolution in worker threads.
        # Done after the hook loop so items don't interfere with each
        # other's _setupstate.setup() assertions.
        for item in parallel_items:
            session._setupstate.stack[item] = ([item.teardown], None)  # pyright: ignore[reportPrivateUsage]

        # Phase 3: parallel setup + call
        workers = min(self._nthreads, len(parallel_items) or 1)
        call_results = {}
        reported = set()
        live = _LiveReporter(session, items, view_manager=self._view_manager)
        if after_sequential:
            live._needs_leading_newline = True
        live.pre_print()

        cancelled = threading.Event()

        def _drive_extra_hooks(hook_caller, item, core_prefix="_pytest."):
            """Manually run non-core hookwrapper implementations of a
            pytest_runtest_setup/call/teardown hook (e.g. allure-pytest's).

            The parallel worker path calls item.setup()/item.runtest()
            directly instead of going through ihook.pytest_runtest_setup(...)
            etc., because pytest's own default implementation touches
            session._setupstate, which is not thread-safe. That also means
            instrumentation plugins that hook these same events (allure,
            possibly others) never get invoked, silently breaking their
            per-test bookkeeping. This drives everyone EXCEPT pytest's own
            core implementation (identified by module prefix), so allure's
            wrapper still fires without reintroducing the thread-unsafe path.
            """
            gens = []
            for hookimpl in hook_caller.get_hookimpls():
                plugin_module = getattr(type(hookimpl.plugin), "__module__", "") or ""
                if plugin_module.startswith(core_prefix):
                    continue
                if not (getattr(hookimpl, "hookwrapper", False) or getattr(hookimpl, "wrapper", False)):
                    continue
                gen = hookimpl.function(item=item)
                try:
                    next(gen)
                    gens.append(gen)
                except StopIteration:
                    pass
            return gens

        def _finish_extra_hooks(gens):
            for gen in gens:
                try:
                    gen.send(None)
                except StopIteration:
                    pass

        def _do_setup_call_teardown(test_item):
            """Setup + call + teardown worker (for items with cloned FixtureDefs).

            Fixture setup uses cloned function-scoped FixtureDefs so each
            worker creates independent fixture instances without racing
            on shared FixtureDef state.  Shared fixtures are already cached
            from the first item's sequential setup, so their FixtureDef.execute()
            is a cache-hit read.

            Function-scoped fixture teardown (yield cleanup, addfinalizer
            callbacks, xunit teardown_method) also runs in the worker since
            each item's finalizers are independent.

            addfinalizer is redirected to a per-item list to avoid the
            setupstate stack assertion (the item IS in the stack, but
            list.append on the stack entry is also safe on free-threaded Python;
            the redirect simply avoids coupling to setupstate internals).
            """
            if cancelled.is_set():
                return (
                    test_item,
                    None,
                    None,
                    None,
                )  # pragma: no cover -- requires KeyboardInterrupt race
            return _do_worker_body(test_item)

        def _do_worker_body(test_item):
            # Redirect node.addfinalizer to per-item list
            original_addfinalizer = test_item.addfinalizer
            node_fins = []
            test_item.addfinalizer = lambda fin: node_fins.append(fin)

            extra_setup_hooks = _drive_extra_hooks(
                test_item.ihook.pytest_runtest_setup, test_item
            )
            try:
                setup_info = CallInfo.from_call(lambda: test_item.setup(), when="setup")
            finally:
                test_item.addfinalizer = original_addfinalizer
                _finish_extra_hooks(extra_setup_hooks)

            if setup_info.excinfo is None:
                fixture_fins = FixtureManager.save_and_clear_function_fixtures(test_item)
                call_info = None

                # Bridge caplog fixture to the thread-local log handler.
                # Instead of adding a handler to the root logger (which
                # would race on logger.setLevel via caplog.at_level),
                # install a fresh LogCaptureHandler and register it with
                # our _ThreadLocalLogHandler.  The thread-local handler
                # forwards records to caplog's handler, which applies its
                # own level filter set by at_level() / set_level().
                caplog_handler = None
                if "caplog" in test_item.fixturenames:
                    caplog_handler = LogCaptureHandler()
                    test_item.stash[caplog_handler_key] = caplog_handler
                    if caplog_records_key not in test_item.stash:
                        test_item.stash[caplog_records_key] = {}
                    test_item.stash[caplog_records_key]["call"] = caplog_handler.records
                    if log_handler is not None:
                        log_handler.set_caplog_handler(caplog_handler)

                if not cancelled.is_set():
                    live.mark_running(test_item)
                    extra_call_hooks = _drive_extra_hooks(
                        test_item.ihook.pytest_runtest_call, test_item
                    )
                    try:
                        call_info = CallInfo.from_call(lambda: test_item.runtest(), when="call")
                    finally:
                        _finish_extra_hooks(extra_call_hooks)
                    if not cancelled.is_set():
                        live.mark_call_done(test_item, call_info.excinfo)

                if caplog_handler is not None and log_handler is not None:
                    log_handler.set_caplog_handler(None)

                # Run function-scoped fixture teardown in the worker.
                # Includes yield cleanup, addfinalizer callbacks, and
                # node-level finalizers captured during setup.
                extra_teardown_hooks = _drive_extra_hooks(
                    test_item.ihook.pytest_runtest_teardown, test_item
                )
                all_fins = list(node_fins) + fixture_fins
                try:
                    teardown_info = CallInfo.from_call(
                        lambda fns=all_fins: FixtureManager.run_finalizers(fns), when="teardown"
                    )
                finally:
                    _finish_extra_hooks(extra_teardown_hooks)
                return test_item, setup_info, call_info, teardown_info

            FixtureManager.clear_function_fixture_caches(test_item)
            return test_item, setup_info, None, None

        def _report_item(item):
            """Report a single item — live cursor mode or plain fallback."""
            ihook = item.ihook

            live.suppress()
            try:
                call_rep = None
                if setup_passed[item] and item in call_results:
                    call_info = call_results[item]
                    call_rep = ihook.pytest_runtest_makereport(item=item, call=call_info)

                # Let plugins handle captured output first via hook.
                # If hook returns True, skip default output handling.
                out, err = captured_output.get(item, ("", ""))
                handled = any(
                    session.config.pluginmanager.hook.pytest_threadpool_report(
                        item=item,
                        report=call_rep or setup_reports[item],
                        captured_out=out,
                        captured_err=err,
                    )
                )

                # Attach captured worker output to the report.
                # For failures, pytest displays sections automatically.
                if not handled and call_rep is not None:
                    if out:
                        call_rep.sections.append(("Captured stdout call", out))
                    if err:
                        call_rep.sections.append(("Captured stderr call", err))

                # Attach captured log records to the report.
                if call_rep is not None and item in captured_log:
                    _, log_text = captured_log[item]
                    if log_text:
                        call_rep.sections.append(("Captured log call", log_text))

                report = call_rep or setup_reports[item]

                # Strip shared-scope captured sections from setup report.
                # CaptureManager re-adds captured output from phase 1+2
                # (session/package/module/class setup) to the worker's
                # setup report.  Remove those sections so they don't
                # appear in the first test's report in IDE reporters.
                if item in setup_captured_sections:
                    shared_content = {
                        sec_content for _, sec_content in setup_captured_sections[item]
                    }
                    setup_reports[item].sections = [
                        (name, content)
                        for name, content in setup_reports[item].sections
                        if content not in shared_content
                    ]

                ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
                ihook.pytest_runtest_logreport(report=setup_reports[item])
                if setup_passed[item]:
                    if session.config.getoption("setupshow", False):
                        show_test_item(item)

                    # In passive mode (``-s``), re-emit the result line
                    # and captured output via real stdout since ``_tw`` is
                    # suppressed and pytest's own capture is disabled.
                    # In dumb mode (default capture, no TTY), the file line
                    # already shows results and pytest captures output.
                    if live.passive:
                        real_out = stdout_proxy._real if stdout_proxy else sys.stdout
                        word = report.outcome.upper()
                        color = live.passive_color(report)
                        if color:
                            result_text = f"{item.nodeid} {color}{word}{_RESET}"
                        else:
                            result_text = f"{item.nodeid} {word}"
                        real_out.write(f"\n{result_text}\n")  # type: ignore[union-attr]

                        if not handled:
                            real_err = stderr_proxy._real if stderr_proxy else sys.stderr
                            if out:
                                real_out.write(out)  # type: ignore[union-attr]
                                if not out.endswith("\n"):
                                    real_out.write("\n")  # type: ignore[union-attr]
                            if err:
                                real_err.write(err)  # type: ignore[union-attr]
                                if not err.endswith("\n"):
                                    real_err.write("\n")  # type: ignore[union-attr]

                    if call_rep is not None:
                        ihook.pytest_runtest_logreport(report=call_rep)
            finally:
                live.restore()
            live.mark_done(item, report)

            # Populate the per-test output buffer for the live tree view.
            if self._view_manager is not None:
                self._view_manager.set_test_output(
                    item.nodeid,
                    _format_test_output(item, report, call_rep),
                    outcome=_outcome_key(report),
                )

            reported.add(item)

        teardown_infos = {}
        captured_output: dict = {}

        # Install thread-local stream proxies so worker print() output
        # doesn't corrupt test result lines.  Always installed (even
        # with ``-s``) so captured output can be reported alongside
        # the test result instead of leaking into global stdout.
        stdout_proxy: _ThreadLocalStream | None = None
        stderr_proxy: _ThreadLocalStream | None = None
        patched_stream_handlers: list[tuple[logging.StreamHandler, object]] = []
        if workers > 1 and len(parallel_items) > 1:
            stdout_proxy = _ThreadLocalStream(sys.stdout)
            stderr_proxy = _ThreadLocalStream(sys.stderr)
            real_stdout = sys.stdout
            sys.stdout = stdout_proxy  # type: ignore[assignment]
            sys.stderr = stderr_proxy  # type: ignore[assignment]

            # Patch existing StreamHandlers so their writes go through
            # our per-thread proxy instead of directly to the original
            # stream.  Handler.stream may be the raw stderr, a pytest
            # EncodedFile wrapper, or any other object — we can't reliably
            # identify which stream it targets by name or identity because
            # pytest's capture plugin swaps streams between collection and
            # execution.
            #
            # We redirect ALL StreamHandler streams (except FileHandlers
            # and our own _ThreadLocalLogHandler) through the proxy.
            # stdout-targeting handlers go to stdout_proxy; everything
            # else (stderr is the default) goes to stderr_proxy.
            for lg in [logging.getLogger()] + [
                logging.getLogger(name)
                for name in list(logging.Logger.manager.loggerDict)
                if isinstance(logging.Logger.manager.loggerDict.get(name), logging.Logger)
            ]:
                for h in list(lg.handlers):
                    if (
                        isinstance(h, logging.StreamHandler)
                        and not isinstance(
                            h, (logging.FileHandler, _ThreadLocalLogHandler, LogCaptureHandler)
                        )
                        and hasattr(h, "stream")
                    ):
                        original = h.stream
                        name = str(getattr(original, "name", ""))
                        if original is real_stdout or "<stdout>" in name:
                            patched_stream_handlers.append((h, original))
                            h.stream = stdout_proxy  # type: ignore[assignment]
                        else:
                            # Default: redirect to stderr proxy (StreamHandler
                            # defaults to stderr when no stream is specified).
                            patched_stream_handlers.append((h, original))
                            h.stream = stderr_proxy  # type: ignore[assignment]

        # Install thread-local log handler so worker log records are
        # captured per-item instead of going to the root logger's handlers
        # (which are not thread-safe for per-test reporting).
        log_handler: _ThreadLocalLogHandler | None = None
        saved_root_level: int = logging.WARNING
        captured_log: dict = {}
        if workers > 1 and len(parallel_items) > 1:
            log_level_raw = session.config.getoption("log_level", None)
            if not log_level_raw:
                log_level_raw = session.config.getini("log_level")
            log_level: int | None = None  # type: ignore[no-redef]
            if log_level_raw:
                with contextlib.suppress(ValueError, TypeError):
                    log_level = int(getattr(logging, str(log_level_raw).upper(), log_level_raw))
            log_fmt = session.config.getini("log_format")
            log_date_fmt = session.config.getini("log_date_format")
            formatter = logging.Formatter(log_fmt, log_date_fmt)
            handler_level = log_level if log_level is not None else logging.WARNING
            log_handler = _ThreadLocalLogHandler(level=handler_level, formatter=formatter)
            root_logger = logging.getLogger()
            root_logger.addHandler(log_handler)
            # Set root logger to NOTSET so all records reach our handler
            # regardless of caplog.at_level() / set_level() calls in
            # parallel tests (which race on root logger level).  Our
            # handler applies --log-level filtering via its own level,
            # and caplog handlers apply their own level filters.
            #
            # Patch root logger's setLevel to store levels in thread-local
            # storage so caplog.at_level() in one thread doesn't clobber
            # the root level for all other threads.  getEffectiveLevel is
            # also patched so child loggers walking up the hierarchy see
            # the per-thread level (or NOTSET if no thread-local override).
            saved_root_level = root_logger.level
            root_logger.setLevel(logging.NOTSET)
            _root_tl_levels = threading.local()
            _original_root_setLevel = type(root_logger).setLevel

            def _tl_root_setLevel(self: logging.Logger, level: int | str) -> None:
                _root_tl_levels.level = logging._checkLevel(level)  # type: ignore[attr-defined]

            _original_root_getEffectiveLevel = type(root_logger).getEffectiveLevel

            def _tl_root_getEffectiveLevel(self: logging.Logger) -> int:
                tl = getattr(_root_tl_levels, "level", None)
                if tl is not None:
                    return tl  # type: ignore[return-value]
                return logging.NOTSET

            type(root_logger).setLevel = _tl_root_setLevel  # type: ignore[assignment]
            type(root_logger).getEffectiveLevel = _tl_root_getEffectiveLevel  # type: ignore[assignment]

        tc_active = _is_teamcity(session.config)

        def _emit_tc_message(text: str) -> None:
            """Emit a TeamCity message event to the real stdout."""
            real_out = stdout_proxy._real if stdout_proxy else sys.stdout
            ts = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            real_out.write(  # type: ignore[union-attr]
                f"##teamcity[message timestamp='{ts}' text='{_tc_escape(text)}' status='NORMAL']\n"
            )

        def _emit_shared_setup_sections():
            """Emit shared-scope fixture output before any test is reported.

            In passive mode (``-s``): writes captured setup sections
            (session, package, module, class) to real stdout/stderr so they
            appear at the scope level, not inside the first test's output.

            In TeamCity mode (default capture): emits captured setup
            sections as ``##teamcity[message]`` events so they appear at
            the suite level in IDE test runners rather than inside an
            individual test's output.
            """
            if not live.passive and not tc_active:
                return
            real_out = stdout_proxy._real if stdout_proxy else sys.stdout
            real_err = stderr_proxy._real if stderr_proxy else sys.stderr
            for item in parallel_items:
                sections = setup_captured_sections.get(item, [])
                for sec_name, sec_content in sections:
                    if tc_active and not live.passive:
                        _emit_tc_message(sec_content)
                    else:
                        target = real_err if "stderr" in sec_name else real_out
                        target.write(sec_content)  # type: ignore[union-attr]
                        if not sec_content.endswith("\n"):
                            target.write("\n")  # type: ignore[union-attr]

        interrupted = False
        if workers > 1 and len(parallel_items) > 1:
            work_queue = queue.SimpleQueue()
            result_queue = queue.SimpleQueue()

            def _pool_worker():
                while True:
                    work_item = work_queue.get()
                    if work_item is None or cancelled.is_set():
                        return
                    if stdout_proxy is not None:
                        stdout_proxy.activate()
                        stderr_proxy.activate()  # type: ignore[union-attr]
                    if log_handler is not None:
                        log_handler.activate()
                    call_info = None
                    try:
                        item, setup_info, call_info, td_info = _do_setup_call_teardown(work_item)
                        if setup_info is not None:
                            setup_rep = item.ihook.pytest_runtest_makereport(
                                item=item, call=setup_info
                            )
                            setup_reports[item] = setup_rep
                            setup_passed[item] = setup_info.excinfo is None
                        if td_info is not None:
                            teardown_infos[item] = td_info
                    except BaseException as exc:
                        call_info = CallInfo.from_call(
                            lambda e=exc: (_ for _ in ()).throw(e),
                            when="call",
                        )
                        # Ensure setup_reports/setup_passed are populated so
                        # _report_item doesn't KeyError on unexpected exceptions.
                        if work_item not in setup_reports:
                            setup_info = CallInfo.from_call(
                                lambda e=exc: (_ for _ in ()).throw(e),
                                when="setup",
                            )
                            setup_reports[work_item] = work_item.ihook.pytest_runtest_makereport(
                                item=work_item, call=setup_info
                            )
                            setup_passed[work_item] = False
                    finally:
                        # Capture output BEFORE queueing the result so the
                        # main thread's _report_item always sees the data.
                        if stdout_proxy is not None:
                            out = stdout_proxy.deactivate()
                            err = stderr_proxy.deactivate()  # type: ignore[union-attr]
                            if out or err:
                                captured_output[work_item] = (out, err)
                        if log_handler is not None:
                            records, log_text = log_handler.deactivate()
                            if records:
                                captured_log[work_item] = (records, log_text)
                        result_queue.put((work_item, call_info))

            threads = []
            for _ in range(workers):
                t = threading.Thread(target=_pool_worker, daemon=True)
                t.start()
                threads.append(t)

            # Submit all parallel items (setup + call + teardown)
            for item in parallel_items:
                work_queue.put(item)

            # Emit shared-scope fixture output (setup_session, etc.)
            # before any test is reported so it appears at the scope
            # level, not inside the first test.
            _emit_shared_setup_sections()

            try:
                collected = 0
                if tc_active:
                    # In TC/IDE mode, report in collection order so
                    # parametrized tests are grouped correctly.
                    ready = set()
                    next_report_idx = 0
                    while collected < len(parallel_items):
                        finished_item, call_info = result_queue.get()
                        if call_info is not None:
                            call_results[finished_item] = call_info
                        ready.add(finished_item)
                        collected += 1
                        while (
                            next_report_idx < len(parallel_items)
                            and parallel_items[next_report_idx] in ready
                        ):
                            _report_item(parallel_items[next_report_idx])
                            next_report_idx += 1
                else:
                    # In terminal mode, report in completion order for
                    # immediate feedback (fast tests show before slow ones).
                    while collected < len(parallel_items):
                        finished_item, call_info = result_queue.get()
                        if call_info is not None:
                            call_results[finished_item] = call_info
                        _report_item(finished_item)
                        collected += 1
            except KeyboardInterrupt:
                interrupted = True
                cancelled.set()

            for _ in range(workers):
                work_queue.put(None)
            if not interrupted:
                for t in threads:
                    t.join()
        else:
            # Single worker fallback
            _emit_shared_setup_sections()
            try:
                for item in parallel_items:
                    if (
                        stdout_proxy is not None
                    ):  # pragma: no cover -- single-worker without capture proxy
                        stdout_proxy.activate()
                        stderr_proxy.activate()  # type: ignore[union-attr]
                    try:
                        item, setup_info, call_info, td_info = _do_setup_call_teardown(item)
                        if setup_info is not None:
                            setup_rep = item.ihook.pytest_runtest_makereport(
                                item=item, call=setup_info
                            )
                            setup_reports[item] = setup_rep
                            setup_passed[item] = setup_info.excinfo is None
                        if td_info is not None:
                            teardown_infos[item] = td_info
                    finally:
                        if (
                            stdout_proxy is not None
                        ):  # pragma: no cover -- single-worker without capture proxy
                            stdout_proxy.deactivate()
                            stderr_proxy.deactivate()  # type: ignore[union-attr]
                    if call_info is not None:
                        call_results[item] = call_info
                    _report_item(item)
            except KeyboardInterrupt:
                interrupted = True

        # Report any remaining items (setup failures, stragglers)
        if not interrupted:
            for item in items:
                if item not in reported:
                    _report_item(item)

        live.finish()

        # Restore real stdout/stderr and remove log handler after all workers are done
        if stdout_proxy is not None:
            sys.stdout = stdout_proxy._real  # type: ignore[assignment]
            sys.stderr = stderr_proxy._real  # type: ignore[union-attr, assignment]
        # Restore patched StreamHandlers to their original streams
        for h, original_stream in patched_stream_handlers:
            h.stream = original_stream
        if log_handler is not None:
            logging.getLogger().removeHandler(log_handler)
            # Restore patched Logger methods before restoring level
            type(logging.getLogger()).setLevel = _original_root_setLevel  # type: ignore[possibly-undefined]
            type(logging.getLogger()).getEffectiveLevel = _original_root_getEffectiveLevel  # type: ignore[possibly-undefined]
            logging.getLogger().setLevel(saved_root_level)

        # Pop parallel items from setupstate stack
        # noinspection PyProtectedMember
        for item in parallel_items:
            if item in session._setupstate.stack:  # pyright: ignore[reportPrivateUsage]
                session._setupstate.stack.pop(item)  # pyright: ignore[reportPrivateUsage]

        # Teardown reporting + collector teardown (always runs, even after interrupt)
        self._teardown_all(
            items,
            per_item_fixture_fins,
            teardown_infos,
            saved_collector_fins,
            next_group_first,
            passive=live.passive,
            tc_active=tc_active,
        )

        if interrupted:
            raise KeyboardInterrupt

    def _teardown_all(
        self,
        items,
        per_item_fixture_fins,
        teardown_infos,
        saved_collector_fins,
        next_group_first=None,
        passive: bool = False,
        tc_active: bool = False,
    ) -> None:
        """Report teardown results, run any remaining finalizers, and tear down
        collectors.

        For items with pre-computed teardown_infos (from parallel workers),
        only reporting happens here.  For items without (setuponly mode),
        finalizers from per_item_fixture_fins are executed sequentially.

        next_group_first: first item of the next group (or None if last group).
        Passed to teardown_exact so session/module/class-scoped fixtures that
        are still needed by the next group are preserved rather than torn down.
        """
        session = self._session

        for item in items:
            if item in teardown_infos:
                # Teardown already ran in the worker — just report.
                teardown_info = teardown_infos[item]
            else:
                # Fallback: run finalizers now (setuponly mode).
                fins = per_item_fixture_fins.get(item, [])
                teardown_info = CallInfo.from_call(
                    lambda fns=fins: FixtureManager.run_finalizers(fns), when="teardown"
                )

            rep = item.ihook.pytest_runtest_makereport(item=item, call=teardown_info)
            item.ihook.pytest_runtest_logreport(report=rep)

            # noinspection PyProtectedMember
            if hasattr(item, "_request"):
                item._request = False  # pyright: ignore[reportPrivateUsage]
                item.funcargs = None

            item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)

        # noinspection PyProtectedMember
        # Pass next_group_first so session/module/class nodes needed by the
        # next group stay in the stack with their cached fixtures intact.
        #
        # Shared-scope fixture teardown (session/package/module/class)
        # runs here.  In ``-s`` mode the output would leak to raw stdout;
        # in default capture mode it leaks because our proxy is already
        # restored.  Redirect stdout/stderr to capture the output so it
        # can be emitted in passive mode and suppressed otherwise.
        td_saved_out, td_saved_err = sys.stdout, sys.stderr
        td_buf_out, td_buf_err = io.StringIO(), io.StringIO()
        sys.stdout, sys.stderr = td_buf_out, td_buf_err  # type: ignore[assignment]
        try:
            session._setupstate.teardown_exact(nextitem=next_group_first)  # pyright: ignore[reportPrivateUsage]
        finally:
            sys.stdout, sys.stderr = td_saved_out, td_saved_err  # type: ignore[assignment]

        # In passive mode, emit the captured teardown output so it
        # appears after all tests in the output.  In TeamCity mode,
        # emit as ##teamcity[message] events for suite-level display.
        td_out = td_buf_out.getvalue()
        td_err = td_buf_err.getvalue()
        if passive:
            if td_out:
                td_saved_out.write(td_out)
                if not td_out.endswith("\n"):
                    td_saved_out.write("\n")
            if td_err:
                td_saved_err.write(td_err)
                if not td_err.endswith("\n"):
                    td_saved_err.write("\n")
        elif tc_active and td_out:
            ts = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            td_saved_out.write(
                f"##teamcity[message timestamp='{ts}'"
                f" text='{_tc_escape(td_out)}' status='NORMAL']\n"
            )

        all_collector_fins = [
            fin for _node, fins in reversed(saved_collector_fins) for fin in reversed(fins)
        ]
        FixtureManager.run_finalizers(
            all_collector_fins, msg="errors during deferred collector teardown"
        )
