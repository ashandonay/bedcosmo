"""Method, loop, and section timing helpers for training and evaluation."""

import contextlib
import functools
import inspect
import os
import threading
import time

import psutil


_profile_depth = threading.local()


def get_profile_depth():
    """Get current profile nesting depth for indentation."""
    if not hasattr(_profile_depth, "depth"):
        return 0
    return _profile_depth.depth


def _get_memory_usage():
    """Get current memory usage in MB. Returns None if unavailable."""
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)
    except Exception:
        return None


def _profile_enabled(owner):
    """True when owner.profile is set and this is rank 0 (if distributed)."""
    if not getattr(owner, "profile", False):
        return False
    if hasattr(owner, "global_rank") and owner.global_rank != 0:
        return False
    return True


def _quarter_milestones(total):
    """Iteration counts at 25%, 50%, and 75% of *total* (excluding the first call)."""
    if total is None or total < 2:
        return frozenset()
    return frozenset(
        min(total, max(1, (total * i + 3) // 4)) for i in range(1, 4)
    )


@contextlib.contextmanager
def _profile_depth_scope():
    """Indent nested profiling output one level deeper."""
    if not hasattr(_profile_depth, "depth"):
        _profile_depth.depth = 0
    _profile_depth.depth += 1
    try:
        yield
    finally:
        _profile_depth.depth -= 1


def profile_method(func):
    """Decorator to profile method execution time and memory usage - checks self.profile."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if hasattr(args[0], "profile") and args[0].profile:
            print_profile = not hasattr(args[0], "global_rank") or args[0].global_rank == 0

            if not hasattr(_profile_depth, "depth"):
                _profile_depth.depth = 0
            _profile_depth.depth += 1
            indent = "  " * (_profile_depth.depth - 1)

            start_time = time.time()
            start_memory = _get_memory_usage()
            result = func(*args, **kwargs)
            end_time = time.time()
            end_memory = _get_memory_usage()
            execution_time = end_time - start_time

            if print_profile:
                if start_memory is not None and end_memory is not None:
                    memory_diff = end_memory - start_memory
                    print(
                        f"{indent}{func.__name__} took {execution_time:.5f} seconds, "
                        f"memory: {end_memory:.2f} MB (Δ: {memory_diff:+.2f} MB)"
                    )
                elif end_memory is not None:
                    print(
                        f"{indent}{func.__name__} took {execution_time:.5f} seconds, "
                        f"memory: {end_memory:.2f} MB"
                    )
                else:
                    print(f"{indent}{func.__name__} took {execution_time:.5f} seconds")

            _profile_depth.depth -= 1
            return result
        return func(*args, **kwargs)

    return wrapper


def profile_function(profile=False):
    """Decorator to profile function execution time - only active when profile is True."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if profile:
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"⏱️  {func.__name__} took {execution_time:.2f} seconds")
                return result
            return func(*args, **kwargs)

        return wrapper

    return decorator


class ProfileLoopTimer:
    """Low-level per-iteration accumulator (prefer :func:`profile_loop`)."""

    def __init__(
        self,
        owner,
        label,
        *,
        interim_reports=True,
        total=None,
        on_interim=None,
    ):
        self.owner = owner
        self.label = label
        self.interim_reports = interim_reports
        self.total = total
        self.on_interim = on_interim
        self.count = 0
        self.elapsed = 0.0
        self._last_reported_count = 0

    @contextlib.contextmanager
    def track(self):
        if not _profile_enabled(self.owner):
            yield
            return
        start = time.time()
        try:
            yield
        finally:
            self.elapsed += time.time() - start
            self.count += 1
            self._maybe_report_interim()

    def _maybe_report_interim(self):
        if not _profile_enabled(self.owner) or self.count == 0:
            return
        if not self.interim_reports:
            return
        first = self.count == 1
        quarter = self.count in _quarter_milestones(self.total)
        if first or quarter:
            self.report(interim=True, initial_estimate=first)
            if self.on_interim is not None:
                self.on_interim(interim=True)

    def report(self, interim=False, initial_estimate=False, indent_depth=None):
        if not _profile_enabled(self.owner) or self.count == 0:
            return
        if not interim and self.count == self._last_reported_count:
            return
        if indent_depth is None:
            # Loop-level lines sit one indent under their parent context.
            depth = max(get_profile_depth() - 1, 0)
        else:
            depth = indent_depth
        indent = "  " * depth
        mean = self.elapsed / self.count
        if initial_estimate:
            if self.total is not None:
                expected_total = mean * self.total
                print(
                    f"{indent}{self.label} (estimate after 1 call): "
                    f"~{expected_total:.0f}s expected total "
                    f"({mean:.3f}s/call, {self.total} calls)"
                )
            else:
                print(
                    f"{indent}{self.label} (estimate after 1 call): "
                    f"{mean:.3f}s/call"
                )
        else:
            if self.total is not None:
                progress = f"{self.count}/{self.total}"
                remaining = max(self.total - self.count, 0)
                eta = f", ~{remaining * mean:.0f}s remaining" if remaining > 0 else ""
            else:
                progress = str(self.count)
                eta = ""
            tag = " (in progress)" if interim else ""
            print(
                f"{indent}{self.label}{tag}: {progress} calls, "
                f"{self.elapsed:.1f}s elapsed, {mean:.3f}s mean{eta}"
            )
        self._last_reported_count = self.count


class ProfileTimerGroup:
    """Named substage timers inside a :func:`profile_loop` (e.g. LikelihoodDataset vs nf_loss)."""

    def __init__(self, owner):
        self.owner = owner
        self._timers = {}

    def track(self, label):
        if label not in self._timers:
            self._timers[label] = ProfileLoopTimer(self.owner, label, interim_reports=False)
        return self._timers[label].track()

    def report_all(self, interim=False, indent_depth=None):
        if not _profile_enabled(self.owner):
            return
        depth = get_profile_depth() if indent_depth is None else indent_depth
        for label in sorted(self._timers):
            self._timers[label].report(interim=interim, indent_depth=depth)


class _ProfileLoopIterator:
    """Iterator wrapper that times each loop iteration (internal)."""

    def __init__(
        self,
        owner,
        label,
        iterable,
        *,
        total=None,
        on_interim=None,
    ):
        self.owner = owner
        self.label = label
        self.iterable = iterable
        self.total = total
        self.on_interim = on_interim

    def __iter__(self):
        if not _profile_enabled(self.owner):
            yield from self.iterable
            return
        total = self.total
        if total is None:
            try:
                total = len(self.iterable)
            except TypeError:
                pass
        timer = ProfileLoopTimer(
            self.owner,
            self.label,
            total=total,
            on_interim=self.on_interim,
        )
        with _profile_depth_scope():
            try:
                for item in self.iterable:
                    with timer.track():
                        yield item
            finally:
                timer.report()


def _resolve_profile_loop_on_interim(bound, on_interim, on_interim_from):
    if on_interim is not None:
        return on_interim
    if on_interim_from is None:
        return None
    obj = bound.arguments.get(on_interim_from)
    if obj is None:
        return None
    return getattr(obj, "report_all", obj)


def _profile_loop_decorator(
    label,
    *,
    total=None,
    total_from=None,
    on_interim=None,
    on_interim_from=None,
):
    """Wrap a generator *method* so each ``yield`` is timed (internal)."""

    def decorator(gen_func):
        @functools.wraps(gen_func)
        def wrapper(self, *args, **kwargs):
            if not _profile_enabled(self):
                yield from gen_func(self, *args, **kwargs)
                return
            bound = inspect.signature(gen_func).bind(self, *args, **kwargs)
            bound.apply_defaults()
            n_total = total if total is not None else bound.arguments.get(total_from)
            interim = _resolve_profile_loop_on_interim(bound, on_interim, on_interim_from)
            timer = ProfileLoopTimer(
                self,
                label,
                total=n_total,
                on_interim=interim,
            )
            with _profile_depth_scope():
                try:
                    for item in gen_func(self, *args, **kwargs):
                        with timer.track():
                            yield item
                finally:
                    timer.report()

        return wrapper

    return decorator


def profile_loop(
    owner,
    label=None,
    iterable=None,
    *,
    total=None,
    on_interim=None,
    total_from=None,
    on_interim_from=None,
):
    """Profile a hot loop — the loop analogue of :func:`profile_method`.

    **Iterable form** (drop-in replacement for ``for item in iterable:``)::

        for j in profile_loop(self, "per-design", range(n_designs), total=n_designs):
            ...

    **Decorator form** (generator methods only)::

        @profile_loop("per-design", total_from="n_designs")
        def _design_indices(self, n_designs):
            yield from range(n_designs)

        for j in self._design_indices(n_designs):
            ...
    """
    if label is None and iterable is None and isinstance(owner, str):
        return _profile_loop_decorator(
            owner,
            total=total,
            total_from=total_from,
            on_interim=on_interim,
            on_interim_from=on_interim_from,
        )
    if iterable is None:
        raise TypeError(
            "profile_loop(owner, label, iterable, ...) or @profile_loop('label') on a generator"
        )
    return _ProfileLoopIterator(
        owner,
        label,
        iterable,
        total=total,
        on_interim=on_interim,
    )


@contextlib.contextmanager
def profile_section(owner, label):
    """Time a single block once; print one line on exit when owner.profile is set."""
    if not _profile_enabled(owner):
        yield
        return
    indent = "  " * get_profile_depth()
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"{indent}{label} took {elapsed:.5f} seconds")
