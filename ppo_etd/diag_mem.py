# Diagnostic only: periodic memory census of the main process (enabled via ERELELA_MEMDIAG=<path>).
import gc
import os
import sys
import threading
import time
from collections import Counter, defaultdict


def _rss_mb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS'):
                return int(line.split()[1]) // 1024
    return -1


def census(out, top=20):
    import numpy as np
    import torch as th
    t0 = time.time()
    tensors = defaultdict(lambda: [0, 0])
    arrays = defaultdict(lambda: [0, 0])
    types = Counter()
    seen_storage = set()
    for o in gc.get_objects():
        try:
            types[type(o).__name__] += 1
            if isinstance(o, th.Tensor):
                st = o.untyped_storage()
                key = (st.data_ptr(), o.device.type)
                if key in seen_storage:
                    continue
                seen_storage.add(key)
                k = (o.device.type, str(o.dtype), tuple(o.shape))
                tensors[k][0] += 1
                tensors[k][1] += st.nbytes()
            elif isinstance(o, np.ndarray) and o.base is None:
                k = (str(o.dtype), o.shape if o.ndim <= 4 else o.shape[:4])
                arrays[k][0] += 1
                arrays[k][1] += o.nbytes
        except Exception:
            pass
    tot_t = sum(v[1] for k, v in tensors.items() if k[0] == 'cpu')
    tot_a = sum(v[1] for v in arrays.values())
    out.write(f"=== {time.strftime('%F %T')} rss={_rss_mb()}MB cpu_tensors={tot_t>>20}MB ndarrays={tot_a>>20}MB "
              f"census={time.time()-t0:.1f}s\n")
    for k, v in sorted(tensors.items(), key=lambda kv: -kv[1][1])[:top]:
        out.write(f"  T {v[1]>>20:7d}MB x{v[0]:<7d} {k}\n")
    for k, v in sorted(arrays.items(), key=lambda kv: -kv[1][1])[:top]:
        out.write(f"  A {v[1]>>20:7d}MB x{v[0]:<7d} {k}\n")
    out.write("  types: " + ", ".join(f"{n}={c}" for n, c in types.most_common(25)) + "\n")
    out.flush()


def trace_diff(out, base, top=15):
    import tracemalloc
    snap = tracemalloc.take_snapshot().filter_traces([tracemalloc.Filter(False, tracemalloc.__file__)])
    stats = snap.compare_to(base, 'traceback')
    out.write(f"  tracemalloc traced={tracemalloc.get_traced_memory()[0]>>20}MB, top growth since start:\n")
    for st in stats[:top]:
        out.write(f"  + {st.size_diff>>20:6d}MB n={st.count_diff:+d}\n")
        for line in st.traceback.format()[-12:]:
            out.write(f"      {line}\n")
    out.flush()


def start(path, period=300):
    trace = bool(os.environ.get('ERELELA_TRACEMALLOC'))
    if trace:
        import tracemalloc
        tracemalloc.start(12)

    def loop():
        base = None
        with open(path, 'a') as out:
            while True:
                time.sleep(period)
                try:
                    census(out)
                    if trace:
                        import tracemalloc
                        if base is None:
                            base = tracemalloc.take_snapshot()
                        else:
                            trace_diff(out, base)
                except Exception as e:
                    out.write(f"census failed: {e}\n")
                    out.flush()
    threading.Thread(target=loop, daemon=True, name='memdiag').start()
