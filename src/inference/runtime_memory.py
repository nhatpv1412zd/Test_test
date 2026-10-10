"""Windows commit admission for the measured CPU CRAFT working set.

This is a preflight resource policy, not a reservation or a native fault handler.
Other processes can consume memory after the snapshot. Keep the pixel cap too.
"""
import sys


def _available_commit_bytes():
    if sys.platform != 'win32':
        return None
    import ctypes
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in (
                'total_phys', 'avail_phys', 'total_pagefile', 'avail_pagefile',
                'total_virtual', 'avail_virtual', 'avail_extended_virtual')]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(status.avail_pagefile)


def ocr_memory_admission(plan, workspace_factor, reserve_bytes):
    """Profiled CPU budget: four 64ch buffers plus 128 MiB by default.

    A/B traces need up to ~1.9 GB above the loaded models at CRAFT entry,
    despite a single buffer being only ~0.53 GB. This policy leaves headroom
    for overlapping feature maps, normalization buffers and native workspace.
    It is an estimate; neither total peak RAM nor all native faults are bounded.
    """
    available = _available_commit_bytes()
    required = plan['full_resolution_64ch_float32_bytes'] * workspace_factor + reserve_bytes
    return {'applied': available is not None, 'available_commit_bytes': available,
            'required_headroom_bytes': required, 'workspace_factor': workspace_factor,
            'reserve_bytes': reserve_bytes, 'allowed': available is None or available >= required}
