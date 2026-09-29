"""Opt our own processes out of Windows EcoQoS power throttling.

Windows 11 may run background processes (no foreground window — e.g. a
training script and its Node simulators) as "efficiency mode": efficiency
cores, low clock. That is a scheduling hint only; opting out changes no
results, only speed. Per process, via SetProcessInformation — no system
setting is touched. No-op on other platforms.
"""

import ctypes
import sys

_PROCESS_POWER_THROTTLING = 4  # ProcessPowerThrottling
_EXECUTION_SPEED = 0x1
_IGNORE_TIMER_RESOLUTION = 0x4


class _State(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]


def disable_throttling(handle=None):
    """handle: a process handle (e.g. Popen._handle); None = this process. Returns True on success."""
    if sys.platform != "win32":
        return False
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    k32.SetProcessInformation.restype = ctypes.c_int
    h = k32.GetCurrentProcess() if handle is None else int(handle)
    state = _State(1, _EXECUTION_SPEED | _IGNORE_TIMER_RESOLUTION, 0)  # control these, state off = not throttled
    return bool(k32.SetProcessInformation(h, _PROCESS_POWER_THROTTLING, ctypes.byref(state), ctypes.sizeof(state)))
