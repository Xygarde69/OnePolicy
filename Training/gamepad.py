"""DualSense -> velocity command, via the Linux joystick API.

Right stick: up/down = forward/back, left/right = sideways. L1 turns left,
R1 turns right (both held: no turn).

Reads /dev/input/jsN directly (no pygame/evdev). Each event is 8 bytes:
u32 time [ms], s16 value, u8 type (0x01 button, 0x02 axis, |0x80 initial
state), u8 number. Axis and button numbers are looked up from the kernel's
maps (JSIOCGAXMAP / JSIOCGBTNMAP) by their evdev codes, so they don't depend
on the driver's ordering; with hid-playstation the right stick is axes 3/4
(ABS_RX/ABS_RY, +Y down) and L1/R1 are buttons 4/5 (BTN_TL/BTN_TR).
"""

import array
import fcntl
import struct
import threading
from pathlib import Path

_EVENT = struct.Struct("IhBB")
_BUTTON, _AXIS = 0x01, 0x02

ABS_RX, ABS_RY = 0x03, 0x04
BTN_TL, BTN_TR = 0x136, 0x137  # L1, R1


def _ior(nr: int, size: int) -> int:
  return (2 << 30) | (size << 16) | (ord("j") << 8) | nr


def _input_maps(f) -> tuple[list[int], list[int]]:
  """evdev code of each joystick axis and button number."""
  counts = array.array("B", [0, 0])
  fcntl.ioctl(f, _ior(0x11, 1), memoryview(counts)[0:1])  # JSIOCGAXES
  fcntl.ioctl(f, _ior(0x12, 1), memoryview(counts)[1:2])  # JSIOCGBUTTONS
  axes = array.array("B", [0] * 64)  # ABS_CNT
  fcntl.ioctl(f, _ior(0x32, len(axes)), axes)  # JSIOCGAXMAP
  buttons = array.array("H", [0] * (0x2FF - 0x100 + 1))  # KEY_MAX - BTN_MISC + 1
  fcntl.ioctl(f, _ior(0x34, len(buttons) * 2), buttons)  # JSIOCGBTNMAP
  return list(axes[: counts[0]]), list(buttons[: counts[1]])


class DualSenseGamepad:
  """Background reader: right stick translates, L1/R1 turn."""

  def __init__(self, device: str = "/dev/input/js0", deadzone: float = 0.1):
    path = Path(device)
    if not path.exists():
      raise FileNotFoundError(
        f"No joystick at {device}. Connect the DualSense (USB or Bluetooth) "
        "or pass --joystick-device."
      )
    self.deadzone = deadzone
    self._file = open(path, "rb")  # Needs read access (the 'input' group).
    axes, buttons = _input_maps(self._file)
    try:
      self._rx, self._ry = axes.index(ABS_RX), axes.index(ABS_RY)
      self._l1, self._r1 = buttons.index(BTN_TL), buttons.index(BTN_TR)
    except ValueError as e:
      raise RuntimeError(f"{device} has no right stick or L1/R1: {e}") from e
    self._axes = {self._rx: 0.0, self._ry: 0.0}
    self._buttons = {self._l1: False, self._r1: False}
    self._lock = threading.Lock()
    threading.Thread(target=self._read, daemon=True).start()

  def _read(self) -> None:
    try:
      while data := self._file.read(_EVENT.size):
        _, value, kind, number = _EVENT.unpack(data)
        with self._lock:
          if kind & _AXIS and number in self._axes:
            self._axes[number] = value / 32767.0
          elif kind & _BUTTON and number in self._buttons:
            self._buttons[number] = bool(value)
    except OSError:  # Controller disconnected: leave the command at rest.
      with self._lock:
        self._axes = dict.fromkeys(self._axes, 0.0)
        self._buttons = dict.fromkeys(self._buttons, False)

  def _shape(self, x: float) -> float:
    """Remove the deadzone and rescale the rest to [-1, 1]."""
    if abs(x) < self.deadzone:
      return 0.0
    return (abs(x) - self.deadzone) / (1.0 - self.deadzone) * (1.0 if x > 0 else -1.0)

  def read(self) -> tuple[float, float, float]:
    """(forward, left, turn left), each in [-1, 1].

    Stick up = forward, stick left = left, L1 = turn left, R1 = turn right.
    """
    with self._lock:
      x, y = self._axes[self._rx], self._axes[self._ry]
      turn = float(self._buttons[self._l1]) - float(self._buttons[self._r1])
    return self._shape(-y), self._shape(-x), turn
