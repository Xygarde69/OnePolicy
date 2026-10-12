"""DualSense right stick -> velocity command, via the Linux joystick API.

Reads /dev/input/jsN directly (no pygame/evdev). Each event is 8 bytes:
u32 time [ms], s16 value, u8 type (0x01 button, 0x02 axis, |0x80 initial
state), u8 number. With the kernel's hid-playstation driver the axes are
LX, LY, L2, RX, RY, R2, D-pad X, D-pad Y; the right stick is 3 (RX) and 4 (RY),
each in [-32767, 32767] with +Y pointing down.
"""

import struct
import threading
from pathlib import Path

_EVENT = struct.Struct("IhBB")
_AXIS = 0x02
RIGHT_X = 3
RIGHT_Y = 4


class DualSenseRightStick:
  """Background reader for the right stick: up/down drives, left/right turns."""

  def __init__(self, device: str = "/dev/input/js0", deadzone: float = 0.1):
    path = Path(device)
    if not path.exists():
      raise FileNotFoundError(
        f"No joystick at {device}. Connect the DualSense (USB or Bluetooth) "
        "or pass --joystick-device."
      )
    self.deadzone = deadzone
    self._axes = {RIGHT_X: 0.0, RIGHT_Y: 0.0}
    self._lock = threading.Lock()
    self._file = open(path, "rb")  # Needs read access (the 'input' group).
    threading.Thread(target=self._read, daemon=True).start()

  def _read(self) -> None:
    try:
      while data := self._file.read(_EVENT.size):
        _, value, kind, number = _EVENT.unpack(data)
        if kind & _AXIS and number in self._axes:
          with self._lock:
            self._axes[number] = value / 32767.0
    except OSError:  # Controller disconnected: leave the command at rest.
      with self._lock:
        self._axes = dict.fromkeys(self._axes, 0.0)

  def _shape(self, x: float) -> float:
    """Remove the deadzone and rescale the rest to [-1, 1]."""
    if abs(x) < self.deadzone:
      return 0.0
    return (abs(x) - self.deadzone) / (1.0 - self.deadzone) * (1.0 if x > 0 else -1.0)

  def read(self) -> tuple[float, float]:
    """(forward, turn left) in [-1, 1]: stick up = forward, stick left = turn left."""
    with self._lock:
      x, y = self._axes[RIGHT_X], self._axes[RIGHT_Y]
    return self._shape(-y), self._shape(-x)
