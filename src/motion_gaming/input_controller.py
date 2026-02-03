"""Input simulation for sending keypresses to Windows."""

import time
from typing import Optional

from motion_gaming.gesture_recognizer import Direction


# Virtual key codes for Windows
VK_CODES = {
    "W": 0x57,
    "A": 0x41,
    "S": 0x53,
    "D": 0x44,
    "1": 0x31,
    "2": 0x32,
    "3": 0x33,
    "4": 0x34,
    "5": 0x35,
}

# Direction to key mapping
DIRECTION_KEYS: dict[Direction, list[str]] = {
    Direction.NORTH: ["W"],
    Direction.SOUTH: ["S"],
    Direction.WEST: ["A"],
    Direction.EAST: ["D"],
    Direction.NORTHEAST: ["W", "D"],
    Direction.NORTHWEST: ["W", "A"],
    Direction.SOUTHEAST: ["S", "D"],
    Direction.SOUTHWEST: ["S", "A"],
}


class InputController:
    """Controls keyboard input simulation for game control."""

    ABILITY_REPEAT_INTERVAL = 0.250  # 250ms

    def __init__(self) -> None:
        """Initialize the input controller."""
        self._current_movement_keys: set[str] = set()
        self._current_ability: int = 0
        self._last_ability_time: float = 0.0
        self._windows_input: Optional["WindowsInput"] = None

        # Try to initialize Windows input
        try:
            self._windows_input = WindowsInput()
        except OSError:
            # Not on Windows, will use mock/print mode
            pass

    def set_movement(self, direction: Optional[Direction]) -> None:
        """Set movement based on pointing direction.

        Args:
            direction: Pointing direction, or None to stop.
        """
        new_keys = set(DIRECTION_KEYS.get(direction, [])) if direction else set()

        # Release keys that are no longer needed
        for key in self._current_movement_keys - new_keys:
            self._release_key(key)

        # Press keys that are newly needed
        for key in new_keys - self._current_movement_keys:
            self._press_key(key)

        self._current_movement_keys = new_keys

    def set_ability(self, finger_count: int) -> None:
        """Set ability key based on finger count.

        Args:
            finger_count: Number of fingers (0-5). 0 means no ability.
        """
        current_time = time.time()

        if finger_count == 0:
            # Release current ability
            if self._current_ability > 0:
                self._release_key(str(self._current_ability))
            self._current_ability = 0
            return

        if finger_count != self._current_ability:
            # Changed ability - release old, press new
            if self._current_ability > 0:
                self._release_key(str(self._current_ability))
            self._press_key(str(finger_count))
            self._current_ability = finger_count
            self._last_ability_time = current_time
        elif current_time - self._last_ability_time >= self.ABILITY_REPEAT_INTERVAL:
            # Same ability, repeat interval elapsed - tap again
            self._release_key(str(finger_count))
            self._press_key(str(finger_count))
            self._last_ability_time = current_time

    def release_all(self) -> None:
        """Release all held keys."""
        for key in self._current_movement_keys:
            self._release_key(key)
        self._current_movement_keys = set()

        if self._current_ability > 0:
            self._release_key(str(self._current_ability))
            self._current_ability = 0

    def _press_key(self, key: str) -> None:
        """Press a key."""
        if self._windows_input:
            self._windows_input.press_key(VK_CODES[key])
        # Silently ignore if not on Windows

    def _release_key(self, key: str) -> None:
        """Release a key."""
        if self._windows_input:
            self._windows_input.release_key(VK_CODES[key])
        # Silently ignore if not on Windows


class WindowsInput:
    """Windows-specific input simulation using SendInput."""

    def __init__(self) -> None:
        """Initialize Windows input. Raises OSError if not on Windows."""
        import ctypes
        import ctypes.wintypes

        self._user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        self._ctypes = ctypes

        # Input structure for SendInput
        self._KEYEVENTF_KEYUP = 0x0002
        self._INPUT_KEYBOARD = 1

    def press_key(self, vk_code: int) -> None:
        """Press a key down."""
        self._send_key_event(vk_code, 0)

    def release_key(self, vk_code: int) -> None:
        """Release a key."""
        self._send_key_event(vk_code, self._KEYEVENTF_KEYUP)

    def _send_key_event(self, vk_code: int, flags: int) -> None:
        """Send a keyboard event."""
        import ctypes

        # KEYBDINPUT structure
        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]

        # INPUT structure
        class INPUT(ctypes.Structure):
            _fields_ = [
                ("type", ctypes.c_ulong),
                ("ki", KEYBDINPUT),
            ]

        ki = KEYBDINPUT(
            wVk=vk_code,
            wScan=0,
            dwFlags=flags,
            time=0,
            dwExtraInfo=ctypes.pointer(ctypes.c_ulong(0)),
        )
        inp = INPUT(type=self._INPUT_KEYBOARD, ki=ki)

        self._user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
