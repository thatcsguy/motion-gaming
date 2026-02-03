"""Input simulation for sending keypresses to games via DirectInput."""

import time
from typing import Optional

from motion_gaming.gesture_recognizer import Direction

# Try to import pydirectinput for DirectInput scan code support
# This is required for games like FFXIV that ignore virtual key codes
try:
    import pydirectinput

    pydirectinput.PAUSE = 0  # Disable default pause between actions
    HAS_DIRECTINPUT = True
except ImportError:
    HAS_DIRECTINPUT = False

# Direction to key mapping (lowercase for pydirectinput)
DIRECTION_KEYS: dict[Direction, list[str]] = {
    Direction.NORTH: ["w"],
    Direction.SOUTH: ["s"],
    Direction.WEST: ["a"],
    Direction.EAST: ["d"],
    Direction.NORTHEAST: ["w", "d"],
    Direction.NORTHWEST: ["w", "a"],
    Direction.SOUTHEAST: ["s", "d"],
    Direction.SOUTHWEST: ["s", "a"],
}


class InputController:
    """Controls keyboard input simulation for game control."""

    ABILITY_REPEAT_INTERVAL = 0.250  # 250ms

    def __init__(self) -> None:
        """Initialize the input controller."""
        self._current_movement_keys: set[str] = set()
        self._current_ability: int = 0
        self._last_ability_time: float = 0.0
        self._input_available = HAS_DIRECTINPUT

        if not self._input_available:
            print("Warning: pydirectinput not available. Install with: pip install pydirectinput")

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
        """Press a key using DirectInput."""
        if self._input_available:
            pydirectinput.keyDown(key)

    def _release_key(self, key: str) -> None:
        """Release a key using DirectInput."""
        if self._input_available:
            pydirectinput.keyUp(key)
