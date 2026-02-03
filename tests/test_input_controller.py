"""Tests for input controller."""

import time
from unittest.mock import MagicMock

import pytest

from motion_gaming.gesture_recognizer import Direction
from motion_gaming.input_controller import (
    DIRECTION_KEYS,
    VK_CODES,
    InputController,
)


@pytest.fixture
def controller() -> InputController:
    """Create an InputController with mocked Windows input."""
    ctrl = InputController()
    ctrl._windows_input = MagicMock()
    return ctrl


class TestSetMovement:
    """Tests for movement key control."""

    def test_direction_north_presses_w(self, controller: InputController) -> None:
        """Pointing north should press W key."""
        controller.set_movement(Direction.NORTH)

        controller._windows_input.press_key.assert_called_once_with(VK_CODES["W"])
        controller._windows_input.release_key.assert_not_called()

    def test_direction_south_presses_s(self, controller: InputController) -> None:
        """Pointing south should press S key."""
        controller.set_movement(Direction.SOUTH)

        controller._windows_input.press_key.assert_called_once_with(VK_CODES["S"])

    def test_direction_east_presses_d(self, controller: InputController) -> None:
        """Pointing east should press D key."""
        controller.set_movement(Direction.EAST)

        controller._windows_input.press_key.assert_called_once_with(VK_CODES["D"])

    def test_direction_west_presses_a(self, controller: InputController) -> None:
        """Pointing west should press A key."""
        controller.set_movement(Direction.WEST)

        controller._windows_input.press_key.assert_called_once_with(VK_CODES["A"])

    def test_diagonal_presses_two_keys(self, controller: InputController) -> None:
        """Diagonal direction should press two keys."""
        controller.set_movement(Direction.NORTHEAST)

        calls = controller._windows_input.press_key.call_args_list
        pressed_codes = {call[0][0] for call in calls}
        assert pressed_codes == {VK_CODES["W"], VK_CODES["D"]}

    def test_none_releases_all_movement_keys(self, controller: InputController) -> None:
        """Setting direction to None should release held keys."""
        controller.set_movement(Direction.NORTH)
        controller._windows_input.reset_mock()

        controller.set_movement(None)

        controller._windows_input.release_key.assert_called_once_with(VK_CODES["W"])
        controller._windows_input.press_key.assert_not_called()

    def test_direction_change_releases_old_presses_new(
        self, controller: InputController
    ) -> None:
        """Changing direction should release old keys and press new ones."""
        controller.set_movement(Direction.NORTH)
        controller._windows_input.reset_mock()

        controller.set_movement(Direction.SOUTH)

        controller._windows_input.release_key.assert_called_once_with(VK_CODES["W"])
        controller._windows_input.press_key.assert_called_once_with(VK_CODES["S"])

    def test_same_direction_no_change(self, controller: InputController) -> None:
        """Same direction should not re-press keys."""
        controller.set_movement(Direction.NORTH)
        controller._windows_input.reset_mock()

        controller.set_movement(Direction.NORTH)

        controller._windows_input.press_key.assert_not_called()
        controller._windows_input.release_key.assert_not_called()

    def test_diagonal_to_cardinal_releases_one_key(
        self, controller: InputController
    ) -> None:
        """Going from diagonal to cardinal should release one key."""
        controller.set_movement(Direction.NORTHEAST)  # W + D
        controller._windows_input.reset_mock()

        controller.set_movement(Direction.NORTH)  # Just W

        # Should release D, keep W pressed
        controller._windows_input.release_key.assert_called_once_with(VK_CODES["D"])
        controller._windows_input.press_key.assert_not_called()


class TestSetAbility:
    """Tests for ability key control."""

    def test_finger_count_presses_key(self, controller: InputController) -> None:
        """Finger count should press corresponding number key."""
        controller.set_ability(1)
        controller._windows_input.press_key.assert_called_with(VK_CODES["1"])

        controller._windows_input.reset_mock()
        controller.set_ability(0)  # Release
        controller.set_ability(3)
        controller._windows_input.press_key.assert_called_with(VK_CODES["3"])

    def test_zero_fingers_releases_ability(self, controller: InputController) -> None:
        """Zero fingers should release held ability key."""
        controller.set_ability(2)
        controller._windows_input.reset_mock()

        controller.set_ability(0)

        controller._windows_input.release_key.assert_called_once_with(VK_CODES["2"])
        controller._windows_input.press_key.assert_not_called()

    def test_changing_ability_releases_old_presses_new(
        self, controller: InputController
    ) -> None:
        """Changing finger count should release old and press new."""
        controller.set_ability(1)
        controller._windows_input.reset_mock()

        controller.set_ability(4)

        controller._windows_input.release_key.assert_called_once_with(VK_CODES["1"])
        controller._windows_input.press_key.assert_called_once_with(VK_CODES["4"])

    def test_same_ability_no_immediate_repeat(
        self, controller: InputController
    ) -> None:
        """Same finger count should not immediately repeat."""
        controller.set_ability(2)
        controller._windows_input.reset_mock()

        controller.set_ability(2)

        controller._windows_input.press_key.assert_not_called()
        controller._windows_input.release_key.assert_not_called()

    def test_ability_repeats_after_interval(self, controller: InputController) -> None:
        """Same ability should repeat after 250ms interval."""
        controller.set_ability(3)
        controller._windows_input.reset_mock()

        # Simulate time passing
        controller._last_ability_time -= 0.3  # 300ms ago

        controller.set_ability(3)

        # Should release and re-press for repeat
        controller._windows_input.release_key.assert_called_once_with(VK_CODES["3"])
        controller._windows_input.press_key.assert_called_once_with(VK_CODES["3"])

    def test_ability_no_repeat_before_interval(
        self, controller: InputController
    ) -> None:
        """Same ability should not repeat before 250ms interval."""
        controller.set_ability(3)
        controller._windows_input.reset_mock()

        # Simulate only 100ms passing
        controller._last_ability_time = time.time() - 0.1

        controller.set_ability(3)

        controller._windows_input.press_key.assert_not_called()
        controller._windows_input.release_key.assert_not_called()


class TestReleaseAll:
    """Tests for release_all method."""

    def test_releases_movement_keys(self, controller: InputController) -> None:
        """release_all should release held movement keys."""
        controller.set_movement(Direction.NORTHEAST)  # W + D
        controller._windows_input.reset_mock()

        controller.release_all()

        release_calls = controller._windows_input.release_key.call_args_list
        released_codes = {call[0][0] for call in release_calls}
        assert VK_CODES["W"] in released_codes
        assert VK_CODES["D"] in released_codes

    def test_releases_ability_key(self, controller: InputController) -> None:
        """release_all should release held ability key."""
        controller.set_ability(5)
        controller._windows_input.reset_mock()

        controller.release_all()

        controller._windows_input.release_key.assert_called_with(VK_CODES["5"])

    def test_clears_state(self, controller: InputController) -> None:
        """release_all should clear internal state."""
        controller.set_movement(Direction.NORTH)
        controller.set_ability(3)

        controller.release_all()

        assert controller._current_movement_keys == set()
        assert controller._current_ability == 0


class TestNoWindowsInput:
    """Tests for behavior when Windows API is unavailable."""

    def test_operations_succeed_without_windows(self) -> None:
        """Operations should not crash when Windows API unavailable."""
        ctrl = InputController()
        ctrl._windows_input = None  # Simulate non-Windows

        # These should all succeed silently
        ctrl.set_movement(Direction.NORTH)
        ctrl.set_ability(3)

        # State should still be tracked internally
        assert ctrl._current_movement_keys == {"W"}
        assert ctrl._current_ability == 3

        # release_all should also work
        ctrl.release_all()
        assert ctrl._current_movement_keys == set()
        assert ctrl._current_ability == 0


class TestDirectionKeyMapping:
    """Verify DIRECTION_KEYS mapping is complete."""

    def test_all_directions_mapped(self) -> None:
        """All Direction enum values should have key mappings."""
        for direction in Direction:
            assert direction in DIRECTION_KEYS
            assert len(DIRECTION_KEYS[direction]) >= 1
