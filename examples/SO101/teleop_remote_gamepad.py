"""Remote gamepad controller that receives input via WebSocket."""


class RemoteGamepadController:
    """
    Gamepad controller that receives state from web interface.
    
    Mimics GamepadController interface but gets input from WebTeleoperationServer.
    """
    
    def __init__(self, web_server, deadzone: float = 0.15):
        self.web_server = web_server
        self.deadzone = deadzone
        
        # Sticks (-1 to 1)
        self.left_x = self.left_y = self.right_x = self.right_y = 0.0
        
        # Triggers (0 to 1)
        self.left_trigger = self.right_trigger = 0.0
        
        # Buttons
        self.left_bumper = self.right_bumper = False
        self.button_a = self.button_b = self.button_x = self.button_y = False
        self._prev_x = False
        self._prev_a = False
        
        # D-pad
        self.dpad_up = self.dpad_down = self.dpad_left = self.dpad_right = False
    
    def connect(self):
        """No-op for remote gamepad - connection handled by web server."""
        print("[Remote] Waiting for web client connection...")
    
    def disconnect(self):
        """No-op for remote gamepad."""
        pass
    
    def update(self):
        """Update state from web server."""
        state = self.web_server.get_gamepad_state()
        
        # Apply deadzone to sticks
        self.left_x = self._apply_deadzone(state.get("left_x", 0.0))
        self.left_y = self._apply_deadzone(state.get("left_y", 0.0))
        self.right_x = self._apply_deadzone(state.get("right_x", 0.0))
        self.right_y = self._apply_deadzone(state.get("right_y", 0.0))
        
        # Triggers (no deadzone)
        self.left_trigger = max(0.0, state.get("left_trigger", 0.0))
        self.right_trigger = max(0.0, state.get("right_trigger", 0.0))
        
        # Buttons
        self.button_a = state.get("button_a", False)
        self.button_b = state.get("button_b", False)
        self.button_x = state.get("button_x", False)
        self.button_y = state.get("button_y", False)
        self.left_bumper = state.get("button_lb", False)
        self.right_bumper = state.get("button_rb", False)
        
        # D-pad
        self.dpad_up = state.get("dpad_up", False)
        self.dpad_down = state.get("dpad_down", False)
        self.dpad_left = state.get("dpad_left", False)
        self.dpad_right = state.get("dpad_right", False)
    
    def check_x_pressed(self) -> bool:
        """Returns True on rising edge of X or A button press (clutch toggle)."""
        x_pressed = self.button_x and not self._prev_x
        a_pressed = self.button_a and not self._prev_a
        self._prev_x = self.button_x
        self._prev_a = self.button_a
        return x_pressed or a_pressed
    
    def _apply_deadzone(self, value: float) -> float:
        """Apply deadzone to axis value."""
        if abs(value) < self.deadzone:
            return 0.0
        return value
