"""Gamepad controller for teleoperation."""

import sys


class GamepadController:
    """
    Gamepad reader with custom control scheme.
    
    Controls:
        X Button: Toggle clutch
        Left Stick: Translation + Arc sweep
        Right Stick: Pitch + Roll
        D-pad: Vertical motion
        Triggers: Gripper open/close
    """
    
    def __init__(self, deadzone: float = 0.15):
        self.deadzone = deadzone
        self.device = None
        self._is_hid = False
        
        # Sticks (-1 to 1)
        self.left_x = self.left_y = self.right_x = self.right_y = 0.0
        
        # Triggers (0 to 1)
        self.left_trigger = self.right_trigger = 0.0
        
        # Buttons
        self.left_bumper = self.right_bumper = False
        self.button_b = self.button_x = False
        self._prev_x = False
        
        # D-pad
        self.dpad_up = self.dpad_down = self.dpad_left = self.dpad_right = False
        
    def connect(self):
        """Connect to gamepad via HID (macOS) or pygame (Linux/Windows)."""
        if sys.platform == "darwin":
            self._connect_hid()
        else:
            self._connect_pygame()
    
    def _connect_hid(self):
        import hid
        for device in hid.enumerate():
            name = device.get("product_string", "")
            if any(c in name for c in ["Xbox", "Controller", "Logitech", "PS4", "PS5"]):
                print(f"Found gamepad: {name}")
                self.device = hid.device()
                self.device.open_path(device["path"])
                self.device.set_nonblocking(1)
                self._is_hid = True
                return
        raise RuntimeError("No gamepad found. Make sure it's connected.")
    
    def _connect_pygame(self):
        import pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad found. Make sure it's connected.")
        self.device = pygame.joystick.Joystick(0)
        self.device.init()
        self._is_hid = False
        print(f"Found gamepad: {self.device.get_name()}")
    
    def update(self):
        """Read latest gamepad state."""
        if self._is_hid:
            self._update_hid()
        else:
            self._update_pygame()
    
    def _apply_deadzone(self):
        if abs(self.left_x) < self.deadzone: self.left_x = 0
        if abs(self.left_y) < self.deadzone: self.left_y = 0
        if abs(self.right_x) < self.deadzone: self.right_x = 0
        if abs(self.right_y) < self.deadzone: self.right_y = 0
    
    def _update_hid(self):
        if not self.device:
            return
        
        # Read multiple times for stable reading
        data = None
        for _ in range(10):
            d = self.device.read(64)
            if d:
                data = d
        
        if not data or len(data) < 12:
            return
        
        try:
            # Xbox controller HID mapping (16-bit sticks, 10-bit triggers)
            self.left_x = (int.from_bytes(data[1:3], 'little') - 32768) / 32768.0
            self.left_y = (int.from_bytes(data[3:5], 'little') - 32768) / 32768.0
            self.right_x = (int.from_bytes(data[5:7], 'little') - 32768) / 32768.0
            self.right_y = (int.from_bytes(data[7:9], 'little') - 32768) / 32768.0
            self.left_trigger = int.from_bytes(data[9:11], 'little') / 1023.0
            self.right_trigger = int.from_bytes(data[11:13], 'little') / 1023.0
            
            if len(data) > 14:
                buttons1, buttons2 = data[13], data[14]
                self.left_bumper = bool(buttons2 & 0x40)
                self.right_bumper = bool(buttons2 & 0x80)
                self.button_b = bool(buttons1 & 0x02)
                self.button_x = bool(buttons2 & 0x04) or bool(buttons2 & 0x08)
                
                dpad = buttons1 & 0x0F
                self.dpad_up = dpad in [1, 2, 8]
                self.dpad_down = dpad in [4, 5, 6]
                self.dpad_left = dpad in [6, 7, 8]
                self.dpad_right = dpad in [2, 3, 4]
        except Exception:
            # Fallback: Logitech-style 8-bit layout
            self.left_x = (data[1] - 128) / 128.0
            self.left_y = (data[2] - 128) / 128.0
            self.right_x = (data[3] - 128) / 128.0
            self.right_y = (data[4] - 128) / 128.0
            if len(data) > 6:
                t = data[6]
                self.left_trigger = 1.0 if t in [4, 6, 12, 14] else 0.0
                self.right_trigger = 1.0 if t in [8, 10, 12, 14] else 0.0
            if len(data) > 5:
                self.button_b = bool(data[5] & 0x20)
        
        self._apply_deadzone()
    
    def _update_pygame(self):
        import pygame
        pygame.event.pump()
        
        self.left_x = self.device.get_axis(0)
        self.left_y = self.device.get_axis(1)
        self.right_x = self.device.get_axis(2) if self.device.get_numaxes() > 2 else self.device.get_axis(3)
        self.right_y = self.device.get_axis(3) if self.device.get_numaxes() > 3 else self.device.get_axis(4)
        
        if self.device.get_numaxes() > 5:
            self.left_trigger = (self.device.get_axis(4) + 1) / 2
            self.right_trigger = (self.device.get_axis(5) + 1) / 2
        
        self.left_bumper = self.device.get_button(4) if self.device.get_numbuttons() > 4 else False
        self.right_bumper = self.device.get_button(5) if self.device.get_numbuttons() > 5 else False
        self.button_b = self.device.get_button(1) if self.device.get_numbuttons() > 1 else False
        self.button_x = self.device.get_button(2) if self.device.get_numbuttons() > 2 else False
        
        self._apply_deadzone()
    
    def check_x_pressed(self) -> bool:
        """Returns True on rising edge of X button press."""
        pressed = self.button_x and not self._prev_x
        self._prev_x = self.button_x
        return pressed
    
    def disconnect(self):
        if self.device and self._is_hid:
            self.device.close()
