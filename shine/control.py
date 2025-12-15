from os import environ
# NOTE: Ensure you run 'sudo pigpiod' in your terminal before running this script
environ["GPIOZERO_PIN_FACTORY"] = "pigpio"

from gpiozero import Servo
import time
import warnings

warnings.filterwarnings("ignore")

servo1 = Servo(18, min_pulse_width=0.6/1000, max_pulse_width=2.3/1000)
servo2 = Servo(12, min_pulse_width=0.6/1000, max_pulse_width=2.3/1000)

def move_servo(servo_type, angle):
    # --- FIX #1: REMOVED time.sleep and servo detach for responsiveness ---
    if not -90 <= angle <= 90:
        return
    # This keeps the servo motor powered on and holding its position
    servo_type.value = angle / 90.0

def convert_angle(servo_type, coordinate):
    if servo_type == servo1:    
        # OLD: angle = coordinate // 9.142
        # NEW: Change 9.142 to 25.0 (Higher number = Slower movement)
        angle = coordinate / 12.0 
        return max(-90, min(90, angle))
    elif servo_type == servo2:
        # OLD: angle = coordinate // 6.857
        # NEW: Change 6.857 to 20.0 (Higher number = Slower movement)
        angle = coordinate / 12.0
        return max(-90, min(90, angle))

def return_center(servo_type, hold_time=0.1):
    servo_type.value = 0.0
    time.sleep(hold_time)
    # Detach only when centering at start/end
    servo_type.value = None
