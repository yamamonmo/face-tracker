import cv2
import time
import random
# --- FIX: Changed 'vision' back to 'camera' to match your file name ---
from shine.vision import find_and_open, camera, end_camera
# --- FIX: Changed 'control' back to 'servo' to match your file name ---
from shine.control import move_servo, convert_angle, return_center, servo1, servo2

# --- Search Pattern Variables ---

SEARCH_DIRECTION = 1.0 # 1.0 for right, -1.0 for left
SEARCH_STEP_SIZE = 0.5 # Small angle change per loop (in degrees, determines speed)
SEARCH_LIMIT = 45.0 # How far from center (0.0) to scan (in degrees)
SEARCH_DELAY = 0.05 # Delay between search steps (seconds)
SEARCH_PAN_DIRECTION = 1.0   # 1.0 = Right, -1.0 = Left
SEARCH_TILT_DIRECTION = 1.0  # 1.0 = Up, -1.0 = Down

# SPEEDS
SEARCH_PAN_SPEED = 1.5       # How fast it looks left/right
SEARCH_TILT_SPEED = 0.6      # Slower speed for up/down (looks more natural)

# LIMITS (How far to look)
SEARCH_PAN_LIMIT = 60.0      # Left/Right range (degrees)
SEARCH_TILT_LIMIT = 25.0     # Up/Down range (degrees) - keep small to avoid looking at floor/ceiling
SEARCH_DELAY = 0.02

# --- PREDATOR MODE VARIABLES ---
# Targets: Where the predator wants to look next
target_pan = 0.0
target_tilt = 0.0

# State: Are we moving or waiting?
is_waiting = False
wait_start_time = 0

# Limits: How far it can look
LIMIT_PAN = 60.0
LIMIT_TILT = 25.0


def handle_predator(current_pan, current_tilt):
    global target_pan, target_tilt, is_waiting, wait_start_time

    # 1. CHECK: Are we waiting/staring?
    if is_waiting:
        # If we have waited long enough (random time between 0.1s and 0.8s)
        if time.time() - wait_start_time > random.uniform(0.1, 0.8):
            is_waiting = False # Wake up, time to move again
        else:
            return current_pan, current_tilt # Stay still

    # 2. CHECK: Are we close to the target?
    # If distance to target is less than 2 degrees, we "arrived"
    if abs(current_pan - target_pan) < 2.0 and abs(current_tilt - target_tilt) < 2.0:
        
        # Pick a NEW random target
        target_pan = random.uniform(-LIMIT_PAN, LIMIT_PAN)
        target_tilt = random.uniform(-LIMIT_TILT, LIMIT_TILT)
        
        # Start waiting/staring before moving again
        is_waiting = True
        wait_start_time = time.time()
        return current_pan, current_tilt

    # 3. MOVE: Take a step towards the target
    # Predator moves at random speeds!
    # Sometimes fast (0.8), sometimes stealthy slow (0.2)
    step_size = random.uniform(0.3, 1.2) 

    # Logic to move Pan towards Target
    if current_pan < target_pan:
        current_pan += step_size
    else:
        current_pan -= step_size

    # Logic to move Tilt towards Target
    if current_tilt < target_tilt:
        current_tilt += step_size
    else:
        current_tilt -= step_size

    # Execute the move
    move_servo(servo1, current_pan)
    move_servo(servo2, current_tilt)
    
    # Tiny sleep to keep movement smooth-ish but jerky enough to look robotic/alive
    time.sleep(0.01)

    return current_pan, current_tilt


def main():
    
    debug = True # Set to True to see the video window

    cap, cascade = find_and_open()
    if cap is None:
        return

    current_pan = 0.0
    current_tilt = 0.0

    return_center(servo1)
    return_center(servo2)

    DEAD_ZONE = 50
    is_tracking = False

    try:
        while True:
            response = camera(cap, cascade, debug)

            if response is not None:
                # --- FACE FOUND: Tracking Logic ---
                is_tracking = True
                
                diff_x, diff_y = response

                if abs(diff_x) < DEAD_ZONE: diff_x = 0
                if abs(diff_y) < DEAD_ZONE: diff_y = 0
                
                # Convert pixel difference to angle steps (e.g., 5 degrees)
                angle_step_x = convert_angle(servo1, diff_x)
                angle_step_y = convert_angle(servo2, -diff_y) # Y inverted for camera movement

                if angle_step_x != 0 or angle_step_y != 0:
                    
                    # --- FIX #2: Use P-Control (Proportional) ---
                    # We add the calculated *step* (which is proportional to the error)
                    current_pan += angle_step_x
                    current_tilt += angle_step_y
                    
                    # Clamp values to safe servo limits (-90 to 90)
                    current_pan = max(-90, min(90, current_pan))
                    # --- FIX #2: Changed min(0) to min(90) so camera can look UP ---
                    current_tilt = max(-90, min(90, current_tilt)) 
                
                    move_servo(servo1, current_pan)
                    move_servo(servo2, current_tilt)
            
            else:
                # --- NO FACE FOUND: Predator Search ---
                if is_tracking:
                    time.sleep(0.5)
                    is_tracking = False 
                
                # Use the new Predator logic
                current_pan, current_tilt = handle_predator(current_pan, current_tilt)
                
            
            # Check for quit command
            if debug and (cv2.waitKey(1) & 0xFF == ord('q')):
                break
    
    except KeyboardInterrupt:
        pass
    finally:
        return_center(servo1)
        return_center(servo2)
        end_camera(cap)

if __name__ == "__main__":
    main()
