import time
import os
from datetime import datetime, timedelta
from picamera2 import Picamera2

# Set up the directory to save images
photos_dir = "/home/emman/photos"
os.makedirs(photos_dir, exist_ok=True)

# Initialize and start the camera
picam = Picamera2()
picam.configure(picam.create_still_configuration())
picam.start()

# Define capture interval during the window (in seconds)
capture_interval = 300  # 5 minutes

def wait_until(target_time):
    """Sleep until the specified datetime object target_time."""
    now = datetime.now()
    seconds_to_wait = (target_time - now).total_seconds()
    if seconds_to_wait > 0:
        print(f"Waiting {int(seconds_to_wait)} seconds until {target_time.strftime('%H:%M:%S')}")
        time.sleep(seconds_to_wait)

print("Starting 24-hour capture script. Press Ctrl+C to stop.")

try:
    while True:
        now = datetime.now()
        if 6 <= now.hour < 22:
            # In capture window: Capture an image
            filename = os.path.join(photos_dir, "image_" + now.strftime("%Y%m%d_%H%M%S") + ".jpg")
            picam.capture_file(filename)
            print(f"{datetime.now().strftime('%H:%M:%S')}: Captured {filename}")
            time.sleep(capture_interval)
        else:
            # Outside capture window: Wait until the next 6 AM occurs
            if now.hour >= 22:
                # After 10 PM: wait until 6 AM next day
                next_capture = datetime(now.year, now.month, now.day, 6, 0, 0) + timedelta(days=1)
            else:
                # Before 6 AM: wait until 6 AM today
                next_capture = datetime(now.year, now.month, now.day, 6, 0, 0)
            print(f"{datetime.now().strftime('%H:%M:%S')}: Outside capture window. Sleeping until {next_capture.strftime('%Y-%m-%d %H:%M:%S')}")
            wait_until(next_capture)
except KeyboardInterrupt:
    print("Capture interrupted by user.")
finally:
    picam.stop()
    print("Camera stopped.")