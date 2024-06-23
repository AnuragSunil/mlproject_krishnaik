import logging
import os
from datetime import datetime

# Generate log file name and directory name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_file_name = os.path.splitext(LOG_FILE)[0]  # Remove the .log extension for directory name
logs_dir = os.path.join(os.getcwd(), "logs", log_file_name)
os.makedirs(logs_dir, exist_ok=True)  # Create the directory

# Path for the log file within the new directory
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("Logging has started")
