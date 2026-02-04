import sys
import os

class Tee:
    """
    Allows simultaneous writing to the console and to a log file by overwriting sys.stdout
    This is used to keep console output for future investigation.
    """
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def setup_logging(user_date):
    """
    Resets the log file to avoid adding the same result multiple times.
    Settings up the Tee class to write to both console and log files.
    """
    log_path = f"logs/{user_date}/recommendations.txt"

    # Ensure logs directory exists
    os.makedirs(f'logs/{user_date}', exist_ok=True)

    # Remove existing file if it exists
    if os.path.exists(log_path):
        os.remove(log_path)
        
    # Create new empty file
    open(log_path, 'w').close()

    # Open in append mode
    log_file_handle = open(log_path, 'a')
    sys.stdout = Tee(sys.stdout, log_file_handle)
    return log_file_handle