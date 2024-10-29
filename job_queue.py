#!/usr/bin/env python3
import time
import subprocess
import os
import fcntl

QUEUE_FILE = "training_queue.txt"
LOG_DIR = "job_logs"


def read_first_job():
    try:
        # Open with exclusive lock for reading and writing
        with open(QUEUE_FILE, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            lines = f.readlines()
            if not lines:
                fcntl.flock(f, fcntl.LOCK_UN)
                return None

            # Get first job and rewrite file without it
            job = lines[0].strip()
            f.seek(0)
            f.writelines(lines[1:])
            f.truncate()
            fcntl.flock(f, fcntl.LOCK_UN)
            return job
    except FileNotFoundError:
        return None


def run_job(command):
    # Create logs directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create log file with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(LOG_DIR, f"job_{timestamp}.log")

    print(f"Starting job: {command}")
    print(f"Logging to: {log_file}")

    with open(log_file, "w") as f:
        f.write(f"Command: {command}\nStarted: {timestamp}\n\n")
        f.flush()

        # Run the command and tee output to both console and log
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.rstrip())
                f.write(output)
                f.flush()

        return_code = process.poll()
        f.write(f"\nFinished: {time.strftime('%Y%m%d-%H%M%S')}")
        f.write(f"\nReturn code: {return_code}")

    return return_code


def main():
    print("Job queue started. Watching for jobs...")

    try:
        while True:
            job = read_first_job()
            if job:
                run_job(job)
            else:
                time.sleep(10)  # Wait before checking again
    except KeyboardInterrupt:
        print("Exiting...")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    main()
