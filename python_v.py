import subprocess

try:
    # Run the 'python -V' command with a timeout
    version = subprocess.run(["python", "-V"], capture_output=True, text=True, timeout=10)  # 10-second timeout
    print(version.stdout)
except subprocess.TimeoutExpired:
    print("The command took too long to execute.")
