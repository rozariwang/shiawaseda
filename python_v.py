import subprocess

# Run the 'python -V' command
version = subprocess.run(["python", "-V"], capture_output=True, text=True)

# Output the result
print(version.stdout)
