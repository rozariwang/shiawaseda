#!/bin/bash

# Run the Python command directly
python3 -c "
import subprocess
version = subprocess.run(['python', '-V'], capture_output=True, text=True)
print(version.stdout)
"