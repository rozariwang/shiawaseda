import subprocess

def run_nvidia_smi():
    try:
        # Run the nvidia-smi command and capture the output
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Print the output
        print("nvidia-smi Output:\n", result.stdout)
        
        # If there's an error, print the error
        if result.stderr:
            print("Error:\n", result.stderr)
            
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function
run_nvidia_smi()
