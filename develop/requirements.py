import subprocess
import sys

def save_dependencies_to_file(file_name="requirements.txt"):
    # Use pip from the current virtual environment
    pip_path = sys.executable.replace('python', 'pip')
    
    # Run `pip freeze`
    result = subprocess.run([pip_path, 'freeze'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        print(f"Error running pip: {result.stderr.decode()}")
        return
    
    # Decode the output and split into lines
    packages = result.stdout.decode().splitlines()
    
    # Save to file
    with open(file_name, "w") as f:
        for package in packages:
            f.write(package + "\n")
    
    print(f"Dependencies saved to {file_name}")

# Call the function
save_dependencies_to_file()
