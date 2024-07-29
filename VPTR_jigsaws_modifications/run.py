import subprocess
import sys

def run_script(script_name):
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}: {result.stderr}")
    else:
        print(f"Successfully ran {script_name}: {result.stdout}")

if __name__ == "__main__":
    run_script('train_AutoEncoder.py')
    run_script('train_FAR.py')