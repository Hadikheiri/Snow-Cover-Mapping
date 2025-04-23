import subprocess

def run_command(command):
    """Runs a shell command and returns the output."""
    try:
        result = subprocess.run(command, check=True, text=True, shell=True)
        print(f"✓ Command succeeded: {command}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed: {command}\n{e}")

# Commands to push the project
commands = [
    "git init",
    "git remote remove origin",  # In case already added
    "git remote add origin https://github.com/Hadikheiri/Snow-Cover-Mapping.git",
    "git add .",
    "git commit -m \"Automated push from script\"",
    "git branch -M main",
    "git push -u origin main"
]

if __name__ == "__main__":
    for cmd in commands:
        run_command(cmd)
