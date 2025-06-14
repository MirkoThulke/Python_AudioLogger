import json
import sys
import subprocess

def check_python_version():

    version_info = sys.version_info
    
    print(f"Detected Python version: {version_info.major}.{version_info.minor}.{version_info.micro}")
    
    return True


def save_outdated_packages_to_json(output_file='outdated_packages.json'):
    try:
        result = subprocess.run(
            ["pip", "list", "--outdated", "--format=json"],
            capture_output=True,
            text=True,
            check=True
        )

        packages = json.loads(result.stdout)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(packages, f, indent=4)

        print(f"Saved {len(packages)} outdated package(s) to {output_file}")
        return packages

    except subprocess.CalledProcessError as e:
        print("Failed to get outdated pip packages.", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return []
    except json.JSONDecodeError:
        print("Could not parse pip output as JSON.", file=sys.stderr)
        return []
        
        
if __name__ == "__main__":
	check_python_version()
    save_outdated_packages_to_json()