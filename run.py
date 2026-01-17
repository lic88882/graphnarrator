import argparse
import os
from pathlib import Path


def save_configuration(sbatch_script_path, job_name, output_file_path, num_gpus, command):
    content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_file_path}
#SBATCH --gres=gpu:{num_gpus}

{command}"""

    with open(sbatch_script_path, 'w', encoding="utf-8") as file:
        file.write(content)


def execute_scripts(sbatch_script_path):
    command = f"sbatch {sbatch_script_path}"
    print(command)
    os.system(command)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--job_name", type=str, default="untitled_job")
    parser.add_argument("-d", "--dir", type=str, default=".")
    parser.add_argument("-n", "--num_gpus", type=int, default=1)
    parser.add_argument("-c", "--command", type=str, required=True)
    args = parser.parse_args()

    return args


def main():
    args = parse()
    output_dir = Path(args.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sbatch_script_path = output_dir / "sbatch.sh"
    output_file_path = output_dir / "output.txt"

    save_configuration(sbatch_script_path, args.job_name, output_file_path, args.num_gpus, args.command)
    execute_scripts(sbatch_script_path)


if __name__ == "__main__":
    main()
