# NOTE: need to run in terminal from root dir as:
# uv run -m B_analysis.s1_create_requirements

import toml

out_fname = 'slurm_requirements.txt'

target_packages = ['torch',
                'torchvision',
                'lightning',
                'ocnn',
                'scikit-learn',
                'laspy',
                'wandb',
                'pandas']

with open('pyproject.toml', 'r') as f:
    pyproj_cfg = toml.load(f)

requirements = pyproj_cfg['project']['dependencies']

requirements = [req for req in requirements if any(pkg in req for pkg in target_packages)]

#Add cc suffix on end of each package
sfx = "+computecanada"
requirements = [req + sfx for req in requirements]

with open(out_fname, 'w') as f:
    for req in requirements:
        f.write(f'{req}\n')

print('\nRequirements:\n')
print(".\n".join(requirements))
print('\nRequirements file created at:', out_fname, "\n")