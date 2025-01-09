from setuptools import setup, find_packages

setup(
    name='alpacka',
    description='Alpacka',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'gin-config',
        'gym',
        # TODO(xxx): Move to extras?
        # (need to lazily define alpacka.envs.Sokoban then)
        'gym_sokoban @ git+ssh://git@gitlab.com/awarelab/gym-sokoban.git',
        'neptune-client',
        'numpy',
        'randomdict',
        'ray==2.8.1',
        'tensorflow>=2.2.0',
    ],
    extras_require={
        'dev': ['pylint==2.4.4', 'pylint_quotes', 'pytest', 'ray[debug]==0.8.5'],
        'tracex': ['flask', 'opencv-python', 'Pillow'],
    }
)
