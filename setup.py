from setuptools import setup, find_packages

setup(
    name='deep4downscaling',
    version='2025.11.03',
    description='A package for deep downscaling scripts.',
    author='nobody',
    author_email='noreply@example.com',

    packages=find_packages(), 
    
    install_requires=[
        'torch>=2.0', 'xskillscore','bottleneck','xarray','pandas','scipy','matplotlib','cartopy','ipykernel'
    ],
    python_requires='>=3.9',
)