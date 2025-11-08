from setuptools import setup, find_packages

setup(
    name='deep4downscaling',
    version='2025.11.03',
    description='A package for deep downscaling scripts.',
    author='nobody',
    author_email='noreply@example.com',

    packages=find_packages(), 
    
    install_requires=[
        'torch==2.5.1', 'xskillscore','bottleneck','xarray','pandas','scipy','matplotlib','cartopy','ipykernel'
    ],
    python_requires='>=3.9',
)

# python -c "import deep4downscaling.viz;import deep4downscaling.trans;import deep4downscaling.deep.loss;import deep4downscaling.deep.utils;import deep4downscaling.deep.models;import deep4downscaling.deep.train;import deep4downscaling.deep.pred;import deep4downscaling.metrics;import deep4downscaling.metrics_ccs"