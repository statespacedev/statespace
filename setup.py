import setuptools, os

version = None
if os.environ.get('CI_COMMIT_TAG'):
    version = os.environ['CI_COMMIT_TAG']
elif os.environ.get('CI_JOB_ID'):
    version = os.environ['CI_JOB_ID']

with open('requirements.txt') as fin:
    required = fin.read().splitlines()

setuptools.setup(
    name='statespace',
    version=version,
    author='noah smith',
    author_email='noahhsmith@gmail.com',
    description='state-space distributions and decisions',
    url='https://gitlab.com/noahhsmith/statespace',
    packages=setuptools.find_packages(),
    entry_points={'console_scripts': ['statespace = statespace.__main__:main']},
    install_requires=required,
)
