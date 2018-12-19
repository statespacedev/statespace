import setuptools, os

version = None
if os.environ.get('CI_COMMIT_TAG'):
    version = os.environ['CI_COMMIT_TAG']
elif os.environ.get('CI_JOB_ID'):
    version = os.environ['CI_JOB_ID']
if version:
    with open('version', 'wt') as fout:
        fout.write(version)
if os.path.exists('version'):
    with open('version', 'rt') as fin:
        for line in fin:
            version = line.strip()

with open('requirements.txt') as fin:
    required = fin.read().splitlines()
with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='statespace',
    version=version,
    author='noah smith',
    author_email='noahhsmith@gmail.com',
    description='state-space distributions and decisions',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.com/noahhsmith/statespace',
    packages=setuptools.find_packages(),
    entry_points={'console_scripts': ['statespace = statespace.__main__:main']},
    install_requires=required,
)
