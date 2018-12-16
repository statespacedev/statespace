import setuptools, os

if os.environ.get('CI_COMMIT_TAG'):
    version = os.environ['CI_COMMIT_TAG']
elif os.environ.get('CI_JOB_ID'):
    version = os.environ['CI_JOB_ID']
else:
    version = 'latest'

setuptools.setup(
    name='statespace',
    version=version,
    author='noah smith',
    author_email='noahhsmith@gmail.com',
    description='state-space distributions and decisions',
    url='https://gitlab.com/noahhsmith/statespace',
    packages=setuptools.find_packages(),
    entry_points={'console_scripts': ['statespace = statespace.__main__:main']},
    zip_safe=False,
)