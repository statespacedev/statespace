import setuptools

with open('readme.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='statespace',
    version='2018.3.1',
    author='noah smith',
    author_email='noahhsmith@gmail.com',
    description='state-space distributions and decisions',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.com/noahhsmith/statespace',
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            'statespace = statespace.__main__:main'
        ]
    },
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ),
)