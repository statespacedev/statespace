from setuptools import setup, find_packages
from statespace.version import __version__


def main():
    with open('requirements.txt') as fin: required = fin.read().splitlines()
    with open('README.md', 'r') as fh: long_description = fh.read()
    # noinspection PyTypeChecker
    setup(
        name='statespace',
        version=__version__,
        author='noah smith',
        author_email='noahhsmith@gmail.com',
        description='statespace processors and models',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://gitlab.com/noahhsmith/statespace',
        packages=find_packages(),
        entry_points={'console_scripts': ['statespace = statespace.__main__:main']},
        install_requires=required,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent", ],
    )


if __name__ == '__main__':
    main()
