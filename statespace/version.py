import subprocess
import os
import git

__version__ = '1.5.1'
name = 'statespace'
image = 'noahhsmith/' + name + ':' + __version__
dist = 'dist/' + name + '-' + __version__ + '.tar.gz'
pathdockerpwd = '../../.dockerpwd'


def main():
    repo = git.Repo('.', search_parent_directories=True)
    os.chdir(repo.working_tree_dir)
    # dockerhub()
    # pypi()


def runner(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def dockerhub():
    res1 = runner(['docker', 'build', '-t', image, '.'])
    with open(pathdockerpwd, 'rt') as fh: pwd = fh.read().strip()
    res2 = runner(['docker', 'login', '-u', 'noahhsmith', '-p', pwd])
    res3 = runner(['docker', 'push', image])
    return


def pypi():
    res1 = runner(['python3', 'setup.py', 'build_ext'])
    res2 = runner(['python3', 'setup.py', 'build_py'])
    res3 = runner(['python3', 'setup.py', 'sdist'])
    res4 = runner(['python3', '-m', 'twine', 'upload', '--repository', 'pypi', dist])
    return


if __name__ == '__main__':
    main()
