from setuptools import setup

'''
Following method given in this StackOverflow page:
https://stackoverflow.com/questions/6323860/sibling-package-imports/50193944#50193944
'''

def parse_requirements(filename):
	lines = (line.strip() for line in open(filename))
	return [line for line in lines if line and not line.startswith("#")]

setup(name='mstat',
        version='0.1',
        description='Python tool for analysis & classification of mass spectrometer data.',
        url='https://github.com/PSI-Github-lab/mstat-repo',
        author='Jackson Russett',
        author_email='jrussett(at)pointsurgical.com',
        license='',
        packages=['mstat'],
        #install_requires=parse_requirements('requirements.txt'),
        zip_safe=True)