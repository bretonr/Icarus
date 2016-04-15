from distutils.core import setup
from setuptools import find_packages


setup(
    name='Icarus',

    version='2.3.1',

    # description
    description='Icarus is a stellar binary light curve synthesis tool initially developed by Rene Breton',
    long_description=open('README.md').read(),

    # The project's main homepage.
    url='https://github.com/bretonr/Icarus',

    # The project's download url.
    download_url='https://github.com/bretonr/Icarus/tarball/v2.3.0',

    # Author details
    author='Dr Rene Breton',
    author_email='superluminique@gmail.com',

    # license
    license='BSD',

    classifiers=[

        'Operating System :: OS Independent',
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Science/Research',

        # relevant topics
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Astronomy',

        # license
        'License :: OSI Approved :: BSD License',

        # python versions this library supports
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    keywords=['astrophysics','cosmology', 'photometry', 'binary', 'modeling', 'space', 'models', 'spectroscopy', 'astronomy', 'science', 'research', 'stars', 'physics'],

    # includes everything except the examples
    packages=find_packages(exclude=['Examples']),

    # as stated on https://github.com/bretonr/Icarus 
    install_requires=['numpy', 'scipy', 'astropy'],


    # including the geodesic data files.
    include_package_data = True,
    package_data={
        '': ['*.txt'],
    }
)

# recommended libraries
try:
    import matplotlib
except:
    print 'matlibplot is not installed. Although not a requirement but in order to get better graphs please install it'

try: 
    import PyGTS
except:
    print 'PyGTS is not installed. Although not a requirement but in order to generate surface geodesic primitives instead of reading the pre-generated one, and calculate occulations and transits in eclipsing binaries please install it'
