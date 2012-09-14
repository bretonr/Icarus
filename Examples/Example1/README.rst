=========
Example 1
=========

This is a very simple example of how to use Icarus with photometric light curve.


Description of files
===========================
atmo_models.txt:
    The file providing the atmosphere model files descriptors. Each line corresponds to one band. (See Photometry class docstring for more info)
    
    - Column0 = band name
    - Column 1 = central wavelength in microns
    - Column 2 = bandpass in microns
    - Column 3: zeropoint flux in erg cm-2 s-1 Hz-1
    - Column 4 = A_band/A_J
    - Column 5 = band data filename

data.txt:
    Each line corresponds to one band. (See Photometry class docstring for more info)
    
    - Column 0 = band name
    - Column 1 = orbital phase column number
    - Column 2 = flux/mag column number
    - Column 3 = flux/mag error column number
    - Column 4 = Phase offset to add to the phases of the data (useful if the data use a different convention than the one used in Icarus, which is 0. for companion at inferior conjunction and 0.5 at superior conjunction)
    - Column 5 = Uncertainty on the band calibration
    - Column 6 = data filename

example1.py:
    The example script itself.

generate_data_example1.py:
    A script to generate mock data used by the example. In real life, one would replace mock_g.txt and mock_i.txt by real data files.

mock_g.txt:
    File containing the g-band data. Should be at least three columns.
    
    - Column 0 = orbital phase
    - Column 1 = flux/mag
    - Column 2 = flux/mag uncertainties

mock_i.txt:
    File containing the i-band data. Should be at least three columns.
    
    - Column 0 = orbital phase
    - Column 1 = flux/mag
    - Column 2 = flux/mag uncertainties

photometric_bands:
    Directory containing atmosphere data files like the ones available from http://phoenix.ens-lyon.fr/simulator.


How to
===========================
1. You should generate mock data using:

>>> ipython generate_data_example1.py

This will update the mock_g.txt and mock_i.txt files.

2. Run the example script:

>>> ipython --pylab example1.py

This should run some basic data fitting and generate a plot.


