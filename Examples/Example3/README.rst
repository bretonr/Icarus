=========
Example 3
=========

This is a simple example in which a star is created and the flux is calculated with and without Doppler boosting.


Description of files
===========================
example3.py:
    The example script itself.

photometric_bands:
    Directory containing atmosphere data files.
    It also needs to contain grid files of Doppler boosting coefficients, such that flux = flux * (1 + boost*v)


How to
===========================
1. Run the example script:

>>> ipython --pylab example3.py

This should generate two plots. The first shows the flux with and without Doppler boosting. The second shows the ratio of the Doppler boosted flux by the non-Doppler boosted one.


