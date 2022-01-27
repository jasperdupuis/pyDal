# pyDal
 MASc code.
 
 # Dependencies
 This was developed in Python 3.8.8. 
 
 Michael Porter's acoustic toolbox: https://oalib-acoustics.org/ 
 Hunter Akins' PyAT*, https://github.com/org-arl/arlpy
 marcuskd's PyRAM: https://github.com/marcuskd/pyram
 Mandar Chitre's ARLPy: https://github.com/org-arl/arlpy
 
 PyPI package haversine 2.5.1
 
 All have an open license except (*), who has been emailed and granted permission for non-commercial use.
 
 PyAT was not provided with a setup.py file, and so must be directly included using sys.path.insert(1,'string_absolutepath'). stringpath must be updated in the following files:
 each environment main file
 environment.py
 comparison_setup.py
 
 
 The windows binary directory within the acoustic toolbox download must be put on your windows path (user or sys).

 

 # Use
 
Set Location using one of the strings, or roll your own in env/locations. This generally manages some fnames, bottom type, and lat/lon coordinates. Be sure DEG_OFFSET is suitable when defining a new environment.

Once the location is defined, the source and environments particular to each propagation loss model must be defined. This is taken care of in comparison_setup(). 

A deeper dive on comparison_setup() shows book keeping on setting up bathy, SSP, bottom properties, surface, before creating the ARL/PYAT/PYRAM environmental models.

Once the models are made it's straighforward to retrieve results. For BELLHOP you can only compute point to point results - it takes a while to do a large amount of frequencies or points.
