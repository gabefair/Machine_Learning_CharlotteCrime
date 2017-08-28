cd "/home/gabriel/Dropbox/Machine Learning Project/Data"

import pandas as pd

pd.set_option('display.mpl_style', 'default') 
pd.set_option('display.line_width', 5000) 
pd.set_option('display.max_columns', 60) 

# Incident File (1 for each year):

incident2012 = pd.read_excel('Incident Data 2012.xlsx')
incident2013 = pd.read_excel('Incident Data 2013.xlsx')
#for some reason, incident2014 only pulls in 76 records
incident2014 = pd.read_excel('Incident Data 2014.xlsx')

# Other Files (6 for each year):
# Victim Business, Victim Person, Victim-Suspect, Offenses, Weapon, Property