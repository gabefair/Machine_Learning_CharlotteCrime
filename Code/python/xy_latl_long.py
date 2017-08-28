__author__ = 'Gabriel'

import pyproj
import csv


def open(path):
    f = open(path + '\\' + 'xy_cords.csv')
    csv_f = csv.reader(f)
    for row in csv_f:
        complaint_no = row[0]
        x = row[1]
        y = row[2]
        x, y = convert(x, y)

    return 1


def save():
    return


def convert(x, y):
    # NAD 1983 StatePlane North Carolina FIPS 3200 Feet

    # Define a projection with Proj4 notation

    isn2004 = pyproj.Proj(
        "+proj=lcc +lat_1=34.33333333333334 +lat_2=36.16666666666666 +lat_0=33.75 +lon_0=-79 +x_0=609601.2199999999 +y_0=0 +ellps=GRS80 +datum=NAD83 +to_meter=0.3048006096012192 +no_defs")
    return isn2004(x, y, inverse=True)
