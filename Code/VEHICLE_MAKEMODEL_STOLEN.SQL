SELECT
VEHICLE_MODEL, VEHICLE_MAKE, COUNT(*) AS cnt 

FROM (Select * from 2013_Stolen_Vehicle a union Select * from 2012_stolen_Vehicle b union Select * from 2014_Stolen_Vehicle c) a

gROUP BY
VEHICLE_MODEL, VEHICLE_MAKE

ORDER BY CNT DESC