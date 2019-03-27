'''
Script to run the 'Detect objects using deep learning' on multiple rasters
or mosaic dataset using a feature layer as input
To be run directly in ArcGIS Pro
'''
import arcpy

arcpy.env.workspace = "F:/ArcGIS/Semester_Log/data.gdb" # Set workspace
fc = "Random sample squares" # Feature dataset
raster = "Composite Bands_1_2_3" # Input raster to dectet against
in_model_definition = "F:/ArcGIS/Semester_Log/esriObjectDetection.emd" # Model definition file
number = 0 # For for loop

with arcpy.da.SearchCursor(fc, "SHAPE@") as cursor:
	for row in cursor:
		outExtractByMask = arcpy.sa.ExtractByMask(raster, row) # Extract
		out_detected_objects = "detected_" + str(number)
		number += 1
		arcpy.ia.DetectObjectsUsingDeepLearning(outExtractByMask, out_detected_objects,
in_model_definition, "padding 0")