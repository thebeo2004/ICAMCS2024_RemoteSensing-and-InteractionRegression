from osgeo import gdal, ogr


src_ds = gdal.Open("Raster File Path", gdal.GA_ReadOnly)

print(src_ds.RasterCount)
srcband = src_ds.GetRasterBand(1)

#  create output datasource

dst_layername = "POLYGONIZED_STUFF"
drv = ogr.GetDriverByName("ESRI Shapefile")
dst_ds = drv.CreateDataSource( dst_layername + ".shp" )
dst_layer = dst_ds.CreateLayer(dst_layername, srs = None )

gdal.Polygonize( srcband, None, dst_layer, -1, [], callback=None )