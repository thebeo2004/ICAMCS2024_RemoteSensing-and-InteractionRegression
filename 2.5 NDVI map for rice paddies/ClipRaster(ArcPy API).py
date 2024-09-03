import arcpy

raster_fin = r"C:\Application\Crop-Yield-Prediction\Data\NDVI of DHMT&DBSCL"
vector_fin = r"C:\Application\Crop-Yield-Prediction\Data\Province Rice Paddies (2000 - 2020)"
f_out = r"C:\Application\Crop-Yield-Prediction\Data\NDVI of Rice Paddies"

provinces = ['Long_An', 'Tien_Giang', 'Ben_Tre', 'Tra_Vinh', 'Dong_Thap', 'An_Giang', 'Kien_Giang', 'Can_Tho', 'Soc_Trang', 'Bac_Lieu', 'Ca_Mau']

for province in provinces:
    for year in range(2000, 2021):
        for month in range(1, 13):

            if (year == 2000 and month == 1):
                continue

            str_year = str(year)
            str_month = str(month)

            if (month < 10):
                str_month = '0'+ str_month

            raster_fname = raster_fin + "\\" + "NDVI_" + str_year + "\\" + str_month + ".tif"
            vector_fname = vector_fin + "\\" + province + "\\" + str_year + ".shp"
            storing_fname = f_out + "\\" + province + "\\" + str_year + "\\" + str_month + ".tif"
            
            arcpy.Clip_management(raster_fname, "#", storing_fname, vector_fname, "#", "ClippingGeometry")             
 