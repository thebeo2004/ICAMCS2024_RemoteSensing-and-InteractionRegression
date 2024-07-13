import arcpy   
input_folder = "C:\Application\Crop-Yield-Prediction\Data\Province LandCover (2000 - 2020)"
output_folder = "C:\Application\Crop-Yield-Prediction\Data\Province LandCover Shape (2000 - 2020)"

provinces = ['Ha_Noi', 'Hai_Duong', 'Hai_Phong', 'Hung_Yen', 'Nam_Dinh', 'Ninh_Binh', 'Quang_Ninh', 'Thai_Binh', 'Vinh_Phuc', 'Bac_Ninh', 'Ha_Nam']

for province in provinces:
    arcpy.env.workspace = input_folder + "\\" + province 
    dataset = arcpy.ListRasters()
    year = 2000
    for raster in dataset:
        output_file = output_folder + "\\" + province + "\\" + str(year)
        year += 1
        arcpy.conversion.RasterToPolygon(raster, output_file, "NO_SIMPLIFY", "VALUE")