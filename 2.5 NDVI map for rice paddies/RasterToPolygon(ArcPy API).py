#Running in Arcmap which intergrated with arcpy
import arcpy   
input_folder = "C:\Application\Crop-Yield-Prediction\Data\Province LandCover (2000 - 2020)"
output_folder = "C:\Application\Crop-Yield-Prediction\Data\Province LandCover Shape (2000 - 2020)"

provinces = ['Long_An', 'Tien_Giang', 'Ben_Tre', 'Tra_Vinh', 'Dong_Thap', 'An_Giang', 'Kien_Giang', 'Can_Tho', 'Soc_Trang', 'Bac_Lieu', 'Ca_Mau']

for province in provinces:
    arcpy.env.workspace = input_folder + "\\" + province 
    dataset = arcpy.ListRasters()
    year = 2000
    for raster in dataset:
        output_file = output_folder + "\\" + province + "\\" + str(year)
        year += 1
        arcpy.conversion.RasterToPolygon(raster, output_file, "NO_SIMPLIFY", "VALUE")