import arcpy 

folders = ["C:\Application\Crop-Yield-Prediction\Data\Province LandCover Shape (2000 - 2020)\Ha_Noi", "C:\Application\Crop-Yield-Prediction\Data\Province LandCover Shape (2000 - 2020)\Hai_Duong", "C:\Application\Crop-Yield-Prediction\Data\Province LandCover Shape (2000 - 2020)\Hai_Phong", "C:\Application\Crop-Yield-Prediction\Data\Province LandCover Shape (2000 - 2020)\Hung_Yen", "C:\Application\Crop-Yield-Prediction\Data\Province LandCover Shape (2000 - 2020)\Nam_Dinh", "C:\Application\Crop-Yield-Prediction\Data\Province LandCover Shape (2000 - 2020)\Ninh_Binh", "C:\Application\Crop-Yield-Prediction\Data\Province LandCover Shape (2000 - 2020)\Quang_Ninh", "C:\Application\Crop-Yield-Prediction\Data\Province LandCover Shape (2000 - 2020)\Thai_Binh", "C:\Application\Crop-Yield-Prediction\Data\Province LandCover Shape (2000 - 2020)\Vinh_Phuc", "C:\Application\Crop-Yield-Prediction\Data\Province LandCover Shape (2000 - 2020)\Bac_Ninh", "C:\Application\Crop-Yield-Prediction\Data\Province LandCover Shape (2000 - 2020)\Ha_Nam"]

for province in folders:
    for year in range (2000, 2021):
        file = province + "\\" + str(year) + ".shp"
        arcpy.AddField_management(file, "area", "DOUBLE")
        with arcpy.da.UpdateCursor(file, ["SHAPE@AREA", "area"]) as cursor:
            for row in cursor:
                row[1] = row[0]
                cursor.updateRow(row)