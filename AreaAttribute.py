import arcpy 

folder = "C:\Application\Crop-Yield-Prediction\Data\Province LandCover Shape (2000 - 2020)"

provinces = ['Ha_Noi', 'Hai_Duong', 'Hai_Phong', 'Hung_Yen', 'Nam_Dinh', 'Ninh_Binh', 'Quang_Ninh', 'Thai_Binh', 'Vinh_Phuc', 'Bac_Ninh', 'Ha_Nam']

for province in provinces:
    for year in range (2000, 2021):
        file = folder + "\\" + province + "\\" + str(year) + ".shp"
        arcpy.AddField_management(file, "area", "DOUBLE")
        with arcpy.da.UpdateCursor(file, ["SHAPE@AREA", "area"]) as cursor:
            for row in cursor:
                row[1] = row[0]
                cursor.updateRow(row)