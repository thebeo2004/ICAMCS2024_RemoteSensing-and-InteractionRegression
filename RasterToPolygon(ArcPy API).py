import arcpy
inputs = ["C:\Users\CA_UOP_MUOI\OneDrive - vnu.edu.vn\Dataset (Crop Yield Prediction)\Practice\Province LandCover (2000 - 2020)\Ha Noi", "C:\Users\CA_UOP_MUOI\OneDrive - vnu.edu.vn\Dataset (Crop Yield Prediction)\Practice\Province LandCover (2000 - 2020)\Hai Duong", "C:\Users\CA_UOP_MUOI\OneDrive - vnu.edu.vn\Dataset (Crop Yield Prediction)\Practice\Province LandCover (2000 - 2020)\Hai Phong", "C:\Users\CA_UOP_MUOI\OneDrive - vnu.edu.vn\Dataset (Crop Yield Prediction)\Practice\Province LandCover (2000 - 2020)\Hung Yen", "C:\Users\CA_UOP_MUOI\OneDrive - vnu.edu.vn\Dataset (Crop Yield Prediction)\Practice\Province LandCover (2000 - 2020)\Nam Dinh", "C:\Users\CA_UOP_MUOI\OneDrive - vnu.edu.vn\Dataset (Crop Yield Prediction)\Practice\Province LandCover (2000 - 2020)\Ninh Binh", "C:\Users\CA_UOP_MUOI\OneDrive - vnu.edu.vn\Dataset (Crop Yield Prediction)\Practice\Province LandCover (2000 - 2020)\Quang Ninh", "C:\Users\CA_UOP_MUOI\OneDrive - vnu.edu.vn\Dataset (Crop Yield Prediction)\Practice\Province LandCover (2000 - 2020)\Thai Binh", "C:\Users\CA_UOP_MUOI\OneDrive - vnu.edu.vn\Dataset (Crop Yield Prediction)\Practice\Province LandCover (2000 - 2020)\Vinh Phuc"]
outputs = ["C:\Users\CA_UOP_MUOI\OneDrive - vnu.edu.vn\Dataset (Crop Yield Prediction)\Practice\Province LandCover Shape (2000 - 2020)\Ha Noi", "C:\Users\CA_UOP_MUOI\OneDrive - vnu.edu.vn\Dataset (Crop Yield Prediction)\Practice\Province LandCover Shape (2000 - 2020)\Hai Duong", "C:\Users\CA_UOP_MUOI\OneDrive - vnu.edu.vn\Dataset (Crop Yield Prediction)\Practice\Province LandCover Shape (2000 - 2020)\Hai Phong", "C:\Users\CA_UOP_MUOI\OneDrive - vnu.edu.vn\Dataset (Crop Yield Prediction)\Practice\Province LandCover Shape (2000 - 2020)\Hung Yen", "C:\Users\CA_UOP_MUOI\OneDrive - vnu.edu.vn\Dataset (Crop Yield Prediction)\Practice\Province LandCover Shape (2000 - 2020)\Nam Dinh", "C:\Users\CA_UOP_MUOI\OneDrive - vnu.edu.vn\Dataset (Crop Yield Prediction)\Practice\Province LandCover Shape (2000 - 2020)\Ninh Binh", "C:\Users\CA_UOP_MUOI\OneDrive - vnu.edu.vn\Dataset (Crop Yield Prediction)\Practice\Province LandCover Shape (2000 - 2020)\Quang Ninh", "C:\Users\CA_UOP_MUOI\OneDrive - vnu.edu.vn\Dataset (Crop Yield Prediction)\Practice\Province LandCover Shape (2000 - 2020)\Thai Binh", "C:\Users\CA_UOP_MUOI\OneDrive - vnu.edu.vn\Dataset (Crop Yield Prediction)\Practice\Province LandCover Shape (2000 - 2020)\Vinh Phuc"]
length = len(inputs)

for i in range(length):
    arcpy.env.workspace = inputs[i]
    dataset = arcpy.ListRasters()
    folder = outputs[i] + "\\"
    year = 2000
    for raster in dataset:
        raster_out = folder + str(year)
        year += 1
        arcpy.conversion.RasterToPolygon(raster, raster_out, "NO_SIMPLIFY", "VALUE")
