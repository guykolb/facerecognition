import ExcelUtliz
import WebDownload
from datetime import datetime
from gender_classification import GenderClassifier
import os


no_image_url = "https://static-exp1.licdn.com/sc/h/7st6vm28lv5le4b37jrjaw2n3"
data_uri = "Benchmark.xlsx"
now = datetime.now()
output_uri = "output" + str(datetime.timestamp(now))+".xls"
crop_face = True

genderClassifier = GenderClassifier()


#data_uri = input ("Insert the file name: ")
#ImageURL_col_number =  input ("Insert the profile imageURL column number : ")
#gender_column_number = input ("Insert the column number you want to insert results: ")

# Return all the rows from the Excel
w_excel = ExcelUtliz.CreateExcelToWrite(data_uri)
data_spreadsheet = ExcelUtliz.ExcelReadImageUrls(data_uri)
image_url_column_number = ExcelUtliz.FindImageURLColumNum(data_spreadsheet,'ImageURL')
gender_column_number = data_spreadsheet.ncols

# Run on each row
for i in range(1, data_spreadsheet.nrows):
    try:
        # Check that profile has a photo
        if no_image_url != str(data_spreadsheet.cell_value(i, int(image_url_column_number))):
            print(data_spreadsheet.cell_value(i, int(image_url_column_number)))
            # Download photo
            WebDownload.DownloadPhoto(data_spreadsheet.cell_value(i, int(image_url_column_number)), "Photos\Photo" + str(i))
            # Send the image URL, return back gender

            gender, confidence = genderClassifier.classify_image("Photos\Photo" +  str(i) + ".jpg", return_confidence=True,crop_face=crop_face)
            os.remove("Photos\Photo" +  str(i) + ".jpg")

            # My original code # gender = gad.face_reco_gender_image(ImageURLlist.cell_value(i, int(ImageURL_col_number)))
            ExcelUtliz.ExcelWriteToSheet(w_excel,i,gender_column_number,gender,confidence,output_uri)
        else:
            ExcelUtliz.ExcelWriteToSheet(w_excel,i,gender_column_number,"No photo",1,output_uri)
    except Exception as e:
            print("guy")

input("Completed")
