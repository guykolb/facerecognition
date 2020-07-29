
import xlrd
from xlutils.copy import copy

def FindImageURLColumNum(sheet,value):
    for col in range(sheet.ncols):
        if sheet.cell_value(0, col) == value:
            return col

def ExcelReadImageUrls(XlsURI):
    wb = xlrd.open_workbook(XlsURI)
    sheet = wb.sheet_by_index(0)
    sheet.cell_value(0, 0)
    return sheet


# Create a copy of the original excel and return the spreadsheet to write on
def CreateExcelToWrite(original_excel_URI):
            rb = xlrd.open_workbook(original_excel_URI)
            r_sheet = rb.sheet_by_index(0)
            wb = copy(rb)
            w_sheet = wb.get_sheet(0)
            return wb

def ExcelWriteToSheet(wb,row, col, value, confidence, output):
        w_sheet = wb.get_sheet(0)
        w_sheet.write(row,col,value)
        w_sheet.write(row, col+1, float(confidence))
        wb.save(output)
