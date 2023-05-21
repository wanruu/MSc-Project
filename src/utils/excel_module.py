import xlrd
import xlwt


class xls_reader:
    def __init__(self, filename) -> None:
        self.sheet = xlrd.open_workbook(filename).sheet_by_index(0)

    def get_fieldnames(self) -> list:
        fieldnames = self.sheet.row_values(0)
        return fieldnames

    def to_dict(self) -> list:
        # get fieldnames
        fieldnames = self.sheet.row_values(0)

        # for each row, generate a dictionary & append to results
        results = []
        for row_idx in range(1, self.sheet.nrows):
            cells = self.sheet.row(row_idx)  # ctype & value
            cells = [str(cell.value).lower() if cell.ctype !=
                     2 else str(int(cell.value)) for cell in cells]  # convert float to str(int(_))
            results.append(
                {fieldname: cells[col_idx] for col_idx, fieldname in enumerate(fieldnames)})

        return results


class xls_writer:
    def __init__(self, filename) -> None:
        self.filename = filename

    def write_dict(self, dict_data):
        assert dict_data
        workbook = xlwt.Workbook()
        worksheet = workbook.add_sheet('Sheet1')
        fieldnames = [key for key in dict_data[0]]
        for col_idx, fieldname in enumerate(fieldnames):
            # write header
            worksheet.write(0, col_idx, fieldname)
            # write data
            for item_idx, item in enumerate(dict_data):
                worksheet.write(item_idx+1, col_idx, item[fieldname])
        workbook.save(self.filename)
