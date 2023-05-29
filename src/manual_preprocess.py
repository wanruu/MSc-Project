import re
from utils import xls_reader, xls_writer

# data to pre-process
data = xls_reader("../data/preprocessed/20230412.xls").to_dict()
print("size:", len(data))

# valid fieldnames
fieldnames = ["用户地址", "POI", "楼", "单元", "层", "房间"]


# convert to half-width, lowercase
# full match: 17528 -> 17533
data = [{key: item[key].lower().replace("（", "(").replace("）", ")") for key in item} for item in data]

# remove space not between digits, alphabet
# full match: 17533 -> 17549
data = [{key: re.sub(r"(?<![a-z0-9])\s(?![a-z0-9])", "", item[key]) for key in item} for item in data]


# full match
n_full_match = 0
for item in data:
    full_match = True
    for fieldname in fieldnames[1:]:
        if item[fieldname] not in item["用户地址"]:
            full_match = False
            break
    n_full_match += full_match
print("full match:", n_full_match)

# manual process
# full match 17549 -> 17777
if_end = False
for item_idx, item in enumerate(data):
    if item_idx < 1856:
        continue
    for fieldname in fieldnames[1:]:
        if item[fieldname] not in item["用户地址"]:
            print(item_idx, item["用户地址"])
            print(fieldname)
            print(item[fieldname])

            new_val = input()
            if not new_val:
                continue
            elif new_val == 'break':
                if_end = True
                break
            elif new_val == ' ':
                data[item_idx][fieldname] = ''
            else:
                data[item_idx][fieldname] = new_val
            print()
    if if_end:
        break


writer = xls_writer("../data/preprocessed/20230412.xls")
writer.write_dict(data)