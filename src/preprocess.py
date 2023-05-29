import os
import json
import argparse

from utils import DataTool, Tag

parser = argparse.ArgumentParser(description="")
parser.add_argument("--raw_data_file", help=".xls raw data file")
parser.add_argument("--save_dir", help="save directory")
parser.add_argument("--tag_method", help="[ bio_13 | ]")
args = parser.parse_args()

assert args.raw_data_file[-4:] == ".xls"


# prepare data
data_header = ["用户地址", "POI", "楼", "单元", "层", "房间"]
train_data, test_data = DataTool(args.raw_data_file, data_header, [0.5, 0.5]).split


# prepare tagging method
t = Tag(args.tag_method)
generate_tags = t.generate_func
label2id = t.label2id


# save training/testing data as json file
def convert(record):
    tokens = list(record[0])
    tags = generate_tags(record)
    if tags:
        # tag_ids = [label2id[tag] for tag in tags]
        return {"raw": record, "tokens": tokens, "ner_tags": tags}
    return None


if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

with open(os.path.join(args.save_dir, "train.json"), "w") as train_file:
    train_json = [convert(record) for record in train_data if convert(record)]
    train_file.write(json.dumps(train_json, indent=4))

with open(os.path.join(args.save_dir, "test.json"), "w") as test_file:
    test_json = [convert(record) for record in test_data if convert(record)]
    test_file.write(json.dumps(test_json, indent=4))


# save features
with open(os.path.join(args.save_dir, "features.json"), "w") as features_file:
    features = {
        "ner_tags": list(label2id.keys()),
    }
    features_file.write(json.dumps(features, indent=4))
