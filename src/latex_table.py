import os
import json

names = {
    "bilstm": "BiLSTM",
    "roberta": "RoBERTa",
    "roberta_crf": "RoBERTa+CRF",
    "roberta_others": "RoBERTa (Trainer)",
    "bert": "BERT",
    "bert_crf": "BERT+CRF",
    "bert-wwm": "BERT-wwm",
    "bert-wwm_crf": "BERT-wwm+CRF",
}

parameters = {
    "bilstm": "147 K",
    "roberta": "101,687 K",
    "roberta_crf": "102,278 K",
    "roberta_others": "101,687 K",
    "bert": "101,687 K",
    "bert_crf": "102,278 K",
    "bert-wwm": "101,687 K",
    "bert-wwm_crf": "102,278 K",
}

epochs = {
    "bilstm": 300,
    "roberta": 10,
    "roberta_crf": 10,
    "roberta_others": 3,
    "bert": 10,
    "bert_crf": 10,
    "bert-wwm": 10,
    "bert-wwm_crf": 10,
}

path = "../tmp"
model_names = os.listdir(path)


all_results = dict()
for model_name in model_names:
    model_path = os.path.join(path, model_name)
    if not os.path.isdir(model_path):
        continue
    
    result_file = os.path.join(model_path, "all_results.json")
    if not os.path.exists(result_file):
        continue

    with open(result_file) as f:
        result_dict = json.loads(f.read())

    if "epoch" in result_dict:
        result_dict = {key[5:]: result_dict[key] for key in result_dict if key[:5] == "test_"}
    
    # lowercase & round
    result_dict = {key.lower(): round(result_dict[key], 3) for key in result_dict}

    all_results[model_name] = result_dict


entities = ["POI", "Building", "Unit", "Floor", "Room", "Floor|Room"]

# token level
for entity in entities:
    print("-"*10, "Token Level:", entity, "-"*10)
    lines = []

    keyword = f"token_{entity.lower()}"
    max_acc, max_prec, max_recall, max_f1 = 0, 0, 0, 0
    for model_name in all_results:
        result_dict = all_results[model_name]
        result_dict = {key: result_dict[key] for key in result_dict if keyword in key}
        
        accuracy = result_dict[f"{keyword}_accuracy"]
        precision = result_dict[f"{keyword}_precision"]
        recall = result_dict[f"{keyword}_recall"]
        f1 = result_dict[f"{keyword}_f1"]

        max_acc = max(max_acc, accuracy)
        max_prec = max(max_prec, precision)
        max_recall = max(max_recall, recall)
        max_f1 = max(max_f1, f1)

        lines.append([names[model_name], accuracy, precision, recall, f1])
    
    for line in lines:
        line = [str(item) for item in line]
        if line[1] == str(max_acc):
            line[1] = "\\textbf{" + line[1] + "}"
        if line[2] == str(max_prec):
            line[2] = "\\textbf{" + line[2] + "}"
        if line[3] == str(max_recall):
            line[3] = "\\textbf{" + line[3] + "}"
        if line[4] == str(max_f1):
            line[4] = "\\textbf{" + line[4] + "}"
        print(" & ".join(line), "\\\\")
        print("\\hline")


# entity level
print("-"*10, "Entity Level:", "-"*10)
max_accs = [0 for _ in entities]
lines = []
for model_name in all_results:
    result_dict = all_results[model_name]

    accuracys = [result_dict[f"entity_{entity.lower()}_accuracy"] for entity in entities]
    for idx, acc in enumerate(accuracys):
        max_accs[idx] = max(max_accs[idx], acc)

    lines.append([names[model_name]] + accuracys)

for line in lines:
    line = [str(item) for item in line]
    for idx, acc in enumerate(line[1:]):
        if acc == str(max_accs[idx]):
            line[idx+1] = "\\textbf{" + line[idx+1] + "}"
    print(" & ".join(line), "\\\\")
    print("\\hline")


# sentence level
print("-"*10, "Sentence Level:", "-"*10)
max_acc = 0
lines = []
for model_name in all_results:
    result_dict = all_results[model_name]
    accuracy = result_dict[f"sentence_accuracy"]
    max_acc = max(max_acc, accuracy)
    lines.append([names[model_name], accuracy, parameters.get(model_name), epochs.get(model_name)])

for line in lines:
    line = [str(item) for item in line]
    
    if line[1] == str(max_acc):
        line[1] = "\\textbf{" + line[1] + "}"
    print(" & ".join(line), "\\\\")
    print("\\hline")