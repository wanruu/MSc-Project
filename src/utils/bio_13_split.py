# ignore ( and ) and space
# spilt by ( and ) and space
import re

# label
label2id = {
    "O": 0,
    "B-POI": 1,
    "B-building": 2,
    "B-unit": 3,
    "B-floor": 4,
    "B-room": 5,
    "I-POI": 6,
    "I-building": 7,
    "I-unit": 8,
    "I-floor": 9,
    "I-room": 10,
    "B-floor|room": 11,
    "I-floor|room": 12,
}
id2label = {label2id[label]: label for label in label2id}


def generate_tags(record: list):
    """
    record: [address, POI, building, unit, floor, room]
    """
    def _add_prefix(labels: list):
        new_labels = labels[:]
        # check each label
        for idx in range(len(labels)):
            # ignore O
            if labels[idx] == "O":
                continue
            # add B-
            if idx == 0 or labels[idx-1] != labels[idx]:
                new_labels[idx] = "B-" + new_labels[idx]
            # add I-
            elif labels[idx-1] == labels[idx]:
                new_labels[idx] = "I-" + new_labels[idx]
        return new_labels

    def _match_entity(addr, entity):
        """
        params addr: raw address, e.g., 碧湖豪苑-十八栋 (18栋502室)
        params entity: e.g. 碧湖豪苑
        return: indexes, e.g. [0,1,2,3] 
        """
        split_entity = re.split(r"[() ]", entity)
        
        res_idxs = []
        last_idx = -1
        for part in split_entity:
            sub_addr = addr[last_idx+1:]
            if part not in sub_addr:
                return None

            last_idx += sub_addr.index(part) + len(part)
            res_idxs += list(range(last_idx-len(part)+1, last_idx+1))
        return res_idxs


    # retrieve data
    addr = record[0]
    entities = record[1:6]
    labels = ["POI", "building", "unit", "floor", "room"]

    # init result
    bio = ["O" for _ in addr]

    # generate
    for entity, label in zip(entities, labels):
        match_res = _match_entity(addr, entity)
        if match_res is None:
            return None

        for idx in match_res:
            if bio[idx] == "floor" and label == "room":
                bio[idx] = "floor|room"
            elif bio[idx] != "O":
                return None
            else:
                bio[idx] = label

    # prefix: B-, I-
    bio = _add_prefix(bio)

    return bio
