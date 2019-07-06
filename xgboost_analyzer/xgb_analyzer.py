import re
import tempfile
import sys
import json
from collections import namedtuple
import xgboost as xgb

_Split = namedtuple("_Split", ("key", "value", "smaller"))
class Split(_Split):
    """Represents a single decision point/split in a decision tree"""
    def __repr__(self):
        if self.key is not None:
            direction = "<" if self.smaller else ">="
            return "%s %s %.3f" % (self.key, direction, self.value)
        else: return "None"

class Tree:
    idx_key_ptrn = re.compile("f([0-9]+)")

    def __init__(self, id_, split_key, split_val,
                 left, right,
                 cover=None, value=None, is_leaf=False):
        self.id_ = id_
        self.split_key = split_key
        self.split_val = split_val
        self.left, self.right = left, right
        self.cover = cover
        self.value = value
        self.is_leaf = is_leaf

    def get_path(self, item, cur_path=None):
        """
        Gets decision path for an item.
        :returns: List[Tuple[int, Split, float]]
        """
        if cur_path is None: cur_path = []

        # add self to the path
        if self.is_leaf:
            split = Split(None, None, None)
        else:
            split = Split(self.split_key, self.split_val, item[self.split_key] < self.split_val)
        cur_path.append((self.id_, split, self.value))

        if self.is_leaf: return cur_path
        elif split.smaller: return self.left.get_path(item, cur_path)
        else: return self.right.get_path(item, cur_path)

    @classmethod
    def _to_key(cls, keystr):
        m = cls.idx_key_ptrn.match(keystr)
        if m is not None:
            return int(m.group(1))
        else: return keystr

    @classmethod
    def from_dict(cls, d, weighted_average=True):
        if "leaf" in d:
            return cls(id_=d["nodeid"], split_key=None,
                       split_val=None, left=None, right=None,
                       cover=d["cover"], value=d["leaf"], is_leaf=True
                      )
        else:
            left_d, right_d = d["children"]
            left, right = (cls.from_dict(left_d, weighted_average=weighted_average),
                           cls.from_dict(right_d, weighted_average=weighted_average))
            if weighted_average:
                this_node_val = float(left.value * left.cover + right.value * right.cover)
                this_node_val /= (left.cover + right.cover)
            else:
                this_node_val = (left.value + right.value) * 0.5
            node = cls(d["nodeid"], split_key=cls._to_key(d["split"]),
                       split_val=d["split_condition"], left=left, right=right,
                       cover=d["cover"], value=this_node_val, is_leaf=False,
                      )
            return node

class XGBAnalyzer:
    def __init__(self, trees, booster=None):
        self.trees = [Tree.from_dict(d) for d in trees]
        self.booster = booster

    @classmethod
    def from_booster(cls, booster):
        with tempfile.NamedTemporaryFile("w+t") as tmp:
            booster.dump_model(tmp, with_stats=True, dump_format="json")
            tmp.seek(0)
            trees = json.load(tmp)
        return cls(trees, booster)

    @classmethod
    def from_model(cls, model):
        return cls.from_booster(model.get_booster())

    def get_tree(self, idx): return self.trees[idx]
    def get_paths(self, item):
        return [t.get_path(item) for t in self.trees]
