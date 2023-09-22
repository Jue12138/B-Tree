from __future__ import annotations
from typing import Union, Tuple, List

# basic utilities
import json
import csv
import argparse
import math

###############################################################################
#### B-tree node class ########################################################
###############################################################################

class BNode():
    def  __init__(self,
                  m : int,
                  keys = [],
                  values = [],
                  children = []
                  ): 
        
        self.m = m
        self.keys = keys
        self.values = values
        self.children = children


    #__str__ method optional for grading, but required for visualizing


###############################################################################
#### I/O utils ################################################################
###############################################################################


def load_tree(json_str: str) -> BNode:
    m = None

    def _from_dict(dict_repr) -> BNode:

        if dict_repr == None or dict_repr == {}:
            return None

        new_node = BNode( m = m,
                          keys = dict_repr["k"], 
                          values = dict_repr["v"],
                          children = [_from_dict(dict_repr["c"][i]) for i in range (0, len(dict_repr["c"]))]
        )
        return new_node
    
    try:
        # create the intermediate dict representation
        # and unpack the top keys
        dict_repr = json.loads(json_str)
        if "m" in dict_repr:
            m = dict_repr["m"]
        if "tree" in dict_repr:
            tree = dict_repr["tree"]
        else:
            tree = dict_repr
    except Exception as e:
        print(f"Exception encountered parsing the json string: {json_str}")
        raise e

    # call the recursor to turn the nested dict into a tree of BNodes
    root = _from_dict(tree)
    return root


def dump_tree(root: BNode) -> str:
    def _to_dict(node) -> dict:
        return {
             "k": [node.keys[i] for i in range (0, len(node.keys))],
             "v": [node.values[i] for i in range (0, len(node.values))],
             "c": [(_to_dict(node.children[i]) if node.children[i] is not None else None) for i in range (0, len(node.children)) ],
         }
    
    # create the intermediate dict representation
    # and pack in the top keys
    if root == None:
        dict_repr = {}
    else:
        # call the recursor to turn the BNode tree into a nested dict
        dict_repr = _to_dict(root)
        dict_repr = {"m":root.m, "tree": dict_repr}

    return json.dumps(dict_repr, indent=4)


def trace_from_file(fname: str) -> dict:
    def parse_tup(tup):
        assert len(tup) > 0, "Trace file must not contain any extra empty lines except for the final line"
        assert len(tup) != 1, (f"Only one line in trace file should have a single value,",
                               f" it should be first (already parsed here), and should be the integer m, got unexpected line {tup}") 
        if tup[0]   == "ins"  : return {"op":tup[0], "k":int(tup[1]), "v":tup[2]}
        elif tup[0] == "del"  : return {"op":tup[0], "k":int(tup[1])}
        elif tup[0] == "load" : return {"op":tup[0], "path": tup[1]}
        elif tup[0] == "dump" : return {"op":tup[0], "path": tup[1]}
        elif tup[0] == "qry" : return {"op":tup[0], "k": int(tup[1])}
        elif tup[0] == "qry_path" : return {"op":tup[0], "path": tup[1]}
        else:
            raise ValueError

    with open(fname, "r") as f:
        reader = csv.reader(f)
        try:
            lines = [l for l in reader]
            m = int(lines[0][0])
            full_trace = [parse_tup(line) for line in lines[1:]]
        except Exception as e:
            print(f"Error while parsing trace file...")
            raise e
        
        load_paths = [(idx, tup["path"]) for idx,tup in enumerate(full_trace) if tup["op"]=="load"]
        assert (len(load_paths) in [0,1]), "If trace includes a load command there must be only one."

        if len(load_paths) == 1:
            idx, path = load_paths[0]
            assert idx == 0, "If trace includes a load command it should be the first op, second line of file"
            init_path = path
        else:
            init_path = None

        dump_paths = [tup["path"] for tup in full_trace if tup["op"]=="dump"]
        assert len(dump_paths) > 0, "Trace file must contain at least one dump command."
        
        mixed_trace = [tup for tup in full_trace if tup["op"]!="load"]
        assert len(mixed_trace) >= 1, "Number of variable ops (ins,del,dump) in trace must be >= 1"

        query_cmds = [tup for tup in full_trace if tup["op"]=="qry"]
        if len(query_cmds) > 0:
            assert mixed_trace[-2]["op"] == "dump", "If trace contains qry cmds, second to last should be a dump"
            assert mixed_trace[-1]["op"] == "qry_path", "If trace contains qry cmds, last line should be a qry_path"
            query_path = mixed_trace[-1]["path"]
            mixed_trace = mixed_trace[:-1]
        else:
            query_path = None
            assert mixed_trace[-1]["op"] == "dump", "If trace has no qry cmds, last line should be a dump"

    return dict(mixed_trace=mixed_trace, m=m, init_path=init_path, query_path=query_path)


def trace_to_file(m: int, init_path: str, trace: list, query_path: str, out_path: str) -> None:
    with open(out_path, "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow([m])
        if init_path: 
            writer.writerow(["load", init_path])
        writer.writerows([d.values() for d in trace])
        if query_path: 
            writer.writerow(["qry_path", query_path])


def query_values_to_file(keychains_values: List[dict], query_path: str) -> None:
    with open(f"{query_path}", "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerows([[k]+keychain+[value] for k,keychain,value in [d.values() for d in keychains_values]])


def parse_autograder_dump(dump_json="./grader_dump.json", 
                          dump_out_dir="./grader_output",
                          overwrite=False):
                    
    # Load the file containing the json autograder dump
    with open(dump_json, "r") as infile:
        dump_obj = json.load(infile)
    
    # create the output dir where the autograder data will be written out
    import os
    try:
        os.mkdir(dump_out_dir)
    except FileExistsError as e:
        if not overwrite:
            print(f"Output dir for dumping the parsed grader data {dump_out_dir} "
                  f"already exists! Pass overwrite flag to overwrite contents")
            return
        else:
            print(f"Overwriting contents of {dump_out_dir}")
    except Exception as e:
        raise e
    
    # the 3 test case components
    trace_list   = dump_obj["tracefile"]
    tree_dumps   = dump_obj["tree_dumps"]
    query_values = dump_obj["query_values"]

    # parse special parts of the trace
    m = trace_list[0][0]
    contains_load = trace_list[1][0]=="load"
    contains_query = trace_list[-1][0]=="qry_path"
    init_path = trace_list[1][1] if contains_load else None
    query_path = trace_list[-1][1] if contains_query else None
    
    if contains_load and contains_query: # both
        trace_ops = trace_list[2:-1]
    elif contains_load: # no query
        trace_ops = trace_list[2:]
    elif contains_query: # no load
        trace_ops = trace_list[1:-1]
    else:
        raise ValueError("Unexpected trace op list contents")
    
    # make it a dictionary since the original starter code definition
    # of trace_to_file expects this
    trace_ops = [{f"k_{i}":v for i,v in enumerate(op)} for op in trace_ops]
    
    trace_to_file(m=m,
                  init_path=init_path,
                  trace=trace_ops,
                  query_path=query_path,
                  out_path=f"{dump_out_dir}/grader.trace")

    for i,tree_str in enumerate(tree_dumps):
        with open(f"{dump_out_dir}/tree_dump_{i+1}.json", "w") as outfile:
            # this is more human readable because indent=4 for json.dump 
            # in the starter dump_tree
            expanded_tree_json = dump_tree(load_tree(tree_str)) 
            outfile.write(expanded_tree_json)
            # but if you haven't gotten dump working (the above will give `{}`), try this
            # outfile.write(tree_str)

    query_values_to_file(keychains_values=query_values,
                         query_path=f"{dump_out_dir}/queries.values")


###############################################################################
#### Operator methods #########################################################
###############################################################################
def insert(m, parent, node, k, v):
    
    if node is None:
        node = BNode(m, [k], [v], [None, None])
        return node
    else: # node is not None
        n = len(node.keys) # number of the keys
        if node.children[0] is None: # Leaf can insert
            if k > node.keys[n-1]: #check if new k > last key of the node
                node.keys.insert(n, k)
                node.values.insert(n, v)
                node.children.insert(0, None)
            else:
                for i in range(0,n):
                    if k == node.keys[i]:
                        break
                    elif k < node.keys[i]:
                        node.keys.insert(i, k)
                        node.values.insert(i, v)
                        node.children.insert(0, None)
                        break

        else: # Not leaf go recusive
            if k > node.keys[n-1]:
                node = insert(m, node, node.children[n], k, v) # check if new k > last key of the node
            else:
                for i in range(0,n): # check the children k will go
                    if k == node.keys[i]:
                        break
                    elif k < node.keys[i]:
                        node = insert(m, node, node.children[i], k, v)
                        break

        numKeys = len(node.keys) # the number of node's keys
        if numKeys > m - 1:
            if parent is None: # split and node is root
                parent = split(None, node) 
            else:
                h = parent.children.index(node)          
                if h > 0 and len(parent.children[h-1].keys) < m - 1: 
                    parent = rotationLeft(parent, parent.children[h-1], node)
                elif h < len(parent.children)-1 and len(parent.children[h+1].keys) < m - 1:
                    parent = rotationRight(parent, parent.children[h+1], node)
                else:
                    parent = split(parent, node)

        if parent is None:
            return node
        else:
            return parent 

def rotationLeft(parent, node1, node2):
    # rotate from node2 to node1

    k = node2.keys[0]
    v = node2.values[0]

    node2.keys[0:1] = [] # remove the key and value from node2
    node2.values[0:1] = []

    i = parent.children.index(node2)

    parent.keys.insert(i, k)
    parent.values.insert(i, v)

    newk = parent.keys[i-1]
    newv = parent.values[i-1]

    parent.keys[i-1:i] = []
    parent.values[i-1:i] = []

    node1.keys.insert(len(node1.keys), newk)
    node1.values.insert(len(node1.values), newv)

    child = node2.children[0]
    node1.children.insert(len(node1.children), child)
    node2.children[0:1] = []

    return parent 

def rotationRight(parent, node1, node2):
    # rotate from node2 to node1

    k = node2.keys[len(node2.keys)-1]
    v = node2.values[len(node2.values)-1]

    node2.keys[len(node2.keys)-1:len(node2.keys)] = [] # remove the key and value from node2
    node2.values[len(node2.values)-1:len(node2.values)] = []

    i = parent.children.index(node2)

    parent.keys.insert(i, k)
    parent.values.insert(i, v)

    newk = parent.keys[i+1]
    newv = parent.values[i+1]

    parent.keys[i+1:i+2] = []
    parent.values[i+1:i+2] = []

    node1.keys.insert(0, newk)
    node1.values.insert(0, newv)

    child = node2.children[len(node2.children)-1]
    node1.children.insert(0, child)
    node2.children[len(node2.children)-1:len(node2.children)] = []

    return parent

def split(parent, node):
    # parent can be null

    n = len(node.keys)
    middle = math.floor((n-1)/2)

    leftNodeK = node.keys[:middle]
    leftNodeV = node.values[:middle]

    rightNodeK = node.keys[middle+1:]
    rightNodeV = node.values[middle+1:]

    leftNodeC = []
    rightNodeC = []
    if node.children[0] is not None:
        leftNodeC = node.children[:middle+1]
        rightNodeC = node.children[middle+1:]
    else:
        w = len(leftNodeK)
        u = len(rightNodeK)

        for i in range(0, w+1):
            leftNodeC.insert(0, None)
    
        for i in range(0, u+1):
            rightNodeC.insert(0, None)

    newLeft = BNode(node.m, leftNodeK, leftNodeV, leftNodeC)
    newRight = BNode(node.m, rightNodeK, rightNodeV, rightNodeC)

    if parent is None:
        parent = BNode(node.m, [node.keys[middle]], [node.values[middle]], [newLeft, newRight])
    else:
        parent.children.remove(node)
        k = node.keys[middle]
        v = node.values[middle]
        n = len(parent.keys)

        if k > parent.keys[n-1]:

            parent.keys.insert(n, k)
            parent.values.insert(n, v)
            parent.children.insert(len(parent.children), newLeft)
            parent.children.insert(len(parent.children), newRight)

        else:
            for i in range(0, n):
                if k < parent.keys[i]:
                    parent.keys.insert(i, k)
                    parent.values.insert(i, v)

                    parent.children.insert(i, newLeft)
                    parent.children.insert(i+1, newRight)
                    break

    return parent


def delete(m, parent, node, k):
    if node is not None:       
        n = len(node.keys)
        if node.children[0] is None: # leaf node

            if k > node.keys[n-1]: # Not found
                if parent is None:
                    return node

            else:
                for i in range(0,n):
                    if k == node.keys[i]: # delete the key
                        node.keys[i:i+1] = []
                        node.values[i:i+1] = []
                        node.children.remove(None)
                        break
                        #check if need rotation or merge

                if parent is None:
                    return node

        else: # not leaf node
            if k > node.keys[n-1]: # not found go to lower level
                node = delete(m, node, node.children[n], k)

            else:
                for i in range(0, n):
                    if k == node.keys[i]: # delete the key
                        node.keys[i:i+1] = []
                        node.values[i:i+1] = []

                        node = successor(m, node, node, i, node.children[i+1])
                        break
                    elif k < node.keys[i]: # not found go to lower level
                        node = delete(m, node, node.children[i], k)
                        break

            if parent is None:
                if len(node.keys) == 0:
                    return node.children[0]
                else:
                    return node
  
        j = len(node.keys)
        h = parent.children.index(node)

        if j < math.ceil(m/2) - 1:
            if h > 0 and len(parent.children[h-1].keys) > math.ceil(m/2) - 1: # rotation from left sibling to node
                parent = rotationRight(parent, node, parent.children[h-1])
            elif h < len(parent.keys) and len(parent.children[h+1].keys) > math.ceil(m/2) - 1: # rotation from right sibling to node
                parent = rotationLeft(parent, node, parent.children[h+1])
            elif h > 0 and len(parent.children[h-1].keys) == math.ceil(m/2) - 1: # merge from left sibling to node
                parent = mergeLeft(parent, parent.children[h-1], node)
            elif h < len(parent.keys) and len(parent.children[h+1].keys) == math.ceil(m/2) - 1: # merge from right sibling to node
                parent = mergeRight(parent, parent.children[h+1], node)

        return parent




def mergeLeft(parent, node1, node2):
    # All keys and values go to node1, delete parent and node2
    h = parent.children.index(node1)
    k = len(node2.keys)

    node1.keys.insert(len(node1.keys), parent.keys[h])
    node1.values.insert(len(node1.values), parent.values[h])

    for i in range(0, k):
        node1.keys.insert(len(node1.keys), node2.keys[i])
        node1.values.insert(len(node1.values), node2.values[i])

    n = len(node2.children)

    for i in range(0, n):
        node1.children.insert(len(node1.children), node2.children[i])

    parent.children[h+1:h+2] = []
    parent.keys[h:h+1] = []
    parent.values[h:h+1] = []

    return parent

def mergeRight(parent, node1, node2):
    # All keys and values go to node1, delete parent and node2
    h = parent.children.index(node1)
    k = len(node2.keys)

    node1.keys.insert(0, parent.keys[h-1])
    node1.values.insert(0, parent.values[h-1])

    for i in range(k-1, -1, -1):
        node1.keys.insert(0, node2.keys[i])
        node1.values.insert(0, node2.values[i])

    n = len(node2.children)

    for i in range(n-1, -1, -1):
        node1.children.insert(0, node2.children[i])

    parent.children[h-1:h] = []
    parent.keys[h-1:h] = []
    parent.values[h-1:h] = []

    return parent


def successor(m, root, parent, index, node):
    if node.children[0] is None: # node is leaf
        root.keys.insert(index, node.keys[0]) # replace
        root.values.insert(index, node.values[0])

        node.keys[0:1] = [] # remove key from leaf node
        node.values[0:1] = []

        node.children.remove(None)

    else: # did not find inorder successor yet
        node = successor(m, root, node, index, node.children[0])

    
    h = parent.children.index(node)
    j = len(node.keys)

    if j < math.ceil(m/2) - 1:
        if h > 0 and len(parent.children[h-1].keys) > math.ceil(m/2) - 1: # rotation from left sibling to node
            parent = rotationRight(parent, node, parent.children[h-1])
        elif h < len(parent.keys) and len(parent.children[h+1].keys) > math.ceil(m/2) - 1: # rotation from right sibling to node
            parent = rotationLeft(parent, node, parent.children[h+1])
        elif h > 0 and len(parent.children[h-1].keys) == math.ceil(m/2) - 1: # merge from left sibling to node
            parent = mergeLeft(parent, parent.children[h-1], node)
        elif h < len(parent.keys) and len(parent.children[h+1].keys) == math.ceil(m/2) - 1: # merge from right sibling to node
            parent = mergeRight(parent, parent.children[h+1], node)

    return parent



    
def query(m, node, k, lst, V):
    n = len(node.keys)

    if k > node.keys[n-1]:

        lst.insert(len(lst), node.keys[n-1])
        if node.children[0] is not None:
            V = query(m, node.children[n], k, lst, V)

    else:
        for i in range(0, n):

            if k == node.keys[i]:

                lst.insert(len(lst), k)
                V = node.values[i]
                break
            elif k < node.keys[i]:

                lst.insert(len(lst), node.keys[i])
                if node.children[0] is not None:
                    V = query(m, node.children[i], k, lst, V) 
                break

    return V

###############################################################################
#### Main driver method #######################################################
###############################################################################
def main(args):
    print(args)

    data  = trace_from_file(f"{args.tracefile}")
    
    m_val = data["m"]
    new_trace = data["mixed_trace"]
    i_path = data['init_path']
    query_path = data["query_path"]
    
    if i_path:
        with open(f"{i_path}", "r") as infile:
            tree = load_tree(infile.read())
        if tree: assert tree.m == m_val, "Should never have a mismatch between trace m value and init tree m value."
    else:
        tree = None

    query_results = []

    for cmd in new_trace:
        
        if cmd["op"] == "ins":
            k, v = cmd["k"], cmd["v"]
            tree = insert(m_val, None, tree, k, v)
            # Need to perform an insertion of key k and value v on tree

        elif cmd["op"] == "del":
            k= cmd["k"]
            tree = delete(m_val, None, tree, k)
            # Need to perform an deletion of key k on tree

        elif cmd["op"] == "qry":
            k = cmd["k"]
            # Need to perform a query on tree for key k
            # and return the "keychain" or list of keys along the path to (and including) the key k
            # as well as the associated value, else None if the key is not found
            keychain = []
            value = None
            value = query(m_val, tree, k, keychain, value)
            if value is None:
                value = 'null'
            query_results.append((k, keychain, value))

        elif cmd["op"] == "dump":
            path = f"{cmd['path']}"
            with open(path, "w") as outfile:
                outfile.write(dump_tree(tree))
        else:
            raise ValueError(f"Unknown op code in tracefile command: {cmd}")

    if query_path:
        with open(f"{query_path}", "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerows([[k]+keychain+[value] for k,keychain,value in query_results])
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-tf", 
                        "--tracefile", 
                        required=True)
    
    args = parser.parse_args()
    main(args)