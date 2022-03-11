from dowhy.causal_identifiers import backdoor
import networkx as nx
import numpy as np
import pandas as pd

def gml_to_string(file):
    """a utility function to parse the .gml file to string"""
    gml_str = ''
    with open(file, 'r') as file:
        for line in file:
            gml_str += line.rstrip()
    return gml_str

def get_backdoor_paths(G, N1, N2):
    """Takes a NetworkX directed graph G, and gives the backdoors between node N1 and node N2."""
    # creating a copy of our graph G that is undirected
    H = G.to_undirected()
    all_possible_paths = [x for x in nx.all_simple_paths(H, N1, N2)]
    graph_nodes = backdoor.Backdoor(G, N1, N2)

    # we apply the is_backdoor function to each path to check if a path is a backdoor path
    backdoor_paths_idx = [graph_nodes.is_backdoor(x) for x in all_possible_paths]

    # finally, we filter out all non-backdoor paths from the list of all paths
    return [i for indx,i in enumerate(all_possible_paths) if backdoor_paths_idx[indx]]


def get_adjustment_variables(G, paths):
    """Takes a NetworkX directed graph G and a list of paths and gives the corresponding adjustment variables."""
    # finally, we add the information to our dataframe, with the path, colliders, and non-colliders
    adjustment_variables = pd.DataFrame(columns=['path', 'colliders_desc', 'non_colliders'])

    for path in paths:
        # we create empty (for now) lists for our colliders and non-colliders
        # we also create a variable for the length of the path
        colliders_desc = np.array([])
        non_colliders = []
        path_len = len(path)

        # we loop through adjacent variables on the path, ignoring the source and target variables as potential colliders
        for node0, node1, node2 in zip(path[0:path_len-2], path[1:path_len-1], path[2:]):
            # if there is an arrow pointing into node1 from both sides on the path, it is a collider
            if G.has_edge(node0, node1) and G.has_edge(node2, node1):
                colliders_desc = np.append(colliders_desc, list(nx.descendants(G,node1)) + [node1]) # so we add it (and all its descendants) to the list
        # we flatten the list of list
        colliders_desc = colliders_desc.flatten()
                
        # any node on the path (excluding the source and target) that is not a collider is a non-collider
        non_colliders = [x for x in path[1:-1] if x not in colliders_desc]

        # finally, we add the information to our dataframe, with the path, colliders, and non-colliders
        adjustment_variables = adjustment_variables.append({'path':path, 'colliders_desc': colliders_desc, 
                                                            'non_colliders': non_colliders}, ignore_index=True)
    
    return adjustment_variables

def get_backdoor_sets(model):
    """Return backdoor sets of a dowhy CausalModel model"""
    model.identify_effect()
    identifier = model.identifier
    return identifier.identify_backdoor(identifier.treatment_name, identifier.outcome_name)

def get_frontdoor_sets(model):
    """Return frontdoor sets of a dowhy CausalModel model (only single variable sets supported)"""
    model.identify_effect()
    identifier = model.identifier
    return identifier.identify_frontdoor()