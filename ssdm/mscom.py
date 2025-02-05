import igraph as ig
import numpy as np
import librosa
from .scluster import reindex
from .formatting import hier2multi, mireval2hier, mireval2multi

def refine_global_membership_level(g, current_membership, base_resolution, dr, max_resolution=5.0):
    """
    Attempts to refine the global membership vector for graph g by applying community_multilevel 
    at a common resolution (starting at base_resolution + dr) for all communities.
    
    For each unique community in current_membership, the induced subgraph is extracted and 
    community_multilevel is run at resolution 'new_res'. This new_res is increased (by dr) 
    until at least one community is split nontrivially (i.e. the induced subgraph yields more 
    than one group) or max_resolution is reached.
    
    If any community is refined, the global membership is updated for those vertices by concatenating 
    the parent's label with the sub-community label (separated by an underscore).
    
    Returns:
      updated (list of str): The new, refined global membership vector.
      changed (bool): True if any vertex’s label was updated.
      used_resolution (float): The resolution at which at least one community split nontrivially.
    """
    new_res = base_resolution + dr
    while new_res <= max_resolution:
        sub_refinements = {}  # Will map a parent's label to the sub-membership (list of ints) if refined.
        any_split = False
        for comm in sorted(set(current_membership)):
            # Get indices of vertices that share this parent's label.
            indices = [i for i, label in enumerate(current_membership) if label == comm]
            if len(indices) <= 1:
                sub_refinements[comm] = None
                continue  # Skip trivial communities.
            subg = g.subgraph(indices)
            sub_cluster = subg.community_multilevel(weights="weight", resolution=new_res)
            if len(set(sub_cluster.membership)) > 1:
                any_split = True
                sub_refinements[comm] = sub_cluster.membership
            else:
                sub_refinements[comm] = None
        if any_split:
            # Use the current resolution new_res to update all communities that can be refined.
            updated = current_membership.copy()
            for comm in sorted(set(current_membership)):
                indices = [i for i, label in enumerate(current_membership) if label == comm]
                if len(indices) <= 1:
                    continue
                if sub_refinements[comm] is not None:
                    for idx, v in enumerate(indices):
                        updated[v] = current_membership[v] + str(sub_refinements[comm][idx])
            return updated, True, new_res
        new_res += dr
    # If no community could be refined (at any resolution up to max_resolution), return current membership.
    return current_membership, False, max_resolution

def process_graph(g, dr, max_resolution=5.0):
    """
    Computes a hierarchy of global membership vectors (lists of strings) that refine the segmentation 
    of graph g.
    
    The initial segmentation is computed at resolution 0. Then, at each level, refine_global_membership_level 
    is called with the current global membership and the current base_resolution; if at least one community 
    splits nontrivially at that common resolution, a new global membership vector is produced (by updating 
    only the communities that split) and recorded. The process continues until no community refines further.
    
    Parameters:
      g           : igraph.Graph (weighted, undirected)
      dr          : float, the resolution increment.
      max_resolution: float, maximum resolution (default 5.0)
    
    Returns:
      levels (list of list of str): A list of global membership vectors (one per level) from coarse to fine.
    """
    # Initial segmentation at resolution 0.
    cluster = g.community_multilevel(weights="weight", resolution=0)
    global_membership = [str(x) for x in cluster.membership]
    levels = [global_membership.copy()]
    
    base_resolution = 0
    while True:
        new_membership, changed, used_res = refine_global_membership_level(g, global_membership, base_resolution, dr, max_resolution)
        if not changed:
            break
        levels.append(new_membership.copy())
        global_membership = new_membership
        base_resolution = used_res  # Update the base resolution to the one used in this refinement.
    return np.array(levels)


def segment(S, ts, max_r=5.0, dr=None):
    """Convert combined recurrence matrix to hierarchical segmentation.
    
    Args:
        S: Combined recurrence matrix (numpy array)
        ts: Time stamps array
    
    Returns:
        jams.Annotation: Multi-level segmentation annotation
    """
    # Force symmetry
    S = (S + S.T)/2
    
    # Resolution parameter
    if dr is None:
        dr = S.sum() / len(S) * 0.5
        print(dr)
    g_full = ig.Graph.Weighted_Adjacency(S.tolist(), mode=ig.ADJ_UNDIRECTED)
    memberships = process_graph(g_full, dr=dr, max_resolution=max_r)
    # return memberships
    # Convert memberships to intervals and labels
    intervals = []
    labels = []
    
    for level_membership in memberships:
        # Find boundaries where labels change (including start)
        bound_idxs = np.concatenate([[0], 
                                    1 + np.flatnonzero(level_membership[:-1] != level_membership[1:]),
                                    [len(ts) - 1]])
        
        # Convert indices to timestamps for intervals
        level_intervals = np.array([[ts[bound_idxs[i]], ts[bound_idxs[i+1]]] 
                                  for i in range(len(bound_idxs)-1)])
        
        # Get labels for segments
        level_labels = [str(level_membership[idx]) for idx in bound_idxs[:-1]]
        
        intervals.append(level_intervals)
        labels.append(level_labels)
    
    hier = reindex(mireval2hier(intervals, labels))
    # reindex and Convert to JAMS annotation
    return hier2multi(hier)