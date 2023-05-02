# From Brian McFee

from collections import Counter
import numpy as np
import jams

def flatten_labels(labels):
    
    # Are there other rules for segment coalescing?
    # VerseA, VerseB -> Verse
    # verse_(guitar solo) ...
    # _(.*)
    # /.*
    return ['{}'.format(_.replace("'", '')) for _ in labels]


def expand_labels(labels):
    
    flat = flatten_labels(labels)

    seg_counter = Counter()
    
    expanded = []
    for label in flat:
        expanded.append('{:s}{:d}'.format(label, seg_counter[label]))
        seg_counter[label] += 1
        
    return expanded


def issame(labs1, labs2):
    
    # Hash the strings
    h1 = [hash(_) for _ in labs1]
    h2 = [hash(_) for _ in labs2]
    
    # Build the segment label agreement matrix
    a1 = np.equal.outer(h1, h1)
    a2 = np.equal.outer(h2, h2)
    
    # Labelings are the same if the segment label agreements are identical
    return np.all(a1 == a2)


from copy import deepcopy


def expand_hierarchy(_ann):
    
    ints, labs = _ann.to_interval_values()
    
    labs_up = flatten_labels(labs)
    labs_down = expand_labels(labs_up)
    
    level = 0
    
    annotations = []
    
    # If contraction did anything, include it.  Otherwise don't.
    if not issame(labs_up, labs):
        ann = jams.Annotation(namespace='segment_open', time=_ann.time, duration=_ann.duration)

        for ival, label in zip(ints, labs_up):
            ann.append(time=ival[0], duration=ival[1]-ival[0], value=label)
        annotations.append(ann)

    # Push the original segmentation
    ann = jams.Annotation(namespace='segment_open', time=_ann.time, duration=_ann.duration)
    for ival, label in zip(ints, labs):
        ann.append(time=ival[0], duration=ival[1]-ival[0], value=label)
    annotations.append(ann)
    
     # If expansion did anything, include it.  Otherwise don't.
    if not issame(labs_down, labs):
        ann = jams.Annotation(namespace='segment_open', time=_ann.time, duration=_ann.duration)

        for ival, label in zip(ints, labs_down):
            ann.append(time=ival[0], duration=ival[1]-ival[0], value=label)
        annotations.append(ann)

    return annotations