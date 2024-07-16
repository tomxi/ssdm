# From Brian McFee

from collections import Counter
import numpy as np
import jams
import re

HMX_MAPPING = {'altchorus': 'chorus',
 'bigoutro': 'outro',
 '(.*)\\d+.*': '\\1',
 '(.*)_instrumental': '\\1',
 'chorus.*': 'chorus',
 'instbridge': 'bridge',
 'instchorus': 'chorus',
 'instintro': 'intro',
 'instrumentalverse': 'verse',
 'intropt2': 'intro',
 'miniverse': 'verse',
 'quietchorus': 'chorus',
 'rhythmlessintro': 'intro',
 'verse_slow': 'verse',
 'verseinst': 'verse',
 'versepart': 'verse',
 'vocaloutro': 'outro'}
HMX_SUBS = [(re.compile(x), HMX_MAPPING[x]) for x in HMX_MAPPING]

def flatten_labels(labels, dataset='hmx'):
    """
    Apply Hierarchy Expansion Rules for a specific dataset
    dataset can be hmx, slm, rwc, or jsd
    """
    # Are there other rules for segment coalescing?
    # VerseA, VerseB -> Verse
    # verse_(guitar solo) ...
    # _(.*)
    # /.*
    if dataset == 'slm':
        return ['{}'.format(_.replace("'", '')) for _ in labels]
    elif dataset == 'hmx':
        # Apply all label substitutions
        labels_out = []
        for lab in labels:
            for (pattern, replace) in HMX_SUBS:
                lab = pattern.sub(replace, lab)
            labels_out.append(lab)
        return labels_out
    elif dataset == 'jsd':
        return ['theme' if 'theme' in _ else _ for _ in labels]
    elif dataset == 'rwc':
        return ['{}'.format(_.split(' ')[0]) for _ in labels]
    else:
        raise NotImplementedError('bad dataset')



def expand_labels(labels, dataset='hmx'):
    
    flat = flatten_labels(labels, dataset)

    seg_counter = Counter()
    
    expanded = []
    for label in flat:
        expanded.append('{:s}_{:d}'.format(label, seg_counter[label]))
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


def expand_hierarchy(_ann, dataset='hmx', always_include=False):
    
    ints, labs = _ann.to_interval_values()
    
    labs_up = flatten_labels(labs, dataset)
    labs_down = expand_labels(labs_up, dataset)
    
    annotations = []
    
    # If contraction did anything, include it.  Otherwise don't.
    if always_include or not issame(labs_up, labs):
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
    if always_include or not issame(labs_down, labs):
        ann = jams.Annotation(namespace='segment_open', time=_ann.time, duration=_ann.duration)

        for ival, label in zip(ints, labs_down):
            ann.append(time=ival[0], duration=ival[1]-ival[0], value=label)
        annotations.append(ann)

    return annotations