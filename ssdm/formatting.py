import jams
import numpy as np

def multi2hier(anno)-> list:
    n_lvl_list = [obs.value['level'] for obs in anno]
    n_lvl = max(n_lvl_list) + 1
    hier = [[[],[]] for i in range(n_lvl)]
    for obs in anno:
        lvl = obs.value['level']
        label = obs.value['label']
        interval = [obs.time, obs.time+obs.duration]
        hier[lvl][0].append(interval)
        hier[lvl][1].append(f'{label}')
    return hier


def hier2multi(hier) -> jams.Annotation:
    anno = jams.Annotation(namespace='multi_segment')
    anno.duration = hier[0][-1][-1][-1]
    anno.time = hier[0][0][0][0]
    for layer, (intervals, labels) in enumerate(hier):
        for ival, label in zip(intervals, labels):
            anno.append(time=ival[0],
                        duration=ival[1]-ival[0],
                        value={'label': str(label), 'level': layer})
    return anno


def hier2mireval(hier) -> tuple:
    intervals = []
    labels = []
    for itv, lbl in hier:
        intervals.append(np.array(itv, dtype=float))
        labels.append(lbl)

    return intervals, labels


def mireval2hier(itvls: np.ndarray, labels: list) -> list:
    hier = []
    n_lvl = len(labels)
    for lvl in range(n_lvl):
        lvl_anno = [itvls[lvl], labels[lvl]]
        hier.append(lvl_anno)
    return hier


def multi2mireval(anno) -> tuple:
    return hier2mireval(multi2hier(anno))


def mireval2multi(itvls: np.ndarray, labels: list) -> jams.Annotation:
    return hier2multi(mireval2hier(itvls, labels))


def openseg2multi(
    annos: list
) -> jams.Annotation:
    multi_anno = jams.Annotation(namespace='multi_segment')

    for lvl, openseg in enumerate(annos):
        for obs in openseg:
            multi_anno.append(time=obs.time,
                              duration=obs.duration,
                              value={'label': obs.value, 'level': lvl},
                             )
    return multi_anno


def multi2mirevalflat(multi_anno, layer=-1):
    all_itvls, all_labels = multi2mireval(multi_anno)
    return all_itvls[layer], all_labels[layer]


def multi2openseg(multi_anno, layer=-1):
    itvls, labels = multi2mirevalflat(multi_anno, layer)
    anno = jams.Annotation(namespace='segment_open')
    for ival, label in zip(itvls, labels):
        anno.append(time=ival[0],
                    duration=ival[1]-ival[0],
                    value=str(label))
    return anno


def openseg2mirevalflat(openseg_anno):
    return multi2mirevalflat(openseg2multi([openseg_anno]))