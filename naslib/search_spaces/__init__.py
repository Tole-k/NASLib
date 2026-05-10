from .simple_cell.graph import SimpleCellSearchSpace
from .nasbench301.graph import NasBench301SearchSpace
try:
    from .nasbench101.graph import NasBench101SearchSpace
except ModuleNotFoundError:
    NasBench101SearchSpace = None
from .nasbench201.graph import NasBench201SearchSpace
from .nasbenchnlp.graph import NasBenchNLPSearchSpace
from .nasbenchasr.graph import NasBenchASRSearchSpace
from .natsbenchsize.graph import NATSBenchSizeSearchSpace
from .hierarchical.graph import HierarchicalSearchSpace
try:
    from .transbench101.graph import TransBench101SearchSpaceMicro
    from .transbench101.graph import TransBench101SearchSpaceMacro
except ModuleNotFoundError:
    TransBench101SearchSpaceMicro = None
    TransBench101SearchSpaceMacro = None

try:
    from .transbench101.api import TransNASBenchAPI
except ModuleNotFoundError:
    TransNASBenchAPI = None

# FIXME Adapt to all search spaces
supported_search_spaces = {
    "nasbench201": NasBench201SearchSpace,
    "nasbench301": NasBench301SearchSpace,
}

if NasBench101SearchSpace is not None:
    supported_search_spaces["nasbench101"] = NasBench101SearchSpace
if TransBench101SearchSpaceMicro is not None:
    supported_search_spaces["transbench101_micro"] = TransBench101SearchSpaceMicro
if TransBench101SearchSpaceMacro is not None:
    supported_search_spaces["transbench101_macro"] = TransBench101SearchSpaceMacro

dataset_n_classes = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet16-120": 120,
    "svhn": 10,
    "ninapro": 18,
    "scifar100": 100,
}

dataset_to_channels = {
    "cifar10": 3,
    "cifar100": 3,
    "imagenet16-120": 3,
    "svhn": 3,
    "ninapro": 1,
    "scifar100": 3,
}

def get_search_space(name, dataset):
    search_space_cls = supported_search_spaces[name.lower()]

    try:
        in_channels = dataset_to_channels[dataset.lower()]
    except KeyError:
        in_channels = 3

    try:
        n_classes = dataset_n_classes[dataset.lower()]
    except KeyError:
        n_classes = -1

    if name == 'transbench101_micro' or name == 'transbench101_macro':
        create_graph = True if dataset.lower() in ['svhn', 'ninapro', 'scifar100'] else False
        return search_space_cls(dataset=dataset,
                                use_small_model=True,
                                create_graph=create_graph,
                                n_classes=n_classes,
                                in_channels=in_channels)
    elif name == 'nasbench301':
        auxiliary = True if dataset.lower() == "cifar10" else False
        return search_space_cls(n_classes=n_classes, in_channels=in_channels,
                                auxiliary=auxiliary)
    elif name == 'nasbench201':
        return search_space_cls(n_classes=n_classes, in_channels=in_channels)
    elif name == 'nasbench101':
        return search_space_cls(n_classes=n_classes)
    else:
        raise NotImplementedError(f'{name} search space not implemented')
