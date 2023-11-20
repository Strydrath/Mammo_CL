from itertools import tee
from typing import (
    Sequence,
    Optional,
    Dict,
    TypeVar,
    Union,
    Any,
    List,
    Callable,
    Set,
    Tuple,
    Iterable,
    Generator,
)

import torch
from avalanche.benchmarks.scenarios.classification_scenario import (
    ClassificationScenario,
)

from avalanche.benchmarks.scenarios.dataset_scenario import (
    DatasetScenario,
    DatasetStream,
    FactoryBasedStream,
    StreamDef,
    TStreamsUserDict,
)
from avalanche.benchmarks.scenarios.detection_scenario import DetectionScenario
from avalanche.benchmarks.scenarios.generic_benchmark_creation import *
from avalanche.benchmarks.scenarios import (
    StreamUserDef,
)
from avalanche.benchmarks.scenarios.generic_scenario import (
    CLStream,
    DatasetExperience,
    SizedCLStream,
)
from avalanche.benchmarks.scenarios.lazy_dataset_sequence import (
    LazyDatasetSequence,
)

from di_scenario import DIScenario

from avalanche.benchmarks.utils.classification_dataset import (
    SupervisedClassificationDataset,
    SupportedDataset,
    as_supervised_classification_dataset,
    make_classification_dataset,
    concat_classification_datasets,
    concat_classification_datasets_sequentially,
)
from avalanche.benchmarks.utils.data import AvalancheDataset


def di_benchmark(
    train_dataset: Union[Sequence[SupportedDataset], SupportedDataset],
    test_dataset: Union[Sequence[SupportedDataset], SupportedDataset],
    n_experiences: int,
    *,
    task_labels: bool = False,
    shuffle: bool = True,
    seed: Optional[int] = None,
    balance_experiences: bool = False,
    min_class_patterns_in_exp: int = 0,
    fixed_exp_assignment: Optional[Sequence[Sequence[int]]] = None,
    train_transform=None,
    eval_transform=None,
    reproducibility_data: Optional[Dict[str, Any]] = None,
) -> DIScenario:


    seq_train_dataset, seq_test_dataset = train_dataset, test_dataset
    exp_patterns = None
    exp_patterns_test = None
    if isinstance(train_dataset, (list, tuple)):
        if not isinstance(test_dataset, (list, tuple)):
            raise ValueError(
                "If a list is passed for train_dataset, "
                "then test_dataset must be a list, too."
            )

        if len(train_dataset) != len(test_dataset):
            raise ValueError(
                "Train/test dataset lists must contain the "
                "exact same number of datasets"
            )
        
        exp_patterns = [[] for _ in range(n_experiences)]
        exp_patterns_test = [[] for _ in range(n_experiences)]
        last_id = 0
        for i,dataset in enumerate(train_dataset):
            exp_patterns[i].extend(list(range(last_id,last_id+len(dataset))))
            last_id += len(dataset)
        last_id = 0
        for i,dataset in enumerate(test_dataset):
            exp_patterns_test[i].extend(list(range(last_id,last_id+len(dataset))))
            last_id += len(dataset)

        train_dataset_sup = list(
            map(as_supervised_classification_dataset, train_dataset)
        )
        test_dataset_sup = list(map(as_supervised_classification_dataset, test_dataset))
        seq_train_dataset = concat_classification_datasets(train_dataset_sup)
        seq_test_dataset = concat_classification_datasets(test_dataset_sup)
    else:
        seq_train_dataset = as_supervised_classification_dataset(train_dataset)
        seq_test_dataset = as_supervised_classification_dataset(test_dataset)

    transform_groups = dict(train=(train_transform, None), eval=(eval_transform, None))

    # Set transformation groups
    final_train_dataset = make_classification_dataset(
        seq_train_dataset,
        transform_groups=transform_groups,
        initial_transform_group="train",
    )

    final_test_dataset = make_classification_dataset(
        seq_test_dataset,
        transform_groups=transform_groups,
        initial_transform_group="eval",
    )

    return DIScenario(
        train_dataset=final_train_dataset,
        test_dataset=final_test_dataset,
        n_experiences=n_experiences,
        task_labels=task_labels,
        shuffle=shuffle,
        seed=seed,
        balance_experiences=balance_experiences,
        min_class_patterns_in_exp=min_class_patterns_in_exp,
        fixed_exp_assignment=exp_patterns,
        fixed_exp_assignment_test=exp_patterns_test,
        reproducibility_data=reproducibility_data,
    )

