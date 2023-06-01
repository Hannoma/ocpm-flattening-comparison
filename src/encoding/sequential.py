from tqdm import tqdm

from ocpa.algo.predictive_monitoring import sequential
from ocpa.algo.predictive_monitoring.obj import Feature_Storage


def generate_trace_prefixes(feature_storage: Feature_Storage, target: tuple, min_trace_length: int = 1,
                            max_trace_length: int = None, index_list='all') -> tuple[list[list[dict]], list[float]]:
    """
    Generate sequential trace prefixes from the feature storage

    :param feature_storage: Feature storage to construct a sequential encoding from.
    :type feature_storage: :class:`Feature Storage <ocpa.algo.predictive_monitoring.obj.Feature_Storage>`
    :param target: The target feature
    :type target: tuple
    :param min_trace_length: The minimum trace length
    :type min_trace_length: int
    :param max_trace_length: The maximum trace length
    :type max_trace_length: int
    :param index_list: list of indices to be encoded as sequences. Default is "all"
    :type index_list: "all" or list(int)

    :return: Tuple of all trace prefixes and their corresponding target values
    :rtype: tuple[list[Any], list[Any]]
    """
    # Use the sequential encoding to generate feature vectors for each trace
    sequences = sequential.construct_sequence(feature_storage, index_list)

    # Generate a list of trace prefixes
    trace_prefixes = []
    target_values = []

    for s in tqdm(sequences):
        result = _generate_prefix_for_sequence((s, target, min_trace_length, max_trace_length))
        trace_prefixes.extend(result[0])
        target_values.extend(result[1])

    return trace_prefixes, target_values


def _generate_prefix_for_sequence(params: tuple[list[dict], tuple, int, int]):
    sequence, target, min_trace_length, max_trace_length = params
    trace_prefixes = []
    target_values = []
    # generate trace prefixes
    for i in range(min_trace_length - 1, len(sequence)):
        if max_trace_length is not None:
            trace_prefixes.append(sequence[max(0, i - max_trace_length + 1): i + 1])
        else:
            trace_prefixes.append(sequence[:i + 1])
        target_values.append(sequence[i][target])
    # remove the target feature from the trace
    for trace in trace_prefixes:
        for event in trace:
            event.pop(target, None)
    return trace_prefixes, target_values
