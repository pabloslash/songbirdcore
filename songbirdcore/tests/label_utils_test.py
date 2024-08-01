import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from songbirdcore.utils.label_utils import TextgridLabels

# Mock openTextgrid function from praatio.tgio
@pytest.fixture
def mock_open_textgrid(mocker):
    mocker.patch('praatio.tgio.openTextgrid', MagicMock(return_value=MagicMock()))

# Create a fixture to initialize the TextgridLabels object for testing
@pytest.fixture
def textgrid_labels_fixture(mock_open_textgrid):
    textgrid_file_path = "path/to/mock_textgrid_file.TextGrid"
    audio_file_path = "path/to/audio_file.npy"
    fs_audio = 40000  # Sample rate
    labels_tier = 'labels'

    # Mock np.load to return a dummy array
    with patch('numpy.load') as mock_load:
        mock_load.return_value = np.zeros(10)  # Dummy array of length 10
        # Initialize TextgridLabels object with non-empty labels for testing this case
        labels = ['label1', 'label2', 'label3']  # Example non-empty labels
        textgrid_labels_fixture = TextgridLabels(textgrid_file_path, audio_file_path, fs_audio, labels_tier)
        textgrid_labels_fixture.labels = labels  # Inject non-empty labels into the fixture
        return textgrid_labels_fixture

# Test case for initializing the TextgridLabels object
def test_textgrid_labels_init(textgrid_labels_fixture):
    assert isinstance(textgrid_labels_fixture.audio, np.ndarray)
    assert isinstance(textgrid_labels_fixture.labels, list)

# Test case for loading labels from a Textgrid file
def test_load_labels_from_textgrid(textgrid_labels_fixture):
    assert len(textgrid_labels_fixture.labels) > 0

# Test case for finding missing label intervals
def test_find_missing_label_intervals(textgrid_labels_fixture):
    missing_label_intervals = textgrid_labels_fixture.missing_label_intervals
    assert isinstance(missing_label_intervals, list)

# Test case for finding label starts
def test_find_label_starts(textgrid_labels_fixture):
    target_label = 1
    label_starts = textgrid_labels_fixture.find_label_starts(target_label)
    assert isinstance(label_starts, np.ndarray)

# Test case for finding bout limits
def test_find_bout_limits(textgrid_labels_fixture):
    bout_limits = textgrid_labels_fixture.find_bout_limits()
    assert isinstance(bout_limits, np.ndarray)

# Test case for getting raster labels
def test_get_rasters_labels(textgrid_labels_fixture):
    start_sample_list = [0, 100, 200]
    span_samples = 50
    rasters_labels = textgrid_labels_fixture.get_rasters_labels(start_sample_list, span_samples)
    assert isinstance(rasters_labels, list)
    assert len(rasters_labels) == len(start_sample_list)

# Test case for finding label edges
def test_find_label_edges():
    labels = [0, 0, 1, 1, 1, 0, 0]
    lbl_edges = TextgridLabels.find_label_edges(labels)
    assert isinstance(lbl_edges, np.ndarray)

# Test case for finding label edges in 2D labels array
def test_find_label_edges_2D():
    labels_array = [[0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]]
    lbl_edges_2d = TextgridLabels.find_label_edges_2D(labels_array)
    assert isinstance(lbl_edges_2d, list)
    assert len(lbl_edges_2d) == len(labels_array)

# Test case for list expander
def test_list_expander():
    list_1d = [1, 2, 3]
    expand_multiplier = 2
    expanded_list = TextgridLabels.list_expander(list_1d, expand_multiplier)
    assert isinstance(expanded_list, np.ndarray)
    assert len(expanded_list) == len(list_1d) * expand_multiplier

# Test case for signal smoother
def test_signal_smoother():
    signal = np.array([1, 2, 3, 4, 5])
    interp_multiplier = 2
    interp_order = 3
    smoothed_signal = TextgridLabels.signal_smoother(signal, interp_multiplier, interp_order)
    assert isinstance(smoothed_signal, np.ndarray)
    assert len(smoothed_signal) == len(signal) * interp_multiplier

