"""CLI wiring keeps the synthetic label gate opt-in and visible in output."""
from argparse import Namespace
from unittest.mock import patch
from cviaf.lab.cli import _cmd_compare


def test_cli_compare_passes_frozen_gate(tmp_path):
    protocol = tmp_path / 'protocol.json'
    protocol.write_text('{}')
    args = Namespace(corpus='runs/mvp', alpha=.05, seed=1, backgrounds=8,
                     max_models=None, json=None, label_gate_protocol=str(protocol))
    result = {'model_axis': {'scores': {'cviaf': {'tpr': None}}}, 'data_axis': {'scores': {}},
              'provenance_axis': {}, 'capability_matrix': {}, 'verdict': 'review'}
    with patch('cviaf.lab.label_gate_protocol.load_synthetic', return_value=(object(), 'digest')) as loader, \
         patch('cviaf.lab.compare.compare_corpus', return_value=result) as compare, \
         patch('cviaf.lab.compare.render_table', return_value='table'):
        assert _cmd_compare(args) == 0
    loader.assert_called_once_with(str(protocol), 'runs/mvp', .05)
    assert compare.call_args.kwargs['label_gate_protocol_sha256'] == 'digest'
    assert compare.call_args.kwargs['label_gate'] is not None
