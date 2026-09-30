import numpy as np
from cviaf.lab.patch_local import patch_local_scores
from cviaf.lab.synth import build_dataset, SceneSpec
from cviaf.lab.poison import apply_trigger, AttackSpec

def test_layout_and_dtype():
    x=build_dataset(4,SceneSpec(),seed_offset=7).images
    np.testing.assert_allclose(patch_local_scores(x),patch_local_scores(x.transpose(0,3,1,2)))
    assert np.all(np.isfinite(patch_local_scores((x*255).astype(np.uint8))))

def test_placement():
    ds=build_dataset(1,SceneSpec(),seed_offset=8)
    base=patch_local_scores(ds.images)[0]
    for loc in ('fixed','random','on_object'):
        img,_=apply_trigger(ds.images[0],AttackSpec(kind='clean_label',trigger='patch',trigger_loc=loc,trigger_size=10,seed=9),np.random.default_rng(2),ds.boxes[0][0])
        assert patch_local_scores(img[None])[0]>base

def test_fail_closed():
    x=np.zeros((1,3,3,3),np.float32)
    try: patch_local_scores(x)
    except ValueError: pass
    else: raise AssertionError('must reject too small input')

if __name__ == '__main__':
    test_layout_and_dtype();test_placement();test_fail_closed();print('3 smoke tests passed')

def test_missing_reference_abstains_without_shape_error():
    from cviaf.lab.compare import _cviaf_item_flags
    from cviaf.lab.baseline import cviaf_data_asset_verdict
    f = _cviaf_item_flags({}, .05, n_items=240)
    assert f.shape == (240,) and not f.any()
    assert cviaf_data_asset_verdict('shifted', {}).abstained


def test_global_chroma_blended_signal():
    from cviaf.lab.trigger_global import global_chroma_scores
    from cviaf.lab.poison import inject
    x = build_dataset(64, SceneSpec(), seed_offset=41)
    y, truth = inject(x, AttackSpec(kind="oda", trigger="blended",
                                   trigger_loc="on_object", rate=.5, seed=5))
    scores = global_chroma_scores(y.images)
    assert np.median(scores[truth.poisoned_indices]) > np.median(
        np.delete(scores, truth.poisoned_indices))


def test_manifest_sensor_stratum_abstains():
    from cviaf.lab.compare import data_axis
    from cviaf.lab.train import TrainSpec
    from copy import deepcopy
    scene = SceneSpec()
    spec = TrainSpec(model_id="clean", scene=scene, n_train=24)
    manifest = {"spec": spec.to_dict(), "ground_truth": {"kind": "clean"},
                "model_id": "clean"}
    shifted = deepcopy(manifest)
    shifted["spec"]["scene"]["sensor_noise"] = .05
    shifted["model_id"] = shifted["spec"]["model_id"] = "shifted"
    rows = data_axis([{"manifest": manifest}, {"manifest": shifted}],
                     log=lambda _: None)
    assert rows[1]["cviaf"]["decision"] == "abstain"
    assert rows[1]["cviaf_item"]["n_flagged"] == 0
    assert rows[1]["calibration_stratum_match"] is False
