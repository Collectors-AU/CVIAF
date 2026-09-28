"""Reproduce the Arm C local, small-sample score search. Run: PYTHONPATH=. python cviaf/lab/arm_c_measure.py

The ASR gate is printed. Do not report B* for an attack that fails it.
"""
import json,time
import numpy as np
from cviaf.lab.detector import DetectorConfig
from cviaf.lab.train import TrainSpec,train_model,build_splits
from cviaf.lab.poison import AttackSpec,trigger_view
from cviaf.lab.synth import SceneSpec
from cviaf.lab.detectors import make_backgrounds,trace_ctc,trace_ftc,reference_divergence
from cviaf.lab.evasion import Candidate,successive_halving
s=time.monotonic()
cfg=DetectorConfig(epochs=600, seed=5, pos_weight=12, batch=2048)
scene=SceneSpec(seed=12)
base=dict(scene=scene,detector=cfg,n_train=240,n_eval=80,n_cal=120)
clean=train_model(TrainSpec(model_id='clean',attack=AttackSpec(kind='clean'),**base))
attack=train_model(TrainSpec(model_id='oga',attack=AttackSpec(kind='oga',trigger_size=10,rate=.3,seed=11),**base), null_model=clean.model)
print('TRAINED',time.monotonic()-s, attack.manifest['metrics']['attack_success_rate'],flush=True)
splits=build_splits(TrainSpec(model_id='oga',attack=AttackSpec(kind='oga',trigger_size=10,rate=.3,seed=11),**base))
imgs=splits.eval_clean.images[:60]
trig,_=trigger_view(splits.eval_clean, AttackSpec(kind='oga',trigger_size=10,rate=.3,seed=11), cfg.seed)
trig=trig[:60]
bg=make_backgrounds(3,seed=12)
cc={};aa={}
for name,fn in [('ctc',lambda im: trace_ctc(attack.model,im,bg)['score']),('ftc',lambda im: trace_ftc(attack.model,im,stride=16)['score']),('refdiv',lambda im: reference_divergence(attack.model,clean.model,im)['score'])]:
  cc[name]=fn(imgs);aa[name]=fn(trig)
  print(name,'finite',sum(np.isfinite(cc[name])),sum(np.isfinite(aa[name])),'medians',np.nanmedian(cc[name]),np.nanmedian(aa[name]),flush=True)
result=successive_halving(cc,aa,[Candidate(f,float(p),float(p)) for f in ('dilution','suppression') for p in np.linspace(0,.9,10)],seed=41,initial_count=48,rounds=3,
   attack_qualified=not attack.manifest['quality_flags']['backdoor_weak'])
print("RESULT",json.dumps({name: {"baseline_tpr":v["baseline_tpr"],"threshold":v["threshold"],"status":v["status"],"b_star_grid":v["b_star_grid"],"curve":v["curve"]} for name,v in result["detectors"].items()}),flush=True)
print('DIGEST',result['search_digest'],'elapsed',time.monotonic()-s,flush=True)
