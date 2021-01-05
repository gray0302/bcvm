# Better Convert Model
This is the code of the paper *Solving the Loss Imbalance Problem in Live Streaming Recommendation with Gradient Normalization*<br/>

Due to business sensitive issues, our data cannot be shared, so only the code and feature logic are shared.<br/>
The comparison method in the paper can be implemented as follows:<br/>
**ESMM:**<br/>
python esmm_train.py --conf esmm_train_conf.ini<br/>
**ESMM+GradNorm:**<br/>
python esmm_train_gn.py --conf esmm_train_conf_gn.ini<br/>
**ESMM+FM:**<br/>
python esmm_train_fm.py --conf esmm_train_conf_fm.ini<br/>
**ESMM+FM+GradNorm:**<br/>
python esmm_train_fm_gn.py --conf esmm_train_conf_fm_gn.ini<br/>
**ESMM+FM+MMOE:**<br/>
python esmm_train_mmoe_fm.py --conf esmm_train_conf_mmoe_fm.ini<br/>
**ESMM+FM+MMOE+GradNorm:**<br/>
python esmm_train_mmoe_fm_gn.py --conf esmm_train_conf_mmoe_fm_gn.ini<br/>
