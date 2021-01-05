# Better Convert Model
This is the code of the paper *Solving the Loss Imbalance Problem in Live Streaming Recommendation with Gradient Normalization*

Due to business sensitive issues, our data cannot be shared, so only the code and feature logic are shared.
The comparison method in the paper can be implemented as follows:
#ESMM:
python esmm_train.py --conf esmm_train_conf.ini
#ESMM+GradNorm
python esmm_train_gn.py --conf esmm_train_conf_gn.ini
#ESMM+FM:
python esmm_train_fm.py --conf esmm_train_conf_fm.ini
#ESMM+FM+GradNorm
python esmm_train_fm_gn.py --conf esmm_train_conf_fm_gn.ini
#ESMM+FM+MMOE
python esmm_train_mmoe_fm.py --conf esmm_train_conf_mmoe_fm.ini
#ESMM+FM+MMOE+GradNorm
python esmm_train_mmoe_fm_gn.py --conf esmm_train_conf_mmoe_fm_gn.ini
