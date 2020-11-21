#!/bin/sh
export PYTHONPATH=../../../src/Tacotron2/:../../../filelists:$PYTHONPATH

export PATH=/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/akulkarni/new_workspace/anaconda/bin:$PATH

source activate baseline_tts

python -m multiproc train.py --output_directory=outdir --log_directory=logdir
