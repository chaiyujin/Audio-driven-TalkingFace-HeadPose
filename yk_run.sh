source yk_functions.sh

RUN_YK_EXP --data_src=celebtalk --epoch_a2e=50 --speaker=m001_trump --use_seqs="trn-000,trn-001,vld-000,vld-001"  # --debug

