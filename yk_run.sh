source yk_functions.sh

RUN_YK_EXP \
  --data_src=celebtalk --speaker=m001_trump --use_seqs="trn-000,trn-001,vld-000,vld-001" \
  --epoch_a2e=50  \
  --test  \
  --media_list=../../media_list.txt \
  --dump_meshes \
;
