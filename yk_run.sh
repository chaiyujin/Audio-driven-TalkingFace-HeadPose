source nohup_run.sh
source yk_functions.sh

# m001_trump
RUN_YK_EXP --epoch_a2e=50 --test --media_list=../../media_list.txt \
           --data_src=celebtalk --speaker=m001_trump --use_seqs="trn-000,trn-001,vld-000,vld-001" ;

# f000_watson
RUN_YK_EXP --epoch_a2e=50 --test --media_list=../../media_list.txt \
           --data_src=celebtalk --speaker=f000_watson --use_seqs="trn-000,trn-001,vld-000,vld-001" ;
