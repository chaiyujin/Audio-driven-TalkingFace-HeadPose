source yk_functions.sh

# RUN_YK_EXP --data_src=celebtalk --epoch_a2e=2 --epoch_r2v=1 --speaker=m000_obama --debug
# RUN_YK_EXP --data_src=vocaset --epoch_a2e=40 --epoch_r2v=1 --speaker=FaceTalk_170908_03277_TA  --debug
# RUN_YK_EXP --data_src=vocaset --epoch_a2e=50 --epoch_r2v=100 --speaker=FaceTalk_170908_03277_TA

RUN_YK_EXP --data_src=celebtalk --epoch_a2e=50 --speaker=m001_trump --debug
