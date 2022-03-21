source yk_scripts/nohup_run.sh

# ? Suggest training args: --epoch_a2e=50
# ? Suggest testing  args: --epoch_a2e=50 --test --media_list=../../media_list.txt

# * Our data: VOCASET pre-train general model -> finetune with data tracked by our tracker
source yk_scripts/flame/functions.sh
# NOHUP_RUN --device=0 --include=yk_scripts/flame/functions.sh -- \
# > Keep avoffset
# RUN_YK_EXP --data_src=celebtalk --speaker=m001_trump "$@";
# > Manually correct avoffset
# RUN_YK_EXP --data_src=celebtalk --speaker=m001_trump --avoffset_ms=33.333333333333336 "$@";

# * Their pre-trained general model -> finetune with data tracked by their tracker
source yk_scripts/bfm/functions.sh
# NOHUP_RUN --device=0 --include=yk_scripts/bfm/functions.sh -- \
# RUN_YK_EXP --data_src=celebtalk --speaker=m001_trump "$@";
