source yk_scripts/nohup_run.sh

# * ---------------------------------------------------------------------------------------------------------------- * #
# *              Our data: VOCASET pre-train general model -> finetune with data tracked by our tracker              * #
# * ---------------------------------------------------------------------------------------------------------------- * #

# ? Suggest training args: --epoch_a2e=40
# ? Suggest testing  args: --epoch_a2e=20 --test --media_list=../../media_list.txt

source yk_scripts/flame/functions.sh
# NOHUP_RUN --device=0 --include=yk_scripts/flame/functions.sh -- \
# > Keep avoffset
# RUN_YK_EXP --data_src=celebtalk --speaker=m001_trump "$@";
# > Manually correct avoffset
# RUN_YK_EXP --data_src=celebtalk --speaker=m001_trump   --avoffset_ms=33.333333333333336 "$@";
# RUN_YK_EXP --data_src=celebtalk --speaker=m000_obama   --avoffset_ms=100.0              "$@";
# RUN_YK_EXP --data_src=celebtalk --speaker=f000_watson  --avoffset_ms=133.33333333333334 "$@";
# RUN_YK_EXP --data_src=celebtalk --speaker=f001_clinton --avoffset_ms=100.0              "$@";

# * ---------------------------------------------------------------------------------------------------------------- * #
# *                  Their pre-trained general model -> finetune with data tracked by their tracker                  * #
# * ---------------------------------------------------------------------------------------------------------------- * #

source yk_scripts/bfm/functions.sh
# NOHUP_RUN --device=0 --include=yk_scripts/bfm/functions.sh -- \
# RUN_YK_EXP --data_src=celebtalk --speaker=m001_trump "$@";
