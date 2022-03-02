#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

# * Global variables
ERROR='\033[0;31m[ERROR]\033[0m'
CWD=${PWD}

# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                                  Tool Functions                                                  * #
# * ---------------------------------------------------------------------------------------------------------------- * #

function DRAW_DIVIDER() {
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" ' ' | tr ' ' '-'
}

function RUN_WITH_LOCK_GUARD() {
  local _end=
  local lock_file=
  local tag=
  local debug=
  local cmd=
  for i in "$@"; do
    if [ -z "${_end}" ]; then
      if [ "$i" = "--" ]; then
        _end=true
        continue
      fi
      case $i in
        -l=* | --lock_file=* ) lock_file=${i#*=}  ;;
        -t=* | --tag=*       ) tag=${i#*=}        ;;
        -d   | --debug       ) debug=true         ;;
        *) echo "[LockFileGuard]: Wrong argument ${i}"; exit 1;;
      esac
    else
      cmd="$cmd '$i'"
    fi
  done

  # lock_file must be given
  if [ -z "$lock_file" ]; then
    printf "lock_file is not set!\n"
    exit 1
  fi

  if [ -f "$lock_file" ]; then
    printf "Command '${tag}' is skipped due to existance of lock_file ${lock_file}\n"
    return
  fi

  # to abspath
  mkdir -p "$(dirname $lock_file)"
  lock_file="$(cd $(dirname $lock_file) && pwd)/$(basename $lock_file)"

  # debug message
  if [ -n "$debug" ]; then
    echo "--------------------------------------------------------------------------------"
    echo "LOCK_FILE: $lock_file"
    echo "COMMAND:   $cmd"
    echo "--------------------------------------------------------------------------------"
  fi

  # Run command in subshell and creat lock file if success
  if (eval "$cmd") ; then
    [ -f "$lock_file" ] || touch "$lock_file"
  else
    echo "Failed to run command '${tag}'"
    exit 1
  fi
}

# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                              Functions for Each Step                                             * #
# * ---------------------------------------------------------------------------------------------------------------- * #

# * ------------------------------------------------ Data Preparing ------------------------------------------------ * #

function PrepareData() {
  local DATA_SRC=
  local DATA_DIR=
  local SPEAKER=
  local DEBUG=
  # Override from arguments
  for var in "$@"; do
    case $var in
      --data_src=* ) DATA_SRC=${var#*=}  ;;
      --data_dir=* ) DATA_DIR=${var#*=}  ;;
      --speaker=*  ) SPEAKER=${var#*=}   ;;
      --epoch=*    ) EPOCH=${var#*=}     ;;
      --debug      ) DEBUG="--debug"     ;;
    esac
  done

  [ -n "$DATA_SRC" ] || { echo "data_src is not set!"; exit 1; }
  [ -n "$DATA_DIR" ] || { echo "data_dir is not set!"; exit 1; }
  [ -n "$SPEAKER"  ] || { echo "speaker is not set!";  exit 1; }

  # prepare data
  if ! python3 yk_tools.py "prepare_${DATA_SRC}" --data_dir $DATA_DIR --speaker $SPEAKER ${DEBUG}; then
    printf "${ERROR} Failed to prepare data for ${DATA_SRC}!\n"
    exit 1
  fi

  cd $CWD/Deep3DFaceReconstruction && \
    RUN_WITH_LOCK_GUARD --tag="reconstruct" --lock_file="$DATA_DIR/../done_recons.lock" -- \
    python3 yk_reconstruct.py $DATA_DIR $DATA_DIR/../reconstructed && \
  cd $CWD;
}

# * -------------------------------------------- Train Audio2Expression -------------------------------------------- * #

function TrainA2E() {
  local DATA_DIR=
  local NET_DIR=
  local SPEAKER=
  local EPOCH=
  # Override from arguments
  for var in "$@"; do
    case $var in
      --data_dir=*  ) DATA_DIR=${var#*=}  ;;
      --net_dir=*   ) NET_DIR=${var#*=}   ;;
      --speaker=*   ) SPEAKER=${var#*=}   ;;
      --epoch=*     ) EPOCH=${var#*=}     ;;
    esac
  done

  [ -n "$DATA_DIR" ] || { echo "data_dir is not set!"; exit 1; }
  [ -n "$NET_DIR"  ] || { echo "net_dir is not set!";  exit 1; }
  [ -n "$SPEAKER"  ] || { echo "speaker is not set!";  exit 1; }
  [ -n "$EPOCH"    ] || { echo "epoch is not set!";    exit 1; }

  # train
  local CKPT_A2C="${NET_DIR}/atcnet/atcnet_lstm_${EPOCH}.pth"
  cd $CWD/Audio/code && \
    RUN_WITH_LOCK_GUARD --tag="Train_Audio2Expresssion" --lock_file="$CKPT_A2C" -- \
    python3 yk_atcnet.py \
      --pose 1 --relativeframe 0 \
      --lr 0.0001 --less_constrain 1 --smooth_loss 1 --smooth_loss2 1 \
      --continue_train 1 --model_name ../model/atcnet_lstm_general.pth \
      --dataset multi_clips \
      --max_epochs ${EPOCH} \
      --save_per_epochs ${EPOCH} \
      --dataset_dir $DATA_DIR \
      --model_dir $NET_DIR/atcnet \
      --device_ids 0 \
    && \
  cd $CWD;
}

# * --------------------------------------------- Train Render to Video -------------------------------------------- * #

function TrainR2V() {
  local DATA_DIR=
  local NET_DIR=
  local SPEAKER=
  local EPOCH=
  local DEBUG=
  # Override from arguments
  for var in "$@"; do
    case $var in
      --data_dir=*  ) DATA_DIR=${var#*=}  ;;
      --net_dir=*   ) NET_DIR=${var#*=}   ;;
      --speaker=*   ) SPEAKER=${var#*=}   ;;
      --epoch=*     ) EPOCH=${var#*=}     ;;
      --debug       ) DEBUG="--debug"     ;;
    esac
  done

  [ -n "$DATA_DIR" ] || { echo "data_dir is not set!"; exit 1; }
  [ -n "$NET_DIR"  ] || { echo "net_dir is not set!";  exit 1; }
  [ -n "$SPEAKER"  ] || { echo "speaker is not set!";  exit 1; }
  [ -n "$EPOCH"    ] || { echo "epoch is not set!";    exit 1; }

  # prepare data
  if ! (python3 yk_tools.py build_r2v_dataset \
          --data_dir  $DATA_DIR/../reconstructed \
          --data_type train ${DEBUG}) ; then
    printf "${ERROR} Failed to prepare data for render-to-video!\n"
    exit 1
  fi

  # prepare arcface feature
  local DONE_FLAG_ARCFACE=${DATA_DIR}/../done_arcface.flag
  if [ -f "${DONE_FLAG_ARCFACE}" ]; then
    printf "Arcface is already runned\n"
  else
    if ! (cd ${CWD}/render-to-video/arcface && python3 yk_test_batch.py \
      --imglist ${DATA_DIR}/../r2v_dataset/list/trainB/bmold.txt --gpu 0) ; then
      printf "${ERROR} Failed to prepare arcface feature for render-to-video!\n"
      exit 1
    else
      touch ${DONE_FLAG_ARCFACE};
    fi
  fi

  # fine tune the mapping
  local DONE_FLAG_R2V=${NET_DIR}/r2v/memory_seq_p2p/${EPOCH}_net_G.pth;
  if [ -f "${DONE_FLAG_R2V}" ]; then
    printf "Render to Video network is already trained: '$DONE_FLAG_R2V'\n"
  else
    if [ ! -f ${NET_DIR}/r2v/memory_seq_p2p/0_net_mem.pth ]; then
      mkdir -p ${NET_DIR}/r2v/memory_seq_p2p &&
      ln -s ${CWD}/render-to-video/checkpoints/memory_seq_p2p/0_net_G.pth   ${NET_DIR}/r2v/memory_seq_p2p/0_net_G.pth &&
      ln -s ${CWD}/render-to-video/checkpoints/memory_seq_p2p/0_net_D.pth   ${NET_DIR}/r2v/memory_seq_p2p/0_net_D.pth &&
      ln -s ${CWD}/render-to-video/checkpoints/memory_seq_p2p/0_net_mem.pth ${NET_DIR}/r2v/memory_seq_p2p/0_net_mem.pth;
    fi
    if ! (cd ${CWD}/render-to-video && \
    python3 train.py \
      --model yk_memory_seq \
      --name memory_seq_p2p \
      --display_env memory_seq_${SPEAKER} \
      --continue_train \
      --epoch 0 \
      --epoch_count 1 \
      --lambda_mask 2 \
      --lr 0.0001 \
      --gpu_ids 0 \
      --dataroot ${DATA_DIR}/../r2v_dataset \
      --dataname bmold_win3 \
      --niter_decay 0 \
      --niter           ${EPOCH} \
      --save_epoch_freq 20 \
      --checkpoints_dir ${NET_DIR}/r2v \
    && cd ${CWD}) ; then
      printf "${ERROR} Failed to train render-to-video network!\n"
      exit 1
    fi

    # rm latest_* to save disk space
    if [ -f "${NET_DIR}/r2v/memory_seq_p2p/latest_net_G.pth" ]; then
      rm "${NET_DIR}/r2v/memory_seq_p2p/latest_net_mem.pth"
      rm "${NET_DIR}/r2v/memory_seq_p2p/latest_net_D.pth"
      rm "${NET_DIR}/r2v/memory_seq_p2p/latest_net_G.pth"
    fi
  fi 
}

# * --------------------------------------------------- Run test --------------------------------------------------- * #

function TestClip() {
  local SRC_DIR=
  local TGT_DIR=
  local RES_DIR=
  local NET_DIR=
  local EPOCH_A2E=
  local EPOCH_R2V=
  local DEBUG=
  # Override from arguments
  for var in "$@"; do
    case $var in
      --src_audio_dir=* ) SRC_DIR=${var#*=}   ;;
      --tgt_video_dir=* ) TGT_DIR=${var#*=}   ;;
      --result_dir=*    ) RES_DIR=${var#*=}   ;;
      --net_dir=*       ) NET_DIR=${var#*=}   ;;
      --epoch_a2e=*     ) EPOCH_A2E=${var#*=} ;;
      --epoch_r2v=*     ) EPOCH_R2V=${var#*=} ;;
      --debug           ) DEBUG="--debug"  ;;
    esac
  done
  # check
  [ -n "$SRC_DIR"   ] || { echo "src_audio_dir is not set!"; exit 1; }
  [ -n "$TGT_DIR"   ] || { echo "tgt_video_dir is not set!"; exit 1; }
  [ -n "$RES_DIR"   ] || { echo "result_dir is not set!";    exit 1; }
  [ -n "$EPOCH_A2E" ] || { echo "epoch_a2e is not set!";     exit 1; }

  local CKPT_A2C="${NET_DIR}/atcnet/atcnet_lstm_${EPOCH_A2E}.pth"
  local AUDIO_PATH="$SRC_DIR/audio/audio.wav"

  printf "Running test for '${SRC_DIR}', reenacting video '${TGT_DIR}'\n"

  # predict coefficients from audio
  cd $CWD/Audio/code && \
  python3 yk_atcnet_test.py --pose 1 --relativeframe 0 --dataset multi_clips --device_ids 0 \
    --model_name ${CKPT_A2C} \
    --in_file    ${AUDIO_PATH} \
    --sample_dir ${RES_DIR}/coeff_pred \
  && \
  cd $CWD

  # reenact with predicted coefficients
  cd $CWD/Deep3DFaceReconstruction && \
    RUN_WITH_LOCK_GUARD --tag="reenact" --lock_file="${RES_DIR}/done_reenact.lock" -- \
    python3 yk_reenact.py ${RES_DIR} ${TGT_DIR} && \
  cd $CWD

  local vpath_render="$RES_DIR-render.mp4"
  if [ ! -f "$vpath_render" ]; then
    mkdir -p "$(dirname $vpath_render)"
    ffmpeg -y -loglevel error \
      -thread_queue_size 8192 -i $RES_DIR/reenact/render/frame%d.png \
      -thread_queue_size 8192 -i $RES_DIR/reenact/render/frame%d_render.png \
      -thread_queue_size 8192 -i $AUDIO_PATH \
      -filter_complex hstack=inputs=2 -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p -shortest \
      $vpath_render \
    ;
  fi

  # * Return if don't have R2V
  if [ -z "${EPOCH_R2V}" ]; then
    return
  fi

  # prepare test data
  python3 yk_tools.py build_r2v_dataset \
    --data_dir  $RES_DIR \
    --data_type test \
    ${DEBUG} \
  ;

  # prepare arcface feature
  local arcface_flag=${RES_DIR}/done_arcface.flag
  if [ -f "${arcface_flag}" ]; then
    printf "Arcface is already runned\n"
  else
    if ! (cd ${CWD}/render-to-video/arcface && python3 yk_test_batch.py \
      --imglist ${RES_DIR}/r2v_dataset/list/testB/bmold.txt --gpu 0 \
    && cd ${CWD}) ; then
      printf "${ERROR} Failed to prepare arcface feature for render-to-video!\n"
      exit 1
    fi
    touch ${arcface_flag};
  fi

  local r2v_flag=${RES_DIR}/done_r2v.flag
  if [ -f "${r2v_flag}" ]; then
    printf "Render-to-Video is already runned\n"
  else
    if ! (cd ${CWD}/render-to-video && \
    python3 yk_test.py \
      --model yk_memory_seq \
      --name memory_seq_p2p \
      --num_test 200 \
      --imagefolder '' \
      --epoch ${EPOCH_R2V} \
      --dataroot ${RES_DIR}/r2v_dataset \
      --dataname bmold \
      --checkpoints_dir ${NET_DIR}/r2v \
      --results_dir ${RES_DIR}/reenact/r2v \
      --gpu_ids 0 \
    && cd ${CWD}) ; then
      printf "${ERROR} Failed to run render-to-video!\n"
      exit 1
    fi
    touch ${r2v_flag};
  fi

  python3 utils/blend_results.py \
    --image_dir  ${TGT_DIR}/crop \
    --coeff_dir  ${RES_DIR}/reenact/coeff \
    --r2v_dir    ${RES_DIR}/reenact/r2v \
    --output_dir ${RES_DIR}/results \
  ;

  local vpath_render="$RES_DIR-r2v.mp4"
  if [ ! -f "$vpath_render" ]; then
    mkdir -p "$(dirname $vpath_render)"
    ffmpeg -y -loglevel error \
      -thread_queue_size 8192 -i ${RES_DIR}/results/%06d_real.png \
      -thread_queue_size 8192 -i ${RES_DIR}/results/%06d_fake.png \
      -thread_queue_size 8192 -i $AUDIO_PATH \
      -filter_complex hstack=inputs=2 -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p -shortest \
      $vpath_render \
    ;
  fi
}

# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                                Collection of steps                                               * #
# * ---------------------------------------------------------------------------------------------------------------- * #

function RUN_YK_EXP() {
  local DATA_SRC=
  local SPEAKER=
  local EPOCH_A2E=
  local EPOCH_R2V=
  local DEBUG=""
  # Override from arguments
  for var in "$@"; do
    case $var in
      --data_src=*  ) DATA_SRC=${var#*=}  ;;
      --speaker=*   ) SPEAKER=${var#*=}   ;;
      --epoch_a2e=* ) EPOCH_A2E=${var#*=} ;;
      --epoch_r2v=* ) EPOCH_R2V=${var#*=} ;;
      --debug       ) DEBUG="--debug"     ;;
    esac
  done
  # Check variables
  [ -n "$DATA_SRC"  ] || { echo "data_src is not set!";   exit 1; }
  [ -n "$SPEAKER"   ] || { echo "speaker is not set!";   exit 1; }
  [ -n "$EPOCH_A2E" ] || { echo "epoch_a2e is not set!"; exit 1; }
  # to lower case
  DATA_SRC="${DATA_SRC,,}"

  # other variables
  local EXP_DIR="$CWD/yk_exp/$DATA_SRC/$SPEAKER"
  local NET_DIR="$EXP_DIR/checkpoints"
  local RES_DIR="$EXP_DIR/results"
  local DATA_DIR="$EXP_DIR/data"

  # Print arguments
  DRAW_DIVIDER;
  printf "Speaker   : $SPEAKER\n"
  printf "Data dir  : $DATA_DIR\n"
  printf "Ckpt dir  : $NET_DIR\n"
  printf "Results   : $RES_DIR\n"
  printf "Epoch A2E : $EPOCH_A2E\n"
  printf "Epoch R2V : $EPOCH_R2V\n"

  # Shared arguments
  local SHARED="--data_src=${DATA_SRC} --data_dir=$DATA_DIR --net_dir=$NET_DIR --speaker=$SPEAKER ${DEBUG}"

  # * Step 1: Prepare data into $DATA_DIR. Reconstructed results saved in $DATA_DIR/../reconstructed
  DRAW_DIVIDER; PrepareData $SHARED

  # * Step 2: Train Audio to Expression Network
  DRAW_DIVIDER; TrainA2E $SHARED --epoch=$EPOCH_A2E

  # * Step 3: (Optional) Train neural renderer
  if [ -n "${EPOCH_R2V}" ]; then
    # * The dataset is genrated from $DATA_DIR/../reconstructed and write int $DATA_DIR/../r2v_dataset
    DRAW_DIVIDER; TrainR2V ${SHARED} --epoch=$EPOCH_R2V
  fi

  # * Step testing
  DRAW_DIVIDER;

  for d in "$DATA_DIR/test"/*; do
    if [ ! -d "$d" ]; then continue; fi
    local clip_id="$(basename $d)"

    TestClip \
      --src_audio_dir="$d" \
      --tgt_video_dir="$d" \
      --result_dir="$RES_DIR/self-reenact/$clip_id" \
      --net_dir="$NET_DIR" \
      --epoch_a2e="$EPOCH_A2E" \
      --epoch_r2v="$EPOCH_R2V" \
    ;
  done
}