#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

source yk_scripts/lib.sh

# * Global variables
ERROR='\033[0;31m[ERROR]\033[0m'
CWD=${PWD}

# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                              Functions for Each Step                                             * #
# * ---------------------------------------------------------------------------------------------------------------- * #

# * ------------------------------------------------ Data Preparing ------------------------------------------------ * #

function PrepareData() {
  local DATA_SRC=
  local DATA_DIR=
  local SPEAKER=
  local DEBUG=
  local USE_SEQS=
  # Override from arguments
  for var in "$@"; do
    case $var in
      --data_src=* ) DATA_SRC=${var#*=}  ;;
      --data_dir=* ) DATA_DIR=${var#*=}  ;;
      --speaker=*  ) SPEAKER=${var#*=}   ;;
      --epoch=*    ) EPOCH=${var#*=}     ;;
      --use_seqs=* ) USE_SEQS=${var#*=}  ;;
      --debug      ) DEBUG="--debug"     ;;
    esac
  done

  [ -n "$DATA_SRC" ] || { echo "data_src is not set!"; exit 1; }
  [ -n "$DATA_DIR" ] || { echo "data_dir is not set!"; exit 1; }
  [ -n "$SPEAKER"  ] || { echo "speaker is not set!";  exit 1; }

  # prepare data
  if ! python3 -m yk_scripts.flame.tools "prepare_talk_video" --data_src ${DATA_SRC} --data_dir $DATA_DIR --speaker $SPEAKER ${DEBUG}; then
    printf "${ERROR} Failed to prepare data for ${DATA_SRC}!\n"
    exit 1
  fi
}

function TrainA2E() {
  local DATA_DIR=
  local NET_DIR=
  local EPOCH=
  local SAVE_GAP_EPOCH=
  # Override from arguments
  for var in "$@"; do
    case $var in
      --data_dir=*  ) DATA_DIR=${var#*=}  ;;
      --net_dir=*   ) NET_DIR=${var#*=}   ;;
      --epoch=*     ) EPOCH=${var#*=}     ;;
      --save_gap_epoch=*) SAVE_GAP_EPOCH=${var#*=} ;;
    esac
  done

  [ -n "$DATA_DIR" ] || { echo "data_dir is not set!"; exit 1; }
  [ -n "$NET_DIR"  ] || { echo "net_dir is not set!";  exit 1; }
  [ -n "$EPOCH"    ] || { echo "epoch is not set!";    exit 1; }
  [ -n "$SAVE_GAP_EPOCH" ] || { echo "save_gap_epoch is not set!";    exit 1; }

  # train
  local CKPT_A2C="${NET_DIR}/atcnet/atcnet_lstm_${EPOCH}.pth"
  cd $CWD/Audio/code && \
    RUN_WITH_LOCK_GUARD --tag="Train_Audio2Expresssion" --lock_file="$CKPT_A2C" -- \
    python3 yk_atcnet_flame.py \
      --pose 0 --relativeframe 0 \
      --lr 0.0001 --smooth_loss2 1 \
      --dataset multi_clips \
      --max_epochs ${EPOCH} \
      --save_per_epochs ${SAVE_GAP_EPOCH} \
      --dataset_dir $DATA_DIR \
      --model_dir $NET_DIR/atcnet \
      --device_ids 0 \
    && \
  cd $CWD;
      # --continue_train 1 --model_name ../model/atcnet_lstm_general.pth \
}

function TestA2E() {
  local AUDIO_PATH=
  local NET_DIR=
  local RES_DIR=
  local EPOCH=
  # Override from arguments
  for var in "$@"; do
    case $var in
      --audio_path=*) AUDIO_PATH=${var#*=}  ;;
      --net_dir=*   ) NET_DIR=${var#*=}   ;;
      --res_dir=*   ) RES_DIR=${var#*=}   ;;
      --epoch=*     ) EPOCH=${var#*=}   ;;
    esac
  done

  [ -n "$AUDIO_PATH" ] || { echo "audio_path is not set!"; exit 1; }
  [ -n "$NET_DIR"    ] || { echo "net_dir is not set!";  exit 1; }
  [ -n "$RES_DIR"    ] || { echo "res_dir is not set!";  exit 1; }
  [ -n "$EPOCH"      ] || { echo "epoch is not set!";    exit 1; }

  # predict coefficients from audio
  cd $CWD/Audio/code && \
  python3 yk_atcnet_test.py --para_dim=53 --pose 0 --relativeframe 0 --dataset multi_clips --device_ids 0 \
    --model_name "${NET_DIR}/atcnet/atcnet_lstm_${EPOCH}.pth" \
    --in_file    "${AUDIO_PATH}" \
    --sample_dir "${RES_DIR}/coeff_pred" \
  && \
  cd $CWD
}

# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                                Collection of steps                                               * #
# * ---------------------------------------------------------------------------------------------------------------- * #

function RUN_YK_EXP() {
  local DATA_SRC=
  local SPEAKER=
  local USE_SEQS=
  local EPOCH_A2E=50
  local DEBUG=""
  local TEST=""
  local MEDIA_LIST=""
  local DUMP_MESHES=""
  # Override from arguments
  for var in "$@"; do
    case $var in
      --data_src=*  ) DATA_SRC=${var#*=}  ;;
      --speaker=*   ) SPEAKER=${var#*=}   ;;
      --use_seqs=*  ) USE_SEQS=${var#*=}  ;;
      --epoch_a2e=* ) EPOCH_A2E=${var#*=} ;;
      --test        ) TEST="true"         ;;
      --media_list=*) MEDIA_LIST=${var#*=};;
      --dump_meshes ) DUMP_MESHES="--dump_meshes" ;;
      --debug       ) DEBUG="--debug"     ;;
    esac
  done
  # Check variables
  [ -n "$DATA_SRC"  ] || { echo "data_src is not set!";   exit 1; }
  [ -n "$SPEAKER"   ] || { echo "speaker is not set!";   exit 1; }
  [ -n "$EPOCH_A2E" ] || { echo "epoch_a2e is not set!"; exit 1; }
  # to lower case
  DATA_SRC="${DATA_SRC,,}"

  local VOCA_EXP_DIR="$CWD/yk_exp/flame/vocaset"
  local VOCA_NET_DIR="$VOCA_EXP_DIR/checkpoints"
  local VOCA_DATA_DIR="$VOCA_EXP_DIR/data"

  DRAW_DIVIDER;
  python3 -m yk_scripts.flame.tools "prepare_vocaset" --data_dir ${VOCA_DATA_DIR} ${DEBUG};

  DRAW_DIVIDER;
  TrainA2E --data_dir=${VOCA_DATA_DIR} --net_dir=${VOCA_NET_DIR} --epoch=50 --save_gap_epoch=10;
  exit 1;

  # other variables
  local EXP_DIR="$CWD/yk_exp/flame/$DATA_SRC/$SPEAKER"
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
  local SHARED="--data_src=${DATA_SRC} --data_dir=$DATA_DIR --net_dir=$NET_DIR --speaker=$SPEAKER --use_seqs=$USE_SEQS ${DEBUG}"

  # * Step 1: Prepare data into $DATA_DIR. Reconstructed results saved in $DATA_DIR/../reconstructed
  DRAW_DIVIDER; PrepareData $SHARED

  # # * Step 2: Fintune Audio to Expression Network
  # DRAW_DIVIDER; TrainA2E $SHARED --epoch=$EPOCH_A2E
}
