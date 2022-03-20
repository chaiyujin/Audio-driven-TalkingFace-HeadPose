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

function TrainA2E() {
  local DATA_DIR=
  local NET_DIR=
  local EPOCH=
  local SAVE_GAP_EPOCH=
  local LR=
  local LOAD_FROM=
  # Override from arguments
  for var in "$@"; do
    case $var in
      --data_dir=*      ) DATA_DIR=${var#*=}       ;;
      --net_dir=*       ) NET_DIR=${var#*=}        ;;
      --epoch=*         ) EPOCH=${var#*=}          ;;
      --save_gap_epoch=*) SAVE_GAP_EPOCH=${var#*=} ;;
      --lr=*            ) LR=${var#*=}             ;;
      --load_from=*     ) LOAD_FROM=${var#*=}      ;;
    esac
  done

  [ -n "$DATA_DIR"       ] || { echo "data_dir is not set!";       exit 1; }
  [ -n "$NET_DIR"        ] || { echo "net_dir is not set!";        exit 1; }
  [ -n "$EPOCH"          ] || { echo "epoch is not set!";          exit 1; }
  [ -n "$SAVE_GAP_EPOCH" ] || { echo "save_gap_epoch is not set!"; exit 1; }
  [ -n "$LR"             ] || { echo "lr is not set!";             exit 1; }

  local FT_ARGS=''
  if [ -n "$LOAD_FROM" ]; then
    FT_ARGS=" --continue_train 1 --model_name ${LOAD_FROM}"
  fi

  # train
  local CKPT_A2C="${NET_DIR}/atcnet/atcnet_lstm_${EPOCH}.pth"
  cd $CWD/Audio/code && \
    RUN_WITH_LOCK_GUARD --tag="Train_Audio2Expresssion" --lock_file="$CKPT_A2C" -- \
    python3 yk_atcnet_flame.py \
      --lr ${LR} --smooth_loss2 1 \
      --dataset multi_clips \
      --max_epochs ${EPOCH} \
      --save_per_epochs ${SAVE_GAP_EPOCH} \
      --dataset_dir $DATA_DIR \
      --model_dir $NET_DIR/atcnet \
      --device_ids 0 \
      ${FT_ARGS} \
    && \
  cd $CWD;
}

function TestA2E() {
  local AUDIO_PATH=
  local NET_DIR=
  local RES_DIR=
  local EPOCH=
  # Override from arguments
  for var in "$@"; do
    case $var in
      --audio_path=*) AUDIO_PATH=${var#*=} ;;
      --net_dir=*   ) NET_DIR=${var#*=}    ;;
      --res_dir=*   ) RES_DIR=${var#*=}    ;;
      --epoch=*     ) EPOCH=${var#*=}      ;;
    esac
  done

  [ -n "$AUDIO_PATH" ] || { echo "audio_path is not set!"; exit 1; }
  [ -n "$NET_DIR"    ] || { echo "net_dir is not set!";    exit 1; }
  [ -n "$RES_DIR"    ] || { echo "res_dir is not set!";    exit 1; }
  [ -n "$EPOCH"      ] || { echo "epoch is not set!";      exit 1; }

  # * make sure we get audio file.
  if [ ! -f "${RES_DIR}/audio.wav" ]; then
    mkdir -p ${RES_DIR};
    ffmpeg -loglevel error -i "${AUDIO_PATH}" "${RES_DIR}/audio.wav" -n;
  fi

  # * predict coefficients from audio
  cd $CWD/Audio/code && \
  python3 yk_atcnet_test.py --para_dim=53 --pose 0 --relativeframe 0 --dataset multi_clips --device_ids 0 \
    --model_name "${NET_DIR}/atcnet/atcnet_lstm_${EPOCH}.pth" \
    --in_file    "${RES_DIR}/audio.wav" \
    --sample_dir "${RES_DIR}/coeff_pred" \
  && cd $CWD
}

# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                                Collection of steps                                               * #
# * ---------------------------------------------------------------------------------------------------------------- * #

function PretrainGeneralOnVocaset() {
  local EPOCH=$1
  shift 1;

  local VOCA_EXP_DIR="$CWD/yk_exp/flame/vocaset"
  local VOCA_NET_DIR="$VOCA_EXP_DIR/checkpoints"
  local VOCA_DATA_DIR="$VOCA_EXP_DIR/data"

  DRAW_DIVIDER;
  python3 -m yk_scripts.flame.tools "prepare_vocaset" --data_dir ${VOCA_DATA_DIR} ${DEBUG};

  DRAW_DIVIDER;
  TrainA2E --data_dir=${VOCA_DATA_DIR} --net_dir=${VOCA_NET_DIR} --epoch=${EPOCH} --save_gap_epoch=${EPOCH} --lr=0.0002;
}

function RUN_YK_EXP() {
  local DATA_SRC=
  local SPEAKER=
  local USE_SEQS=
  local EPOCH_A2E=50
  local DEBUG=""
  local TEST=""
  local MEDIA_LIST=""
  # Override from arguments
  for var in "$@"; do
    case $var in
      --data_src=*  ) DATA_SRC=${var#*=}  ;;
      --speaker=*   ) SPEAKER=${var#*=}   ;;
      --use_seqs=*  ) USE_SEQS=${var#*=}  ;;
      --epoch_a2e=* ) EPOCH_A2E=${var#*=} ;;
      --test        ) TEST="true"         ;;
      --media_list=*) MEDIA_LIST=${var#*=};;
      --debug       ) DEBUG="--debug"     ;;
    esac
  done
  # Check variables
  [ -n "$DATA_SRC"  ] || { echo "data_src is not set!";   exit 1; }
  [ -n "$SPEAKER"   ] || { echo "speaker is not set!";   exit 1; }
  [ -n "$EPOCH_A2E" ] || { echo "epoch_a2e is not set!"; exit 1; }
  # to lower case
  DATA_SRC="${DATA_SRC,,}"

  # * Make sure VOCASET is used to pre-train a general model
  PretrainGeneralOnVocaset 100;
  local CKPT_GENERAL="$CWD/yk_exp/flame/vocaset/checkpoints/atcnet/atcnet_lstm_100.pth";

  # * Speaker specific model
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

  # Shared arguments
  local SHARED="--data_src=${DATA_SRC} --data_dir=$DATA_DIR --net_dir=$NET_DIR --speaker=$SPEAKER --use_seqs=$USE_SEQS ${DEBUG}"

  # * Step 1: Prepare data into $DATA_DIR. Reconstructed results saved in $DATA_DIR/../reconstructed
  DRAW_DIVIDER;
  python3 -m yk_scripts.flame.tools "prepare_talk_video" \
    --data_src ${DATA_SRC} \
    --speaker  ${SPEAKER}  \
    --data_dir ${DATA_DIR} \
    ${DEBUG};

  # * Step 2: Fintune Audio to Expression Network
  DRAW_DIVIDER;
  TrainA2E \
    --data_dir=${DATA_DIR} --net_dir=${NET_DIR} \
    --epoch=${EPOCH_A2E} --save_gap_epoch=10 \
    --lr=0.0001 --load_from="${CKPT_GENERAL}" \
  ;
  
  # * Step 3: Test the trained model
  if [ -n "${TEST}" ]; then
    local test_tasks=()
    local fpath=''
    local seq_id=''
    # Get dataset sequences
    for d in $DATA_DIR/**/*; do
      if [ ! -d "$d" ]; then continue; fi
      if [[ "$d" =~ clip-.* ]]; then
        fpath=$(realpath $d/audio/audio.wav)
        seq_id=$(basename $d)
        test_tasks+=("${fpath}|${seq_id}")
      fi
    done
    # Generate from media_list
    if [ -n "${MEDIA_LIST}" ]; then
      for task in $(LoadMediaList ${MEDIA_LIST}); do
        test_tasks+=($task)
      done
    fi

    local test_args="--net_dir=${NET_DIR} --epoch=${EPOCH_A2E}"
    for task in ${test_tasks[@]}; do
      IFS='|' read -ra SS <<< "$task"
      local fpath="${SS[0]}"
      local seq_id="${SS[1]}"
      TestA2E ${test_args} \
              --audio_path=${fpath} \
              --res_dir="${RES_DIR}/${seq_id}";
    done

    # copy idenity file for rendering
    cp ${DATA_DIR}/identity/identity.obj ${RES_DIR}/identity.obj;

    python3 -m yk_scripts.flame.render --exp_dir ${RES_DIR} --fps=25
  fi
}
