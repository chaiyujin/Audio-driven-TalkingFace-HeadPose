function DRAW_DIVIDER() {
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" ' ' | tr ' ' '-'
}

function RUN_VOCASET() {
  ERROR='\033[0;31m[ERROR]\033[0m'

  local CWD=${PWD}
  local SPEAKER=FaceTalk_170908_03277_TA
  local DATA_DIR=${PWD}/data/vocaset
  local EPOCH_A2C=100
  local EPOCH_R2V=20
  local DEBUG=""

  # Override from arguments
  for var in "$@"
  do
    if [[ $var =~ --speaker\=(.*) ]]; then
      SPEAKER="${BASH_REMATCH[1]}"
    elif [[ $var =~ --epoch_a2c\=(.*) ]]; then
      EPOCH_A2C="${BASH_REMATCH[1]}"
    elif [[ $var =~ --epoch_r2v\=(.*) ]]; then
      EPOCH_R2V="${BASH_REMATCH[1]}"
    elif [[ $var =~ --debug ]]; then
      DEBUG="--debug"
    fi
  done

  # Check variables
  # - Check speaker is FaceTalk
  if [[ ! ${SPEAKER} =~ FaceTalk_.* ]]; then
    printf "${ERROR} SPEAKER=${SPEAKER}, is not one from VOCASET!\n"
    exit 1
  fi

  # The checkpoints directory for this speaker
  local NET_DIR="$DATA_DIR/$SPEAKER/checkpoints"

  # Print arguments
  DRAW_DIVIDER;
  printf "Speaker        : $SPEAKER\n"
  printf "Epoch for A2C  : $EPOCH_A2C\n"
  printf "Data directory : $DATA_DIR\n"
  printf "Checkpoints    : $NET_DIR\n"

  # *------------------------------------------------ Data Preparing ------------------------------------------------* #
  DRAW_DIVIDER;

  # prepare data
  python3 utils/tools_vocaset_video.py prepare --dataset_dir $DATA_DIR --speakers $SPEAKER ${DEBUG}
  if [[ $? != 0 ]]; then
    printf "${ERROR} Failed to prepare data!\n"
    exit 1
  fi

  # *------------------------------------------------ Reconstruct 3D ------------------------------------------------* #
  DRAW_DIVIDER

  cd $CWD/Deep3DFaceReconstruction && python3 demo_vocaset.py $DATA_DIR/$SPEAKER && cd $CWD
  if [[ $? != 0 ]]; then
    printf "${ERROR} Failed to reconstruct 3D!\n"
    exit 1
  fi

  # *------------------------------------------- Train Audio to Expression ------------------------------------------* #
  DRAW_DIVIDER

  local A2C_DONE_FLAG="${NET_DIR}/atcnet/atcnet_lstm_${EPOCH_A2C}.pth"
  if [ -f "$A2C_DONE_FLAG" ]; then
    printf "Audio to Expression network is already trained: '$A2C_DONE_FLAG'\n"
  else
    cd $CWD/Audio/code && \
      python3 yk_atcnet.py \
        --pose 1 --relativeframe 0 \
        --lr 0.0001 --less_constrain 1 --smooth_loss 1 --smooth_loss2 1 \
        --continue_train 1 --model_name ../model/atcnet_lstm_general.pth \
        --dataset multi_clips \
        --max_epochs ${EPOCH_A2C} \
        --save_per_epochs ${EPOCH_A2C} \
        --dataset_dir $DATA_DIR/$SPEAKER \
        --model_dir $NET_DIR/atcnet \
        --device_ids 0 \
      && \
    cd $CWD
    if [[ $? != 0 ]]; then
      printf "${ERROR} Failed to train audio to expression network!\n"
      exit 1
    fi
  fi

  # *--------------------------------------------- Train Render to Video --------------------------------------------* #
  DRAW_DIVIDER
  # prepare data
  python3 utils/tools_vocaset_video.py build_r2v_dataset \
    --dataset_dir $DATA_DIR \
    --speakers $SPEAKER \
    --data_type train \
    ${DEBUG} \
  ;
  if [[ $? != 0 ]]; then
    printf "${ERROR} Failed to prepare data for render-to-video!\n"
    exit 1
  fi

  DRAW_DIVIDER
  # prepare arcface feature
  local DONE_FLAG_ARCFACE=${DATA_DIR}/${SPEAKER}/done_arcface_train.flag
  if [ -f "${DONE_FLAG_ARCFACE}" ]; then
    printf "Arcface is already runned\n"
  else
    cd ${CWD}/render-to-video/arcface && python3 yk_test_batch.py \
      --imglist ${DATA_DIR}/${SPEAKER}/r2v_dataset/list/trainB/${SPEAKER}_bmold.txt --gpu 0
    if [[ $? != 0 ]]; then
      printf "${ERROR} Failed to prepare arcface feature for render-to-video!\n"
      exit 1
    fi
    touch ${DONE_FLAG_ARCFACE};
  fi

  DRAW_DIVIDER
  # fine tune the mapping
  local DONE_FLAG_R2V=${NET_DIR}/r2v/memory_seq_p2p/$SPEAKER/${EPOCH_R2V}_net_G.pth;
  if [ -f "${DONE_FLAG_R2V}" ]; then
    printf "Render to Video network is already trained: '$DONE_FLAG_R2V'\n"
  else
    exit
    if [ ! -f ${NET_DIR}/r2v/memory_seq_p2p/0_net_mem.pth ]; then
      ln -s ${CWD}/render-to-video/checkpoints/memory_seq_p2p/0_net_G.pth ${NET_DIR}/r2v/memory_seq_p2p/0_net_G.pth &&
      ln -s ${CWD}/render-to-video/checkpoints/memory_seq_p2p/0_net_D.pth ${NET_DIR}/r2v/memory_seq_p2p/0_net_D.pth &&
      ln -s ${CWD}/render-to-video/checkpoints/memory_seq_p2p/0_net_mem.pth ${NET_DIR}/r2v/memory_seq_p2p/0_net_mem.pth;
    fi
    cd ${CWD}/render-to-video && \
    python3 train.py \
      --model yk_memory_seq \
      --name memory_seq_p2p/${SPEAKER} \
      --display_env memory_seq_${SPEAKER} \
      --continue_train \
      --epoch 0 \
      --epoch_count 1 \
      --lambda_mask 2 \
      --lr 0.0001 \
      --gpu_ids 0 \
      --dataroot ${DATA_DIR}/${SPEAKER}/r2v_dataset \
      --dataname ${SPEAKER}_bmold_win3 \
      --niter_decay 0 \
      --niter ${EPOCH_R2V} \
      --save_epoch_freq ${EPOCH_R2V} \
      --checkpoints_dir ${NET_DIR}/r2v \
    && cd ${CWD};
    if [[ $? != 0 ]]; then
      printf "${ERROR} Failed to train render-to-video network!\n"
      exit 1
    fi
  fi

  # # *----------------------------------------------------- Test -----------------------------------------------------* #

  # local DONE_FLAG_ARCFACE=${DATA_DIR}/${SPEAKER}/done_arcface_test.flag
  # if [ -f "${DONE_FLAG_ARCFACE}" ]; then
  #   printf "Arcface is already runned\n"
  # else
  #   cd ${CWD}/render-to-video/arcface && python3 yk_test_batch.py \
  #     --imglist ${DATA_DIR}/${SPEAKER}/r2v_dataset/list/testB/${SPEAKER}_bmold.txt --gpu 0
  #   if [[ $? != 0 ]]; then
  #     printf "${ERROR} Failed to prepare arcface feature for render-to-video!\n"
  #     exit 1
  #   fi
  #   touch ${DONE_FLAG_ARCFACE};
  # fi

  # cd ${CWD}/render-to-video && \
  # python3 yk_test.py \
  #   --model yk_memory_seq \
  #   --name memory_seq_p2p/${SPEAKER} \
  #   --num_test 200 \
  #   --imagefolder '' \
  #   --epoch ${EPOCH_R2V} \
  #   --dataroot ${DATA_DIR}/${SPEAKER}/r2v_dataset \
  #   --dataname ${SPEAKER}_bmold \
  #   --checkpoints_dir ${NET_DIR}/r2v \
  #   --results_dir ${DATA_DIR}/${SPEAKER}/r2v_results \
  #   --gpu_ids 0 \
  # ;

}

RUN_VOCASET --epoch_a2c=10 --epoch_r2v=5 --debug
