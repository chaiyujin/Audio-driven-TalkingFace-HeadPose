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

  local CKPT_A2C="${NET_DIR}/atcnet/atcnet_lstm_${EPOCH_A2C}.pth"
  if [ -f "$CKPT_A2C" ]; then
    printf "Audio to Expression network is already trained: '$CKPT_A2C'\n"
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

    # rm latest_* to save disk space
    if [ -f "${NET_DIR}/r2v/memory_seq_p2p/$SPEAKER/latest_net_G.pth" ]; then
      rm "${NET_DIR}/r2v/memory_seq_p2p/$SPEAKER/latest_net_mem.pth"
      rm "${NET_DIR}/r2v/memory_seq_p2p/$SPEAKER/latest_net_D.pth"
      rm "${NET_DIR}/r2v/memory_seq_p2p/$SPEAKER/latest_net_G.pth"
    fi
  fi

  # *----------------------------------------------------- Test -----------------------------------------------------* #


  for ((i=20;i<21;i++)); do
    local clip=$(printf clip"%02d" $i)
    local clip_dir=$DATA_DIR/$SPEAKER/test/$clip
    local audio_path=$clip_dir/audio/audio.wav
    local pred_coeff_dir=$clip_dir/coeff_pred

    printf "Running test for $SPEAKER/test/$clip\n"

    # predict coefficients from audio
    cd $CWD/Audio/code && \
    python3 yk_atcnet_test.py --pose 1 --relativeframe 0 --dataset multi_clips --device_ids 0 \
      --model_name ${CKPT_A2C} \
      --in_file ${audio_path} \
      --sample_dir ${pred_coeff_dir} \
    && \
    cd $CWD

    # reenact with predicted coefficients
    cd $CWD/Deep3DFaceReconstruction && \
    python3 reenact_vocaset.py ${clip_dir} && \
    cd $CWD

    # prepare test data
    python3 utils/tools_vocaset_video.py build_r2v_dataset \
      --dataset_dir $DATA_DIR \
      --speakers $SPEAKER/test/$clip \
      --data_type test \
      ${DEBUG} \
    ;

    # prepare arcface feature
    local arcface_flag=${clip_dir}/done_arcface.flag
    if [ -f "${arcface_flag}" ]; then
      printf "Arcface is already runned\n"
    else
      cd ${CWD}/render-to-video/arcface && python3 yk_test_batch.py \
        --imglist ${clip_dir}/r2v_dataset/list/testB/bmold.txt --gpu 0 \
      && cd ${CWD};
      if [[ $? != 0 ]]; then
        printf "${ERROR} Failed to prepare arcface feature for render-to-video!\n"
        continue
      fi
      touch ${arcface_flag};
    fi

    local r2v_flag=${clip_dir}/done_r2v.flag
    if [ -f "${r2v_flag}" ]; then
      printf "Render-to-Video is already runned\n"
    else
      cd ${CWD}/render-to-video && \
      python3 yk_test.py \
        --model yk_memory_seq \
        --name memory_seq_p2p/${SPEAKER} \
        --num_test 200 \
        --imagefolder '' \
        --epoch ${EPOCH_R2V} \
        --dataroot ${clip_dir}/r2v_dataset \
        --dataname bmold \
        --checkpoints_dir ${NET_DIR}/r2v \
        --results_dir ${clip_dir}/r2v_reenact \
        --gpu_ids 0 \
      && cd ${CWD};
      if [[ $? != 0 ]]; then
        printf "${ERROR} Failed to run render-to-video!\n"
        continue
      fi
      touch ${r2v_flag};
    fi

    python3 utils/blend_results.py \
      --image_dir ${clip_dir}/crop \
      --coeff_dir ${clip_dir}/coeff_reenact \
      --r2v_dir ${clip_dir}/r2v_reenact \
      --output_dir ${clip_dir}/results \
    ;

    ffmpeg -y -loglevel error \
      -thread_queue_size 8192 -i $clip_dir/results/%06d_real.png \
      -thread_queue_size 8192 -i $clip_dir/results/%06d_fake.png \
      -thread_queue_size 8192 -i $audio_path \
      -filter_complex hstack=inputs=2 -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p -shortest \
      $clip_dir/result-real_fake.mp4 \
    ;
  done
}
