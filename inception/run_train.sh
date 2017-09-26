#cd ./inception
#bazel build //inception:flowers_train

MODEL_PATH='/Users/gaurav.kaila/Documents/Projects/Image_Recognition_App/inception-v3/model.ckpt-157585'

FLOWERS_DATA_DIR='/Users/gaurav.kaila/Documents/Projects/Image_Recognition_App/data_dir/model_ready_data'

TRAIN_DIR='/Users/gaurav.kaila/Documents/Projects/Image_Recognition_App/chpk/'

bazel-bin/inception/flowers_train \
  --train_dir="${TRAIN_DIR}" \
  --data_dir="${FLOWERS_DATA_DIR}" \
  --pretrained_model_checkpoint_path="${MODEL_PATH}" \
  --fine_tune=True \
  --initial_learning_rate=0.001 \
  --input_queue_memory_factor=1
