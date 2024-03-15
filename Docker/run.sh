docker run -td \
  --gpus '"device=1,2"'\
  --shm-size=50g \
  -v $(cat experiments_path.txt):/coigan/COIGAN-IROS-2024/experiments \
  -v $(cat datasets_path.txt):/coigan/COIGAN-IROS-2024/datasets \
  coigan-iros-2024:latest