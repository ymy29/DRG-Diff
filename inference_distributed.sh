export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0  
export NCCL_TIMEOUT=7200
export TORCH_NCCL_TIMEOUT=7200
export NCCL_WATCHDOG_TIMEOUT=0
export NCCL_ENABLE_WATCHDOG=0

accelerate launch --num_processes=4 --dynamo_backend=no --main_process_port=29700 inference.py \
  --config configs/batch_default.yaml \
  --batch_mode \
  --use_cross_attention \
  --cross_attention_heads=8 \
  --intra_attention_type="conv" \
  --num_inference_steps 50 \
  --output_dir ../inference_result \

