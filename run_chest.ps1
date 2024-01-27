$env:CUDA_VISIBLE_DEVICES = "1"
python .\train_network_dbt.py --dataset_path /data/chest_xray --category chest --patch_size 128 --batch_size 300
