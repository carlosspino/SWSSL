$env:CUDA_VISIBLE_DEVICES = "1"
python .\train_network_dbt.py --dataset_path \Users\Usuario\Desktop\Neural_Networks\Proyecto\SWSSL\data\chest_xray --category chest --patch_size 128 --batch_size 300
