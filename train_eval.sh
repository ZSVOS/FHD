GPU_ID='0'
PORT='29500'
EXP='FHD-for-ChangeDetection'

CUDA_VISIBLE_DEVICES=${GPU_ID} PORT=${PORT} bash tools/dist_train.sh local_configs/FHD/fhd.levir.py 1 --seed 1208 --deterministic --exp-name ${EXP}
CUDA_VISIBLE_DEVICES=${GPU_ID} python eval.py --config local_configs/FHD/fhd.levir.py --checkpoint work_dirs/fhd.levir/latest.pth --show-dir output_path/levir/label/
python metrics.py --pred_path output_path/levir/ --gt_path ./data/LEVIR/test/

CUDA_VISIBLE_DEVICES=${GPU_ID} PORT=${PORT} bash tools/dist_train.sh local_configs/FHD/fhd.levir+.py 1 --seed 1208 --deterministic --exp-name ${EXP}
CUDA_VISIBLE_DEVICES=${GPU_ID} python eval.py --config local_configs/FHD/fhd.levir+.py --checkpoint work_dirs/fhd.levir+/latest.pth --show-dir output_path/levir+/label/
python metrics_v2.py --pred_path output_path/levir+/ --gt_path ./data/LEVIR+/test/

CUDA_VISIBLE_DEVICES=${GPU_ID} PORT=${PORT} bash tools/dist_train.sh local_configs/FHD/fhd.s2looking.py 1 --seed 1208 --deterministic --exp-name ${EXP}
CUDA_VISIBLE_DEVICES=${GPU_ID} python eval.py --config local_configs/FHD/fhd.s2looking.py --checkpoint work_dirs/fhd.s2looking/latest.pth --show-dir output_path/s2looking/label/
python metrics_v2.py --pred_path output_path/s2looking/ --gt_path ./data/S2Looking_256x256/test/

CUDA_VISIBLE_DEVICES=${GPU_ID} PORT=${PORT} bash tools/dist_train.sh local_configs/FHD/fhd.dsifn.py 1 --seed 1208 --deterministic --exp-name ${EXP}
CUDA_VISIBLE_DEVICES=${GPU_ID} python eval.py --config local_configs/FHD/fhd.dsifn.py --checkpoint work_dirs/fhd.dsifn/latest.pth --show-dir output_path/dsifn/label/
python metrics_v2.py --pred_path output_path/dsifn/ --gt_path ./data/DSIFN/test/