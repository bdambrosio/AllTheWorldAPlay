cd ../../../exllamav2
export CUDA_VISIBLE_DEVICES=1
python3 convert.py -i ../models/DeepSeek-R1-Distill-Qwen-32B -o DeepSeek-R1-Distill-Qwen-32B-exl2 -cf DeepSeek-R1-Distill-Qwen-32B-exl2 -l 2048 -b 8.0 -hb 8 -ss 8192