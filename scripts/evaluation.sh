python training/evaluate.py --config exp/highlight-unify/expert_model/poslen70_bs256_sd1/text/clip_text.yml \
                            --model_path exp/highlight-unify/expert_model/poslen70_bs256_sd1/text/checkpoint/best.ckpt \
                            --eval_split_name val \
                            --eval_path annotation/qvhighlight/highlight_segment_val_release_rand2.jsonl \
                            --results_dir exp/highlight-unify/expert_model/poslen70_bs256_sd1/text \