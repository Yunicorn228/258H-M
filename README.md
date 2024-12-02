# 258H-M

### Run for a single time

```bash
python run_single.py -m SASRecRoPE -d H&M --gpu_id=0
```

### Hyperparameter Tuning

```bash
mkdir hyper_res/    # Run only once at your first time
python run_hyper.py -m SASRecRoPE -d H&M -e test -hp hyper_props/norm.json --gpu_id=0
```
