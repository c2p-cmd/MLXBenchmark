# !/bin/bash
# Run all experiments

scripts=("src.scripts.np" "src.scripts.pytorch" "src.scripts.sklearn" "src.scripts.mlx")

for script in "${scripts[@]}"; do
   echo "Running ${script}..."
   python -m ${script}
done
