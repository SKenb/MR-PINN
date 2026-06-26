## Setup Python Environment

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Run 1 Species Benchmarks

Same-network-epochs 1-species benchmark with `n=30` runs and 2200 total epochs:

    python3 Python/A3_PINN_VS_rPINN.py \
    --runs 30 \
    --output-dir Benchmarks/B1/pinn \
    --data-path Python/Data/flowReactorData.pkl \
    --pinn-epochs 2200 \
    --pinn-threshold 2e-5 \
    --pinn-patience 100 \
    --rpinn-epochs 300,400,500,1000 \
    --rpinn-thresholds 2e-5,2e-5,2e-5,2e-5 \
    --rpinn-patience 100

    python3 Python/A3_PINN_VS_rPINN_interpolation_compare.py \
    --runs 30 \
    --output-root Benchmarks/B1/interpolation_methods \
    --data-path Python/Data/flowReactorData.pkl \
    --pinn-epochs 2200 \
    --pinn-threshold 2e-5 \
    --pinn-patience 100 \
    --rpinn-epochs 300,400,500,1000 \
    --rpinn-thresholds 2e-5,2e-5,2e-5,2e-5 \
    --rpinn-patience 100 \
    --methods nearest,linear,spline

After the benchmark, generate the combined TikZ convergence figure with:

    python3 Python/make_tikz_convergence_4methods.py \
    --pinn-csv Benchmarks/B1/pinn/convergence_tikz_data.csv \
    --history-root Benchmarks/B1/interpolation_methods \
    --output-dir Latex/B1 \
    --pinn-epochs 2200 \
    --xmax 2200

Build the LaTeX plots in Latex/B1


## Run 3 Species Benchmarks

Same-network-epochs 3-species benchmark with `n=30` runs:

    python3 Python/B3_PINN_VS_rPINN_3Species.py \
    --runs 30 \
    --pinn-epochs 2250 \
    --output-dir Benchmarks/B1-3Species \
    --data-path Python/Data/flowReactorData_3species.pkl

After the benchmark, generate the combined TikZ convergence figure with:

    python3 Python/make_tikz_figures_3species.py \
    --input-dir Benchmarks/B1-3Species \
    --output-dir Latex/B1-3Species \
    --pinn-epochs 2250 \
    --rpinn-epochs 2250
    
