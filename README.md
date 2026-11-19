# Physics-Informed Solar Coronal DEM Surrogate (SDO/AIA)

Real-time $1 \times 1$ 2D Convolutional Neural Network surrogate for inverting six SDO/AIA EUV channels ($94, 131, 171, 193, 211, 335\text{ \AA}$) into 18 logarithmic coronal temperature bins ($\log_{10} T \in [5.5, 7.2]\text{ K}$).

## Key Benchmarks (NVIDIA GeForce MX350)
- **Full-Disc ($512 \times 512$) Inference:** `176.62 ms` (`1,472.1×` faster than $L_1$ Basis Pursuit)
- **Log-Space $R^2$:** `0.8495`
- **Median Temperature Residual:** `0.22%`
- **Physical Positivity:** `100%`

## Run Interactive Web App
```bash
pip install torch numpy matplotlib gradio
python src/app.py