import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

# Sample data: Replace these lists with your actual extracted values from the image
mae_scratch = [0.0258, 0.0260, 0.7686, 0.7612, 0.7604, 0.7355, 0.7390,
               0.0586, 0.0585, 0.8866, 0.8404, 4.4945, 0.8409, 0.7870,
               0.0249, 0.0252, 4.4101, 4.4546, 0.8537, 3.9738, 3.9190,
               0.2060, 0.0345, 4.9259, 5.5676, 5.3926, 5.2621, 6.7978]

mae_pretrained = [0.7777, 0.7776, 0.7476, 0.7468, 0.7476, 1.2119, 0.7432,
                  0.9297, 0.9044, 0.9297, 0.9094, 4.3853, 0.8095, 1.0865,
                  4.2911, 4.3784, 0.9212, 3.8967, 3.8690,
                  5.8031, 5.7304, 5.6760, 5.4928, 7.8071]  # shorter list (25) – match to available

r2_scratch = [0.8214, 0.8231, 0.9326, 0.9357, 0.9349, 0.9380, 0.9381,
              0.7288, 0.7233, 0.9439, 0.9480, 0.9859, 0.9497, 0.9554,
              0.9056, 0.9043, 0.9867, 0.9863, 0.9466, 0.9889, 0.9891,
              0.3647, 0.5711, 0.6105, 0.5674, 0.5829, 0.4243, 0.2997]

r2_pretrained = [0.9331, 0.9375, 0.9375, 0.9376, 0.9375, 0.9428, 0.9358,
                 0.9368, 0.9234, 0.9368, 0.9206, 0.9866, 0.9518, 0.9235,
                 0.9873, 0.9868, 0.9932, 0.9892, 0.9894,
                 0.5140, 0.5269, 0.5380, 0.3840, 0.0873]  # 25 values

smape_scratch = [16.3840, 15.6640, 4.7899, 4.7684, 4.7460, 4.7934, 4.6512,
                 2.6986, 2.6890, 1.6271, 1.5457, 1.6940, 1.5390, 1.4411,
                 1.4597, 1.4754, 1.6675, 1.6742, 1.5750, 1.4937, 1.4775,
                 26.5920, 3.6495, 16.3237, 18.1999, 17.5886, 13.4801, 18.6675]

smape_pretrained = [4.8760, 4.8760, 4.6972, 4.6540, 4.6934, 4.3972, 4.6215,
                    1.6984, 1.6591, 1.6984, 1.6519, 1.6552, 1.4773, 1.9763,
                    1.6236, 1.6466, 1.6922, 1.4647, 1.4588,
                    18.8694, 18.7679, 18.6679, 14.6830, 18.5156]  # 25 values

# Trim to match lengths (e.g., only use samples where both scratch and pretrained data exist)
min_len = min(len(mae_pretrained), len(mae_scratch))
mae_scratch = mae_scratch[:min_len]
mae_pretrained = mae_pretrained[:min_len]
r2_scratch = r2_scratch[:min_len]
r2_pretrained = r2_pretrained[:min_len]
smape_scratch = smape_scratch[:min_len]
smape_pretrained = smape_pretrained[:min_len]

# Define a function to run tests and package results
def compare_metrics(scratch, pretrained, metric_name):
    t_stat, t_p = ttest_rel(scratch, pretrained)
    try:
        w_stat, w_p = wilcoxon(scratch, pretrained)
    except:
        w_stat, w_p = None, None
    return {
        'Metric': metric_name,
        'T-test p-value': round(t_p, 6),
        'Wilcoxon p-value': round(w_p, 6) if w_p is not None else 'N/A',
        'Mean (Scratch)': round(sum(scratch) / len(scratch), 4),
        'Mean (Pretrained)': round(sum(pretrained) / len(pretrained), 4),
    }

# Collect results
results = pd.DataFrame([
    compare_metrics(mae_scratch, mae_pretrained, 'MAE'),
    compare_metrics(r2_scratch, r2_pretrained, 'R2'),
    compare_metrics(smape_scratch, smape_pretrained, 'SMAPE')
])

print(results)
