from skimage import measure


def psnr(y_true, y_pred, max_value=2):
  psnr_metric = measure.compare_psnr(y_true, y_pred, max_value)
  return psnr_metric
