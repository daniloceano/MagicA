Changelog
=========

Recent Changes
--------------

* Monte Carlo API: `monte_carlo_fit` now returns only numeric data in the
   xarray.Dataset. Plot generation is optional: pass `fig_output_path` to save a
   2x3 summary figure (parameters on first row, test p-values / RMSE on second),
   with red dashed vertical lines marking stability sample sizes per variable.
   Old parameter `plot` was replaced by `fig_output_path` (None by default) and
   `plot_type` ('series' or 'boxplots'). Figure object is no longer kept in
   memory; the saved path is stored under the `figure_path` attribute.

* Added optional progress bar fallback if `tqdm` is not installed.
