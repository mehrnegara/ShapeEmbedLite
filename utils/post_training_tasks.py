import torch
import random
import itertools
import numpy as np
from utils.display import *

def average_shape( model, zs, lbls, output_d
                 , pre_plot_rescale = None
                 , space_dims = 2
                 , device = 'cpu' ):
  # mean average shape
  print(f'generating mean shapes report to {output_d / "mean_shapes.pdf"}')
  def mean_average_shapes(model, zs, lbls):
    with torch.no_grad():
      mean_shapes = {}
      for lbl in np.unique(lbls):
        mean_z = zs[lbls==lbl].mean(axis=0)
        _, mean_recon = model.decoder(torch.tensor(mean_z, dtype=torch.float).to(device))
        if pre_plot_rescale:
          mean_recon = torch.mul(mean_recon, pre_plot_rescale)
        mean_shapes[lbl] = mean_z, mean_recon
    return mean_shapes
  mean_shapes = mean_average_shapes(model, zs, np.array(lbls))
  mean_shape_rprt(mean_shapes, space_dims, output_d / 'mean_shapes.pdf')

def random_shapes( model, n_samples, ls_sz, output_d
                 , pre_plot_rescale = None
                 , space_dims = 2
                 , device = 'cpu' ):
  # random shape sampling
  with torch.no_grad():
    means = torch.zeros([n_samples, ls_sz])
    stds = torch.ones([n_samples, ls_sz])
    _, recons = model.decoder(torch.normal(means, stds).to(device))
    if pre_plot_rescale:
      recons = torch.mul(recons, pre_plot_rescale)
  for i, recon in enumerate(recons.squeeze().to('cpu')):
    print(f'generating random sample {i} to {output_d / f"random_sample_{i}.pdf"}')
    plot_points( dst_mat_to_coords(asym_to_sym(recon), space_dims)
               , save_path=output_d / f'random_sample_{i}.pdf' )

def latent_space_lerp(model, z0, z1, n_samples):
  # get all latent space samples (start, intermediary steps and end), using
  # linear interpolation
  zs = [(1 - t) * z0 + t * z1 for t in np.linspace(0.0, 1.0, n_samples + 2)]
  # compute each reconstruction by passing z samples to the model's decoder
  recons = []
  for z in zs:
    with torch.no_grad(): _, recon = model.decoder(z)
    recons.append(recon)
  return recons

def trajectory_figure ( tensor_dms, output_path
                      , show_dm = True
                      , no_start_end = True
                      , start_title = 'origin'
                      , sample_title = 'sample'
                      , end_title = 'endpoint'
                      , pre_plot_rescale = None
                      , space_dims = 2 ):
  # prepare figure
  n_rows = len(tensor_dms)
  n_cols = 1
  if not no_start_end: n_cols = 3
  if show_dm: n_cols = n_cols * 2
  fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
  # prepare tensors for display
  def preproc (x):
    if pre_plot_rescale: x = torch.mul(x, pre_plot_rescale)
    x_dm = asym_to_sym(x.squeeze().cpu().numpy())
    return x_dm, dst_mat_to_coords(x_dm, space_dims)
  data = list(map(preproc, tensor_dms))
  # populate the figure
  dm_s, pts_s = data[0]
  dm_e, pts_e = data[-1]
  for idx, (dm, pts) in enumerate(data):
    # determine title based on position
    if idx == 0: title = f'{start_title} (idx {idx})'
    elif idx == n_rows - 1: title = f'{end_title} (idx {idx})'
    else: title = f'{sample_title} {idx}'
    # select axes
    ax_pts_s, ax_dm_s = None, None
    ax_pts, ax_dm = None, None
    ax_pts_e, ax_dm_e = None, None
    if (not show_dm) and (not no_start_end):
      ax_pts_s, ax_pts, ax_pts_e = axes[idx]
    elif (not show_dm) and no_start_end:
      ax_pts = axes[idx]
    elif show_dm and (not no_start_end):
      ax_pts_s, ax_dm_s, ax_pts, ax_dm, ax_pts_e, ax_dm_e = axes[idx]
    elif show_dm and no_start_end:
      ax_pts, ax_dm = axes[idx]
    else: raise Exception('Should not happen')
    # plot points
    if ax_pts_s:
      ax_pts_s.set_title(f'start - points')
      ax_pts_s.scatter(pts_s[:, 0], pts_s[:, 1])
    if ax_pts:
      ax_pts.set_title(f'{title} - points')
      ax_pts.scatter(pts[:, 0], pts[:, 1])
    if ax_pts_e:
      ax_pts_e.set_title(f'end - points')
      ax_pts_e.scatter(pts_e[:, 0], pts_e[:, 1])
    # plot distance matrix
    if ax_dm_s:
      ax_dm_s.set_title(f'start - dist matrix')
      ax_dm_s.imshow(dm_s, aspect='auto')
    if ax_dm:
      ax_dm.set_title(f'{title} - dist matrix')
      ax_dm.imshow(dm, aspect='auto')
    if ax_dm_e:
      ax_dm_e.set_title(f'end - dist matrix')
      ax_dm_e.imshow(dm_e, aspect='auto')
  plt.tight_layout()
  fig.savefig(output_path)
  plt.close(fig)

def intraclass_trajectories ( model, zs, lbls, output_d
                            , n_sample_steps = 10
                            , show_dm = True
                            , no_start_end = True
                            , pre_plot_rescale = None
                            , space_dims = 2
                            , device = 'cpu' ):
  # identify classes
  unique_lbls = np.unique(lbls)
  # for each class, intraclass trajectory sampling:
  # random origin -> n_sample_steps uniform samples -> random endpoint
  for class_idx, lbl in enumerate(unique_lbls):
    idxs = np.where(np.array(lbls) == lbl)[0]
    if len(idxs) < 2: continue # ignore classes with fewer than 2 elements
    # pick two distinct latent samples for this class
    i0, i1 = np.random.choice(idxs, size=2, replace=False)
    z0 = torch.tensor(zs[i0], dtype=torch.float).to(device)
    z1 = torch.tensor(zs[i1], dtype=torch.float).to(device)
    # perform linear interpolation between latent samples,
    # and get corresponding reconstructions
    recons = latent_space_lerp(model, z0, z1, n_sample_steps)
    # generate a visualization from the reconstructions
    save_path = output_d / f'trajectory_intraclass_{class_idx}.pdf'
    trajectory_figure( recons, save_path
                     , show_dm = show_dm
                     , no_start_end = no_start_end
                     , start_title = f'{lbl} origin'
                     , sample_title = f'{lbl} sample'
                     , end_title = f'{lbl} endpoint'
                     , pre_plot_rescale = pre_plot_rescale
                     , space_dims = space_dims )
    print(f'generated {lbl} intraclass trajectory plot at {save_path}')

def interclasses_trajectories ( model, zs, lbls, output_d
                              , n_sample_steps = 10
                              , n_class_pairs = None
                              , show_dm = True
                              , no_start_end = True
                              , pre_plot_rescale = None
                              , space_dims = 2
                              , device = 'cpu' ):
  # identify classes
  unique_lbls = np.unique(lbls)
  if len(unique_lbls) < 2:
    print(f'skipping interclass trajectories (not enough classes)')
    return
  # generate class pairs
  all_pairs = list(itertools.combinations(range(len(unique_lbls)), 2))
  if n_class_pairs: n_class_pairs = min(n_class_pairs, len(all_pairs))
  else: n_class_pairs = len(all_pairs)
  class_pairs = random.choices(all_pairs, k=n_class_pairs)
  # for each class pair, interclass trajectory sampling:
  # random origin -> n_sample_steps uniform samples -> random endpoint
  for c_idx_0, c_idx_1 in class_pairs:
    idxs_0 = np.where(np.array(lbls) == unique_lbls[c_idx_0])[0]
    idxs_1 = np.where(np.array(lbls) == unique_lbls[c_idx_1])[0]
    # pick latent sample for each class
    i0 = np.random.choice(idxs_0, size=1, replace=False)
    i1 = np.random.choice(idxs_1, size=1, replace=False)
    z0 = torch.tensor(zs[i0], dtype=torch.float).to(device)
    z1 = torch.tensor(zs[i1], dtype=torch.float).to(device)
    # perform linear interpolation between latent samples,
    # and get corresponding reconstructions
    recons = latent_space_lerp(model, z0, z1, n_sample_steps)
    # generate a visualization from the reconstructions
    save_path = output_d / f'trajectory_interclass_{c_idx_0}_{c_idx_1}.pdf'
    trajectory_figure( recons, save_path
                     , show_dm = show_dm
                     , no_start_end = no_start_end
                     , start_title = f'{unique_lbls[c_idx_0]} origin'
                     , sample_title = f'sample'
                     , end_title = f'{unique_lbls[c_idx_1]} endpoint'
                     , pre_plot_rescale = pre_plot_rescale
                     , space_dims = space_dims )
    print(f'generated {unique_lbls[c_idx_0]}-{unique_lbls[c_idx_1]} interclass trajectory plot at {save_path}')
