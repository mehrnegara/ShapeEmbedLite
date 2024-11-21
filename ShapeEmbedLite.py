#! /usr/bin/env python3

import sys
import math
import copy
import tqdm
import torch
import pathlib
import argparse
import datetime
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
##import concurrent.futures as F

from utils.nvidia import *
from utils.models import *
from utils.display import *
from utils.datasets import *
from utils.classification import *
from utils.post_training_tasks import *
from utils.distance_matrix_loss import *

# default device
dflt_device = torch.device('cpu')
if torch.cuda.is_available():
  idx = select_most_available_nvidia_device()
  if idx is None: idx = np.random.randint(torch.cuda.device_count())
  dflt_device = torch.device(f"cuda:{idx}")
elif torch.backends.mps.is_available():
  dflt_device = torch.device('mps')

# loss function
################################################################################

def vae_loss( recon_x, x, z_mean, z_log_var, beta
            , use_index_invariant_loss=True
            , use_reflection_invariant_loss=True
            ):
  """
  Custom VAE loss function with min_loss as the new reconstruction loss.
  """
  #print(f'min x: {x.min()}, max x: {x.max()}, man x {x.mean()}')
  #print(f'min recon_x: {recon_x.min()}, max recon_x: {recon_x.max()}, mean recon_x {recon_x.mean()}')
  # track individual loss components
  # start with a 0.0 total accumulator
  agg_loss = torch.Tensor([0.0]).to(recon_x.device)
  losses = { 'total': 0.0 }

  # Recontruction loss: old reconstruction #
  ###################################

  x_ = x.detach().clone()
  #print(f'x_ shape: {x_.shape}')
  # assuming dimensions B,C,H,W
  # using C (dim 1) to stack intermediate results
  # operating on H, W for various transformations (dims 2, 3)

  if use_index_invariant_loss:
    _, _, _, sz = x_.shape
    x_ = torch.cat( tuple(torch.roll(x_, (i,i), dims=(2,3)) for i in range(sz))
                  , dim=1 )
    #print(f'x_ shape after index: {x_.shape}')

  if use_reflection_invariant_loss:
    x_ = torch.cat( (x_, torch.flip(x_, dims=(2,3)))
                  , dim=1)
    #print(f'x_ shape after reflection: {x_.shape}')

  if use_reflection_invariant_loss or use_index_invariant_loss:
    _, sz, _, _ = x_.shape
    diffs = torch.sub(recon_x.repeat(1,sz,1,1), x_)
    means = torch.mean(torch.square(diffs), dim=(2,3))
    minimums = torch.min(means, dim=1).values
    # take the mean of the individual samples'minimums instead of the minimum of all like in the first attempt
    mean_minimum = torch.mean(minimums)
    recon_loss = mean_minimum
    #print(f'recon_loss (index/reflection invariant): {recon_loss}')
  else:
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
  losses['recon'] = recon_loss.item()
  agg_loss += recon_loss

  losses['recon'] = recon_loss.item()
  agg_loss += recon_loss

  # KL Divergence loss #
  ######################
  kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
  beta_kl_loss = beta * kl_loss
  losses['kl'] = kl_loss.item()
  losses['beta_kl'] = beta_kl_loss.item()
  agg_loss += beta_kl_loss

  # return losses
  losses['total'] = agg_loss.item()
  return agg_loss, losses

################################################################################

def single_epoch_train( model, optimizer, train_loader, e_idx, tb_writer
                      , beta=0.0
                      , use_index_invariant_loss=True
                      , use_reflection_invariant_loss=True
                      , dev=dflt_device):
  # loss book-keeping
  running_losses = {
    'total': 0.0
  , 'recon': 0.0
  , 'kl': 0.0
  , 'beta_kl': 0.0
  }

  # get each training batch from the loader
  n_batches = len(train_loader)
  for i, batch in tqdm.tqdm( enumerate(train_loader), desc='training batch'
                           , total=n_batches, leave=False ):
    # reset gradient
    optimizer.zero_grad()
    # our loss does not consider labels, only get the sample out of the batch
    x, _ = batch
    x = x.to(dev)
    # predict
    preproc_x, recon_x, _, z_mean, z_log_var, _ = model(x)
    #print(f'devices x: {x.device}, recon_x: {recon_x.device}, z_mean: {z_mean.device}, z_log_var: {z_log_var.device}')
    # compute losses
    loss, losses = vae_loss( recon_x, preproc_x, z_mean, z_log_var, beta=beta
                           , use_index_invariant_loss=use_index_invariant_loss
                           , use_reflection_invariant_loss=use_reflection_invariant_loss
                           )
    # compute gradients based on loss
    loss.backward()
    # integrate gradients into weights (impacted by learning rate)
    optimizer.step()
    # loss book-keeping
    for k, v in losses.items(): running_losses[k] += v
    if i % 10 == 9:
      tb_x = e_idx * n_batches + i + 1
      last_losses = { k: v/10 for k, v in running_losses.items() }
      running_losses = { k: 0.0 for k in running_losses.keys() }
      for k, v in last_losses.items(): tb_writer.add_scalar(f'{k}_10batch_loss', v, tb_x)

  if last_losses is None:
    last_losses = { k: v/((i%10)+1) for k, v in running_losses.items() }
  for k, v in last_losses.items(): tb_writer.add_scalar(f'{k}_epoch_loss', v, e_idx)

  return last_losses

def single_epoch_validate_tb_logging(og, recon, n_components, e_idx, b_idx, tb_writer):
  #tb_writer.add_figure( f'validation batch {b_idx}', validation_dump(og, recon, n_components)
  #                    , e_idx )
  pass

def single_epoch_validate( model, val_loader, n_components, e_idx, last_e_idx, tb_writer
                         , n_splits=None, classify_with_scale=False
                         , beta=0.0
                         , use_index_invariant_loss=True
                         , use_reflection_invariant_loss=True
                         , dev=dflt_device ):
  # executor pool for tensorboard logging
  #ex = F.ProcessPoolExecutor()
  #ex = F.ThreadPoolExecutor()
  # book-keeping
  running_vloss = 0.0
  Z, lbls = [], []
  if classify_with_scale: Z_ = []
  # Disable gradient computation and reduce memory consumption.
  with torch.no_grad():
    for b_idx, batch in enumerate( tqdm.tqdm( val_loader
                                            , desc='validation batch'
                                            , total=len(val_loader)
                                            , leave=False ) ):
      x, y = batch
      x = x.to(dev)
      preproc_x, recon_x, z, z_mean, z_log_var, og_scale = model(x)
      #print(f'devices x: {x.device}, recon_x: {recon_x.device}, z_mean: {z_mean.device}, z_log_var: {z_log_var.device}')
      _, losses = vae_loss( recon_x, preproc_x, z_mean, z_log_var, beta=beta
                          , use_index_invariant_loss=use_index_invariant_loss
                          , use_reflection_invariant_loss=use_reflection_invariant_loss
                          )
      # book-keeping
      running_vloss += losses['total']
      lbls.extend(y.tolist()), Z.extend(z.tolist())
      if classify_with_scale: Z_.extend(torch.cat((z, og_scale), dim=-1).tolist())
      # tensor board tracing
      single_epoch_validate_tb_logging( torch.nan_to_num(x)#.to('cpu')
                                      , torch.nan_to_num(recon_x)#.to('cpu')
                                      , n_components
                                      , e_idx, b_idx, tb_writer)
  running_vloss /= len(val_loader)
  tb_writer.add_scalar("total_val_loss", running_vloss, e_idx)
  #ex.shutdown()

  # classification on last epoch

  res_classify_val = {}
  if classify_with_scale: res_classify_with_scale_val = None
  if e_idx == last_e_idx:
    res_classify_val['latent_space_only'] = run_classification(Z, lbls, n_splits=n_splits)
    if classify_with_scale:
      res_classify_val['with_scale'] = run_classification(Z_, lbls, n_splits=n_splits)

  return running_vloss, res_classify_val

def test_model( model, dataloader
              , n_splits=None
              , classify_with_scale=False
              , beta=0.0
              , use_index_invariant_loss=True
              , use_reflection_invariant_loss=True
              , dev=dflt_device
              , report_callback=None
              , summary_samples=None
              ):
  # recover dataset #
  subset = dataloader.dataset
  if getattr(subset.dataset, "classes", None):
    n_classes = len(subset.dataset.classes)
  elif callable(getattr(subset.dataset, "get_classes"), None):
    n_classes = len(subset.dataset.get_classes())

  # prepare for summary if needed #
  summary_objs = {}
  if summary_samples:
    n_samples = math.ceil(summary_samples / n_classes)
    smpls = {lbl: [] for lbl in range(n_classes)}
    smpl_idx = copy.deepcopy(subset.indices)
    #np.random.shuffle(smpl_idx)
    for idx in smpl_idx:
      # exit if done
      if all(len(x) >= n_samples for x in smpls.values()): break
      # else, hunt for new samples
      _, lbl = subset.dataset[idx]
      if len(smpls[lbl]) < n_samples: smpls[lbl].append(idx)
    summary_objs = {lbl: [] for lbl in range(n_classes)}

  # run tests #
  with torch.no_grad():
    Z, lbls = [], []
    if classify_with_scale: Z_ = []
    for i, (x, lbl) in (pbar:=tqdm.tqdm( zip(subset.indices, dataloader)
                                       , desc='testing', total=len(dataloader)
                       )):
      pbar.set_postfix_str(f'(idx {i}, lbl {lbl})')
      x = x.to(dev)
      preproc_x, recon_x, z, z_mean, z_log_var, og_scale = model(x)
      _, losses = vae_loss( recon_x, preproc_x, z_mean, z_log_var, beta=beta
                          , use_index_invariant_loss=use_index_invariant_loss
                          , use_reflection_invariant_loss=use_reflection_invariant_loss
                          )
      # values used for classification test
      lbls.extend(lbl.tolist()), Z.extend(z.tolist())
      if classify_with_scale: Z_.extend(torch.cat((z, og_scale), dim=-1).tolist())
      # reporting #
      if callable(report_callback):
        lbl = lbl.item()
        rpt = report_callback(subset.dataset.samples[i][0], lbl, x, preproc_x, recon_x, losses['recon'])
        if summary_samples and i in smpls[lbl]: summary_objs[lbl].append(rpt)
    Z = np.array(Z)
    if classify_with_scale: Z_ = np.array(Z_)
    # run classification #
    res_classify_test = {}
    res_classify_test['latent_space_only'] = run_classification(Z, lbls, n_splits=n_splits)
    if classify_with_scale:
      res_classify_test['with_scale'] = run_classification(Z_, lbls, n_splits=n_splits)

  return [x for objs in summary_objs.values() for x in objs], res_classify_test, Z, lbls

def run_name(clargs):
  nm = f'{get_dataset_name(clargs)}'
  nm += f'_ls{clargs.latent_space_size}'
  if clargs.report_only: nm += '_report_only'
  elif clargs.skip_training: nm += '_skip_training'
  else:
    nm += f'_e{clargs.number_epochs}'
    nm += f'_b{clargs.beta}'
    nm += f'_lr{clargs.learning_rate}'
    if clargs.indexation_invariant_loss: nm += f'_idx_loss'
    if clargs.reflection_invariant_loss: nm += f'_rfl_loss'
    if clargs.reduce_lr_on_plateau: nm += f'_lr_rop'
    if clargs.circular_padding: nm += f'_cir_pad'
    if clargs.preprocess_normalize: nm += f'_preproc_norm{clargs.preprocess_normalize}'
    if clargs.preprocess_rescale: nm += f'_preproc_scale{clargs.preprocess_rescale}'
    if clargs.preprocess_augment: nm += f'_preproc_aug'
  if clargs.classify_with_scale: nm += f'_classify_with_scale'
  if clargs.pre_plot_rescale: nm += f'_pre_plot_scale{clargs.pre_plot_rescale}'
  return nm

def get_dataset_name(clargs):
  if clargs.dataset:
    return clargs.dataset[0]
  elif clargs.train_test_dataset:
    return clargs.train_test_dataset[0]
  else:
    raise ValueError("dataset name not specified")

def main(clargs):

  # which torch device is in use?
  dev = clargs.device
  print(f'using {dev} for torch run')

  # get dataloaders from input dataset
  dataset_name = get_dataset_name(clargs)
  if clargs.dataset:
    train_loader, val_loader, test_loader = get_dataloaders(clargs.dataset[1])
  elif clargs.train_test_dataset:
      train_loader, val_loader, test_loader = get_dataloaders(clargs.train_test_dataset[1:])
  else:
    sys.exit('dataset specification not present or not supported')
  sample_input, _ = next(iter(train_loader))
  batch_size, n_channels, matrix_size, matrix_size_ = sample_input.shape
  assert batch_size == 4, f'unexpected batch size {batch_size}, should be 4'
  assert n_channels == 1, f'unexpected numer of channels {n_channels}, should be 1'
  assert matrix_size == matrix_size_, f'non-square matrix: {matrix_size}x{matrix_size_}'
  assert math.log2(matrix_size).is_integer(), f'non-power-of-two matrix size: {matrix_size}'
  print(f'using loaders with samples of the shape {sample_input.shape}')
  print(f'(batch_size: {batch_size})')
  print(f'(n_channels: {n_channels})')
  print(f'(matrix size: {matrix_size}x{matrix_size})')

  # create output directory
  output_d = clargs.output_dir
  if output_d == None:
    output_d = pathlib.Path(f'./results/output_{run_name(clargs)}')
  output_d.mkdir(parents=True, exist_ok=True)

  # Initialize the VAE and optimizer
  ls_sz = clargs.latent_space_size * 2 # test for multiplying by 2 its size
  decoder_depth = 5
  model = MyNet( latent_dim=ls_sz, matrix_size=matrix_size
               , decoder_hidden_layers = [2**(i-1)*matrix_size for i in list(range(decoder_depth, 0, -1))]
               , space_dim=clargs.space_dim, padding=clargs.circular_padding
               , normalize_input=clargs.preprocess_normalize
               , rescale_input=clargs.preprocess_rescale
               , augment_input=clargs.preprocess_augment
               ).to(dev)
  model_was_trained = False
  if clargs.model_weights:
    print(f'loading model weights from {clargs.model_weights}')
    model.load_state_dict(torch.load( clargs.model_weights
                                    , weights_only=True
                                    , map_location=dev ))
    model_was_trained = True

  ##############################################################################


  if clargs.skip_training or clargs.report_only: # skip training
    print('\n---\nskipping training\n'+'-'*80)
  else: # training
    print('\n---\ntraining\n'+'-'*80)

    # tensorboard
    tb_run_dir =clargs.tensorboard_log_dir / f'{run_name(clargs)}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    tb_writer = SummaryWriter(tb_run_dir)
    tb_writer.add_graph(model, sample_input.to(dev))

    # optimizer
    print(f'learning rate: {clargs.learning_rate}')
    optimizer = optim.Adam(model.parameters(), lr=clargs.learning_rate)
    if clargs.reduce_lr_on_plateau is not None:
      sched_args = clargs.reduce_lr_on_plateau
      factor = 0.1
      patience = 10
      if len(sched_args) > 0: factor = float(sched_args[0])
      if len(sched_args) > 1: patience = int(sched_args[1])
      scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer, 'min'
                                                      , factor=factor
                                                      , patience=patience )
      print(f'dynamic learning rate reduction on plateau - factor {factor}, patience {patience}')

    loss_record = []

    for epoch in (pbar:=tqdm.tqdm(range(clargs.number_epochs), desc='training epochs')):

      # training #
      ############
      model.train(True)
      losses = single_epoch_train( model, optimizer, train_loader, epoch, tb_writer
                                 , beta=clargs.beta
                                 , use_index_invariant_loss=clargs.indexation_invariant_loss
                                 , use_reflection_invariant_loss=clargs.reflection_invariant_loss
                                 , dev=dev )
      postfix = f"loss: train {losses['total']:.2f}"

      # validation #
      ##############
      model.eval()
      losses['total_val'], res_classify_val = single_epoch_validate( model, val_loader
                                                                   , clargs.space_dim
                                                                   , epoch, clargs.number_epochs - 1
                                                                   , tb_writer
                                                                   , n_splits=clargs.number_splits_classify
                                                                   , classify_with_scale=clargs.classify_with_scale
                                                                   , beta = clargs.beta
                                                                   , use_index_invariant_loss = clargs.indexation_invariant_loss
                                                                   , use_reflection_invariant_loss = clargs.reflection_invariant_loss
                                                                   , dev=dev)
      postfix += f", val {losses['total_val']:.2f}"

      # learning rate potential update #
      ##################################
      if clargs.reduce_lr_on_plateau is not None:
        scheduler.step(losses['total_val'])
        postfix += f" - lr: {scheduler.get_last_lr()}"
      else: postfix += f" - lr: {clargs.learning_rate}"

      pbar.set_postfix_str(postfix)

      # book-keeping #
      ################
      loss_record.append(losses)
      tb_writer.flush()
      if epoch == clargs.number_epochs - 1:
        print('---')
        print('validation classification results:')
        for kind, nm, stat in [ (kk, k, v) for kk, vv in res_classify_val.items()
                                           for k, v in vv.items() ]:
          print(f"{kind} - {nm} - mean f1: {stat['f1_mean']:.3f}(std:{stat['f1_std']:.3f})")
          print(f"{kind} - {nm} - mean log_loss: {stat['log_loss_mean']:.3f}(std:{stat['log_loss_std']:.3f})")

    ##############################################################################

    model_was_trained = True

    # save model state_dict
    torch.save(model.state_dict(), output_d / f'{dataset_name}_model_state_dict.pth')

    # Save loss metrics to a table
    df_loss = pd.DataFrame(loss_record)
    print('loss record:')
    print(df_loss)
    df_loss.to_csv(output_d / 'loss_record.csv', index=False)

    # close tensorboard
    tb_writer.close()

  ##############################################################################

  # Create various output folders
  orig_mat_d = output_d / 'original_matrices'
  orig_contour_d = output_d / 'original_contours'
  preproc_mat_d = output_d / 'preprocessed_matrices'
  preproc_mat_npy_d = output_d / 'preprocessed_matrices_npy'
  preproc_contour_d = output_d / 'preprocessed_contours'
  recon_mat_d = output_d / 'reconstructed_matrices'
  recon_mat_npy_d = output_d / 'reconstructed_matrices_npy'
  recon_contour_d = output_d / 'reconstructed_contours'
  report_d = output_d / 'report'
  orig_mat_d.mkdir(parents=True, exist_ok=True)
  orig_contour_d.mkdir(parents=True, exist_ok=True)
  preproc_mat_d.mkdir(parents=True, exist_ok=True)
  preproc_mat_npy_d.mkdir(parents=True, exist_ok=True)
  preproc_contour_d.mkdir(parents=True, exist_ok=True)
  recon_mat_d.mkdir(parents=True, exist_ok=True)
  recon_mat_npy_d.mkdir(parents=True, exist_ok=True)
  recon_contour_d.mkdir(parents=True, exist_ok=True)
  report_d.mkdir(parents=True, exist_ok=True)

  ##############################################################################

  if clargs.report_only: # skip testing
    print('\n---\nskipping testing\n'+'-'*80)
  else: # training
    print('\n---\ntesting\n'+'-'*80)

    # reporting helper, called once per sample in the test set during testing
    def rpt(og_file, og_lbl, og_m, preproc_m, recon_m, recon_loss):
      fname = pathlib.Path(og_file).stem

      # prepare and gather relevant objects
      og_m = og_m.squeeze().to('cpu')
      preproc_m = preproc_m.squeeze().to('cpu')
      recon_m = recon_m.squeeze().to('cpu')
      if clargs.pre_plot_rescale:
        preproc_m = torch.mul(preproc_m, clargs.pre_plot_rescale)
        recon_m = torch.mul(recon_m, clargs.pre_plot_rescale)
      og_cont = dst_mat_to_coords(asym_to_sym(og_m), clargs.space_dim)
      preproc_cont = dst_mat_to_coords(asym_to_sym(preproc_m), clargs.space_dim)
      recon_cont = dst_mat_to_coords(asym_to_sym(recon_m), clargs.space_dim)

      # Save all test visualisations
      # original recon_m
      plt.imsave(orig_mat_d / f'{fname}_original_dm.pdf', og_m, cmap='viridis')
      # original contour
      plot_points(og_cont, save_path=orig_contour_d / f'{fname}_original_contour.pdf')

      # preprocessed preproc_m
      np.save(preproc_mat_npy_d / f'{fname}_preprocessed_dm.npy', preproc_m)
      np.savetxt(preproc_mat_npy_d / f'{fname}_preprocessed_dm.csv', preproc_m, delimiter=",")
      plt.imsave(preproc_mat_d / f'{fname}_preprocessed_dm.pdf', preproc_m, cmap='viridis')
      # preprocessed contour
      plot_points(preproc_cont, save_path=preproc_contour_d / f'{fname}_preprocessed_contour.pdf')

      # reconstructed recon_m
      np.save(recon_mat_npy_d / f'{fname}_reconstructed_dm.npy', recon_m)
      plt.imsave(recon_mat_d / f'{fname}_reconstructed_dm.pdf', recon_m, cmap='viridis')
      # reconstructed contour
      plot_points(recon_cont, save_path=recon_contour_d / f'{fname}_reconstructed_contour.pdf')

      # all together
      dst_mat_compare_plot( og_m, og_cont, preproc_m, preproc_cont, recon_m, recon_cont
                          , report_d / f'{fname}_report.pdf' )
      return og_file, og_m, preproc_m, recon_m, og_cont, preproc_cont, recon_cont, recon_loss

    # testing #
    ###########
    model.eval()
    summary, res_classify_test, Z, lbls = test_model( model, test_loader
                                                    , n_splits=clargs.number_splits_classify
                                                    , classify_with_scale=clargs.classify_with_scale
                                                    , dev=dev
                                                    , report_callback=rpt
                                                    , summary_samples=clargs.report_summary
                                                    )
    # save test latent space
    print(f'saving test latent space to {output_d / "test_latent_space.npy"}')
    np.save(output_d / 'test_latent_space.npy', Z)
    print(f'saving test labels to {output_d / "test_labels.npy"}')
    np.save(output_d / 'test_labels.npy', lbls)

  # reports #
  ###########

  print('\n---\nreporting\n'+'-'*80)

  # test metrics
  if not clargs.report_only:
    print(f'generating report to {output_d / "run_report.txt"}')
    with open(output_d / 'run_report.txt', 'w') as f:
      if not clargs.skip_training:
        for kind, nm, stat in [ (kk, k, v) for kk, vv in res_classify_val.items()
                                           for k, v in vv.items() ]:
          f.write(f"validation - {kind} - {nm} - mean f1: {stat['f1_mean']:.3f}(std:{stat['f1_std']:.3f})\n")
          f.write(f"validation - {kind} - {nm} - mean log_loss: {stat['log_loss_mean']:.3f}(std:{stat['log_loss_std']:.3f})\n")
      for kind, nm, stat in [ (kk, k, v) for kk, vv in res_classify_test.items()
                                         for k, v in vv.items() ]:
        f.write(f"test - {kind} - {nm} - mean f1: {stat['f1_mean']:.3f}(std:{stat['f1_std']:.3f})\n")
        f.write(f"test - {kind} - {nm} - mean log_loss: {stat['log_loss_mean']:.3f}(std:{stat['log_loss_std']:.3f})\n")
      if not clargs.skip_training:
        f.write(df_loss.to_csv(index=False))
    # extra summary report if required
    print(f'generating summary report to {output_d / "summary_report.pdf"}')
    dst_mat_compare_plot_table(summary, output_d / 'summary_report.pdf')

  # prepare latent space and labels for visualisation
  if clargs.report_only:
    Z = np.load(clargs.report_only[0])
    lbls = np.load(clargs.report_only[1])
  lbls = [test_loader.dataset.dataset.classes[lbl] for lbl in lbls]

  # latent space visualisation
  print(f'generating PCA latent space visualisation to {output_d / "PCA_test_latent_space.pdf"}')
  dimensionality_reduction_plot(Z, lbls, 'PCA', output_d / 'PCA_test_latent_space.pdf')
  for p in [5, 10, 20, 30]:
    print(f'generating tSNE (perplexity {p}) latent space visualisation to {output_d / f"tSNE_p{p}_test_latent_space.pdf"}')
    dimensionality_reduction_plot( Z, lbls, 'tSNE', output_d / f'tSNE_p{p}_test_latent_space.pdf'
                                 , tsne_perplexity=p )
  print(f'generating UMAP latent space visualisation to {output_d / "PCA_test_latent_space.pdf"}')
  dimensionality_reduction_plot(Z, lbls, 'UMAP', output_d / 'UMAP_test_latent_space.pdf')

  if not model_was_trained:
    print( 'No model was trained, skipping mean average shapes and random shape samples'
           ' (if you want to generate these when using --report-only'
           ', please also provide the model weights via --model-weights)' )
  else:
    average_shape( model, Z, lbls, output_d
                 , pre_plot_rescale = clargs.pre_plot_rescale
                 , space_dims = clargs.space_dim
                 , device = dev )
    random_shapes( model, clargs.number_random_samples, ls_sz, output_d
                 , pre_plot_rescale = clargs.pre_plot_rescale
                 , space_dims = clargs.space_dim
                 , device = dev )
    intraclass_trajectories ( model, Z, lbls, output_d
                            , n_sample_steps = 10
                            , show_dm = True
                            , no_start_end = False
                            , pre_plot_rescale = clargs.pre_plot_rescale
                            , space_dims = clargs.space_dim
                            , device = dev )
    interclasses_trajectories ( model, Z, lbls, output_d
                              , n_sample_steps = 10
                              , n_class_pairs = None
                              , show_dm = True
                              , no_start_end = False
                              , pre_plot_rescale = clargs.pre_plot_rescale
                              , space_dims = clargs.space_dim
                              , device = dev )

################################################################################

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="run ShapeEmbedLite")
  dataset = parser.add_mutually_exclusive_group(required=True)
  dataset.add_argument( "--dataset", metavar=("DATASET_NAME", 'DATASET_PATH'), nargs=2
                     , help="dataset to use" )
  dataset.add_argument( "--train-test-dataset", metavar=("DATASET_NAME", 'DATASET_TRAIN_PATH', 'DATASET_TEST_PATH'), nargs=3
                                , help="dataset name and path to train and test sets")
  parser.add_argument( '-o', '--output-dir', metavar="OUTPUT_DIR", type=pathlib.Path
                     , help="path to use for output produced by the run" )
  parser.add_argument( '-d', '--device', metavar="DEV", default=dflt_device
                     , help="torch device to use ('cuda' or 'cpu')" )
  parser.add_argument( '-w', '--model-weights', metavar="WEIGTHS_PATH", type=pathlib.Path
                     , help="Path to weights to load the model with" )
  parser.add_argument( '--skip-training', action=argparse.BooleanOptionalAction, default=False
                     , help=f'skip/do not skip training phase' )
  parser.add_argument( '--report-only', nargs=2, metavar=('TEST_LATENT_SPACE_NPY', 'TEST_LABELS_NPY'), type=pathlib.Path
                     , help=f'skip to the reporting based on provided latent space and labels (no training, no testing)')
  parser.add_argument( '-r', '--learning-rate', metavar="LR", type=float, default=0.001
                     , help="learning rate (default=0.001)" )
  parser.add_argument( '-p', '--reduce-lr-on-plateau', metavar=['[FACTOR]', '[PATIENCE]'], nargs='*'
                     , help="dynamically reduce learning rate on plateau (see https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau)" )
  parser.add_argument( '--space-dim', metavar="SPACE_DIM", type=int, choices=[2, 3], default=2
                     , help="dimension of the data passed to the model (default=2)" )
  parser.add_argument( '--indexation-invariant-loss', action=argparse.BooleanOptionalAction, default=True
                     , help=f'enable/disable use of indexation invariant loss as reconstruction loss')
  parser.add_argument( '--reflection-invariant-loss', action=argparse.BooleanOptionalAction, default=True
                     , help=f'enable/disable use of reflection invariant loss as reconstruction loss')
  parser.add_argument( '-e', '--number-epochs', metavar="N_EPOCHS", type=int, default=100
                     , help="desired number of epochs (default=100)" )
  parser.add_argument( '-n', '--number-splits-classify', metavar="N_SPLITS_CLASSIFY", type=int
                     , help="desired number of splits for classification" )
  parser.add_argument( '-l', '--latent-space-size', metavar="LS_SZ", type=int, default=128
                     , help="desired latent space size (default=128)" )
  parser.add_argument( '-b', '--beta', metavar="BETA", type=float, default=0.0
                     , help="beta parameter, scaling KL loss (default=0.0)" )
  parser.add_argument( '-t', '--tensorboard-log-dir', metavar="TB_LOG_DIR", default='./runs', type=pathlib.Path
                     , help="tensorboard log directory (default='./runs')" )
  parser.add_argument( '--circular-padding', action=argparse.BooleanOptionalAction, default=True
                     , help=f'enable/disable circular padding in the encoder')
  parser.add_argument( '--classify-with-scale', action=argparse.BooleanOptionalAction, default=False
                     , help=f'enable/disable scale preservation in classification')
  norms=['max', 'fro']
  parser.add_argument( '--preprocess-normalize', metavar='NORM', choices=norms
                     , help=f'enable normalization preprocessing layer, one of {norms} (default None)')
  parser.add_argument( '--preprocess-rescale', metavar="SCALAR", type=float
                     , help="rescale (post-normalize if enabled) the input by SCALAR (default None)" )
  parser.add_argument( '--preprocess-augment', action=argparse.BooleanOptionalAction, default=False
                     , help=f'enable/disable augmentation preprocessing layer')
  parser.add_argument( '--number-random-samples', metavar="N_RND_SMLPS", type=int, default=5
                     , help="number of random samples to generate and plot" )
  parser.add_argument( '--pre-plot-rescale', metavar="PLOT_RESCALE", type=float
                     , help="optional scaling factor for result contour plotting" )
  parser.add_argument( '-s', '--report-summary', metavar="RPT_SUMMARY", type=int, default=6
                     , help="desired number of report summary lines" )
  clargs = parser.parse_args()
  main(clargs)
