import ml_collections
import torch

def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training =  ml_collections.ConfigDict()
    config.training.batch_size = 128
    config.training.n_iters = 1300001
    config.training.snapshot_freq = 50000
    config.training.log_freq = 50
    config.training.eval_freq = 100
    ## store additional checkpoints for preemption in cloud computing environments
    config.training.snapshot_freq_for_preemption = 10000
    ## produce samples at each snapshot.
    config.training.snapshot_sampling = True
    config.training.likelihood_weighting = False
    config.training.continuous = True
    config.training.reduce_mean = True

    # sampling
    config.sampling = ml_collections.ConfigDict()
    config.sampling.n_steps_each = 1
    config.sampling.noise_removal = True
    config.sampling.probability_flow = False
    config.sampling.snr = 0.16

    # evaluation
    config.eval =  ml_collections.ConfigDict()
    config.eval.begin_ckpt = 8
    config.eval.end_ckpt = 8
    config.eval.batch_size = 2500
    config.eval.enable_sampling = True
    config.eval.num_samples = 50000
    config.eval.enable_loss = False
    config.eval.enable_bpd = False
    config.eval.bpd_dataset = 'test'

    # data
    config.data = ml_collections.ConfigDict()
    config.data.dataset = 'CIFAR10'
    config.data.image_size = 32
    config.data.random_flip = True
    config.data.centered = False
    config.data.uniform_dequantization = False
    config.data.num_channels = 3

    # model
    config.model = ml_collections.ConfigDict()
    config.model.sigma_min = 0.01
    config.model.sigma_max = 50
    config.model.num_scales = 1000
    config.model.beta_min = 0.1
    config.model.beta_max = 20.
    config.model.dropout = 0.1
    config.model.embedding_type = 'fourier'

    # optimization
    config.optim = ml_collections.ConfigDict()
    config.optim.weight_decay = 0
    config.optim.optimizer = 'Adam'
    config.optim.lr = 2e-4
    config.optim.beta1 = 0.9
    config.optim.eps = 1e-8
    config.optim.warmup = 5000
    config.optim.grad_clip = 1.

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config

def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vpsde'
  training.continuous = True
  training.reduce_mean = True
  training.n_iters = 950001

  # sampling
  sampling = config.sampling
  # sampling.method = 'pc'
  # sampling.predictor = 'euler_maruyama'
  # sampling.corrector = 'none'

  sampling.eps = 1e-3
  sampling.method = 'dpm_solver'
  sampling.dpm_solver_method = 'singlestep'
  sampling.dpm_solver_order = 3
  sampling.algorithm_type = 'dpmsolver'
  sampling.thresholding = False
  sampling.noise_removal = False
  sampling.steps = 30
  sampling.skip_type = 'logSNR'
  sampling.rtol = 0.05

  # sampling.method = 'ode'
  # sampling.eps = 1e-4
  # sampling.noise_removal = False
  sampling.rk45_rtol = 1e-5
  sampling.rk45_atol = 1e-5

  # data
  data = config.data
  data.centered = True

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 8
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = False
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'none'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.embedding_type = 'positional'
  model.fourier_scale = 16
  model.conv_size = 3

  return config