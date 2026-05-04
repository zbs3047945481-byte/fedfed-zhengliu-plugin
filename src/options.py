#定义并解析联邦学习实验的所有超参数，把它们整理成一个 dict（options），供整个系统使用。
import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


#argparse 是 Python 的命令行参数解析库
#它允许你定义命令行参数，并自动解析这些参数，将它们转换为 Python 字典。
def input_options():
    parser = argparse.ArgumentParser()
    #•	创建一个“参数解析器”
	#•	后面所有 add_argument 都是在往这个解析器里注册参数
    # iid
    parser.add_argument('-is_iid', type=str2bool, default=False, help='data distribution is iid.')
    #是否使用 IID 数据分布
    
    parser.add_argument('--dataset_name', type=str, default='mnist', help='name of dataset.')
    #数据集名称 / 数据划分方式标识
    #mnist_dir_0.1 很可能表示：MNIST、Dirichlet α=0.1（Non-IID 程度）

    parser.add_argument('--model_name', type=str, default='mnist_cnn', help='the model to train')
    #指定用哪个模型
    #if model_name == 'mnist_cnn': model = MnistCNN()

    parser.add_argument('--gpu', type=str2bool, default=True, help='gpu id to use')
    #是否使用 GPU
    
    parser.add_argument('--round_num', type=int, default=150, help='number of round in comm')
    #通信轮数：每一轮 = 一次 FedAvg 聚合
    
    parser.add_argument('--num_of_clients', type=int, default=20, help='numer of the clients')
    #联邦系统中的客户端总数 K
    
    parser.add_argument('--c_fraction', type=float, default=0.2,
                        help='C fraction, 0 means 1 client, 1 means total clients')
    #每一轮参与训练的客户端比例
    
    parser.add_argument('--local_epoch', type=int, default=5, help='local train epoch')
    #每个客户端本地训练多少个 epoch
    #比如有1000条数据，把这1000个数据全部用来算一次梯度并更新参数，这完整一轮叫一个epoch

    parser.add_argument('--batch_size', type=int, default=64, help='local train batch size')
    #本地训练 batch size
    #每次梯度更新用多少条样本
    #比如有1000条数据，每轮训练只取100条数据来算梯度并更新参数，这100条数据叫一个batch
    parser.add_argument('--dataloader_num_workers', type=int, default=2,
                        help='Number of DataLoader worker processes.')
    parser.add_argument('--dataloader_pin_memory', type=str2bool, default=True,
                        help='Whether DataLoader should pin host memory when using GPU.')
    parser.add_argument('--torch_cudnn_benchmark', type=str2bool, default=True,
                        help='Enable cudnn benchmark for faster fixed-shape GPU training.')
    parser.add_argument('--early_stop_enable', type=str2bool, default=False,
                        help='Stop training when the global test accuracy has plateaued.')
    parser.add_argument('--early_stop_min_rounds', type=int, default=0,
                        help='Minimum communication rounds before early stopping is allowed.')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='Number of evaluated rounds without meaningful improvement before stopping.')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                        help='Minimum absolute accuracy improvement required to reset early-stop patience.')

    parser.add_argument("--lr", type=float, default=0.001, help="learning rate, \
                        use value from origin paper as default")
    #本地学习率
    
    parser.add_argument('--gn0', type=int, default=1, help='gno')
    
    parser.add_argument('--seed', help='seed for randomness;', type=int, default=3001)
    #随机种子：控制数据划分、模型初始化，方便复现实验
    parser.add_argument('--experiment_tag', type=str, default='',
                        help='Optional tag appended to experiment output folder names.')
    
    parser.add_argument('--weight_decay', help='weight_decay;', type=int, default=1)
    #权重衰减：防止过拟合

    # ---------- Heterogeneity modeling ----------
    parser.add_argument('--partition_strategy', type=str, default='dirichlet',
                        choices=['iid', 'dirichlet'],
                        help='Client data partition strategy.')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.3,
                        help='Dirichlet alpha for label skew. Smaller means stronger heterogeneity.')
    parser.add_argument('--unify_heterogeneity_alpha', type=str2bool, default=True,
                        help='Whether to let dirichlet_alpha jointly control label, quantity, and feature heterogeneity.')
    parser.add_argument('--min_samples_per_client', type=int, default=32,
                        help='Ensure each client owns at least this many training samples.')
    parser.add_argument('--enable_quantity_skew', type=str2bool, default=False,
                        help='Whether to vary client dataset sizes.')
    parser.add_argument('--quantity_skew_beta', type=float, default=0.5,
                        help='Dirichlet beta for client quantity skew. Smaller means more imbalance.')
    parser.add_argument('--enable_feature_skew', type=str2bool, default=False,
                        help='Whether to apply client-specific feature shift/noise.')
    parser.add_argument('--feature_alpha_anchor', type=float, default=0.1,
                        help='Reference alpha that corresponds to the strongest feature skew when alpha is unified.')
    parser.add_argument('--feature_max_scale_delta', type=float, default=0.15,
                        help='Maximum multiplicative scale deviation from 1.0 under the strongest unified feature skew.')
    parser.add_argument('--feature_max_bias_std', type=float, default=0.05,
                        help='Maximum additive bias std under the strongest unified feature skew.')
    parser.add_argument('--feature_max_noise_std', type=float, default=0.05,
                        help='Maximum additive Gaussian noise std under the strongest unified feature skew.')
    parser.add_argument('--feature_noise_std', type=float, default=0.05,
                        help='Std of client-specific additive Gaussian noise.')
    parser.add_argument('--feature_scale_low', type=float, default=0.85,
                        help='Lower bound of client-specific multiplicative scale.')
    parser.add_argument('--feature_scale_high', type=float, default=1.15,
                        help='Upper bound of client-specific multiplicative scale.')
    parser.add_argument('--feature_bias_std', type=float, default=0.05,
                        help='Std of client-specific additive bias.')

    # ---------- FedFed image-space feature distillation plugin ----------
    parser.add_argument('--plugin_name', type=str, default='none',
                        choices=['none', 'fedfed_image'],
                        help='Plugin selector. Use none for FedAvg and fedfed_image for FedAvg+FedFed.')
    parser.add_argument('--fedfed_input_channels', type=int, default=3,
                        help='Input channels used by image-space FedFed generator.')
    parser.add_argument('--fedfed_lambda_fd', type=float, default=2.0,
                        help='Weight of CE(f(x - q(x)), y) for image-space FedFed feature distillation.')
    parser.add_argument('--fedfed_two_stage', type=str2bool, default=True,
                        help='Run paper-style FedFed: feature distillation first, then FedAvg over local plus shared features.')
    parser.add_argument('--fedfed_distill_rounds', type=int, default=15,
                        help='Communication rounds used for the feature distillation stage.')
    parser.add_argument('--fedfed_distill_local_epoch', type=int, default=1,
                        help='Local epochs used in each feature distillation round.')
    parser.add_argument('--fedfed_vae_latent_channels', type=int, default=32,
                        help='Latent channel width of the image-space beta-VAE generator.')
    parser.add_argument('--fedfed_vae_z_dim', type=int, default=2048,
                        help='Flattened latent dimension used by the paper beta-VAE generator.')
    parser.add_argument('--fedfed_generator_type', type=str, default='paper_beta_vae',
                        choices=['paper_beta_vae'],
                        help='Image-space generator type used for q(x).')
    parser.add_argument('--fedfed_lambda_recon', type=float, default=5.0,
                        help='Weight of reconstruction loss that keeps q(x) close to x.')
    parser.add_argument('--fedfed_vae_curriculum', type=str2bool, default=True,
                        help='Use FedFed paper curriculum: early reconstruction weight is 10 * VAE_re.')
    parser.add_argument('--fedfed_beta_kl', type=float, default=0.005,
                        help='KL weight for the beta-VAE generator.')
    parser.add_argument('--fedfed_lambda_x_ce', type=float, default=0.4,
                        help='Weight of CE(f(x), y) used by the paper FedFed VAE objective.')
    parser.add_argument('--fedfed_use_augmentation', type=str2bool, default=True,
                        help='Use paper-style VAE-stage augmentation: crop/flip, mixup classifier warmup, and mosaic VAE warmup.')
    parser.add_argument('--fedfed_mixup_alpha', type=float, default=2.0,
                        help='Mixup alpha used when training the distillation classifier in the VAE stage.')
    parser.add_argument('--fedfed_mosaic_batch_size', type=int, default=64,
                        help='Number of mosaic samples generated per VAE augmentation batch. 0 means use current batch size.')
    parser.add_argument('--fedfed_distill_optimizer', type=str, default='adamw',
                        choices=['adamw', 'sgd', 'adam'],
                        help='Optimizer for the feature distillation classifier and generator.')
    parser.add_argument('--fedfed_distill_lr', type=float, default=0.001,
                        help='Learning rate for the feature distillation optimizer.')
    parser.add_argument('--fedfed_distill_weight_decay', type=float, default=1e-6,
                        help='Weight decay for the feature distillation optimizer.')
    parser.add_argument('--fedfed_upload_per_class', type=int, default=0,
                        help='Max sensitive samples uploaded by one client for each class. 0 means full dataset.')
    parser.add_argument('--fedfed_upload_per_client', type=int, default=0,
                        help='Max sensitive samples uploaded by one client. 0 means full dataset.')
    parser.add_argument('--fedfed_shared_buffer_size', type=int, default=0,
                        help='Max server-side shared sensitive samples. 0 means full shared dataset.')
    parser.add_argument('--fedfed_shared_per_class_size', type=int, default=0,
                        help='Max server-side shared samples per class. 0 means no FIFO truncation.')
    parser.add_argument('--fedfed_shared_batch_size', type=int, default=0,
                        help='Batch size sampled for each shared view. 0 means match local batch size.')
    parser.add_argument('--fedfed_num_classes', type=int, default=10,
                        help='Number of dataset classes.')
    parser.add_argument('--diagnostic_epochs', type=int, default=10,
                        help='Epochs per classifier in x/x_s/x_r diagnostic evaluation.')
    parser.add_argument('--diagnostic_train_limit', type=int, default=20000,
                        help='Max global train samples used by the diagnostic classifiers.')
    parser.add_argument('--diagnostic_test_limit', type=int, default=10000,
                        help='Max test samples used by the diagnostic classifiers.')
    
    args = parser.parse_args()
    #从命令行读取参数
    
    options = args.__dict__
    #把参数对象转成字典
    if options['is_iid']:
        options['partition_strategy'] = 'iid'
    if str(options['dataset_name']).lower() in {'cifar10', 'cifar-10'} and options['model_name'] == 'mnist_cnn':
        options['model_name'] = 'cifar_resnet18'

    return options

#Batch size 决定“每次怎么学”，
#Epoch 决定“学多久”，
#Seed 决定“能不能复现”。
