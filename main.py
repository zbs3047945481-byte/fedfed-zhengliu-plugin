#input_options()：通常用于读取/生成超参数配置
from src.options import input_options
from src.utils.tools import configure_runtime, get_each_client_data_index, resolve_heterogeneity_options, set_random_seed
from getdata import GetDataSet
from src.fed_server.fedavg import FedAvgTrainer
from src.fed_server.fedavgm import FedAvgMTrainer
from src.fed_server.fedprox import FedProxTrainer
from src.fed_server.fednova import FedNovaTrainer
from src.fed_server.scaffold import ScaffoldTrainer


TRAINER_REGISTRY = {
    'fedavg': FedAvgTrainer,
    'fedavgm': FedAvgMTrainer,
    'fedprox': FedProxTrainer,
    'fednova': FedNovaTrainer,
    'scaffold': ScaffoldTrainer,
}


def main():
    options = input_options()#解析命令行参数，生成配置字典
    options = resolve_heterogeneity_options(options)
    configure_runtime(options)
    set_random_seed(options["seed"])
    dataset = GetDataSet(options["dataset_name"], options)#加载数据集
    options["image_size"] = getattr(dataset, "image_size", options["image_size"])
    options["input_channels"] = getattr(dataset, "input_channels", options["input_channels"])
    options["num_classes"] = getattr(dataset, "num_classes", options["num_classes"])
    options["fedfed_input_channels"] = options["input_channels"]
    options["fedfed_num_classes"] = options["num_classes"]
    #将训练数据分配给多个客户端，返回每个客户端的数据索引列表
    each_client_label_index = get_each_client_data_index(
        dataset.train_label,
        options["num_of_clients"],
        options,
    )
    algorithm = str(options.get('fed_algorithm', 'fedavg')).lower()
    trainer_cls = TRAINER_REGISTRY[algorithm]
    trainer = trainer_cls(options, dataset, each_client_label_index)
    trainer.train()

if __name__ == '__main__':
    main()
