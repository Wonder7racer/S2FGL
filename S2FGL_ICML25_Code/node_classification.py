import flgo
import flgo.benchmark.partition as fbp
import flgo.algorithm.S2FGL_algorithm as S2FGL

bmkname = 'benchmark001'
bmk_config = './config_cls.py'
bmk = flgo.gen_benchmark_from_file(bmkname, bmk_config, target_path='.', data_type='graph', task_type='node_classification')
task1_config = {
    'benchmark': bmkname,
    'partitioner': {'name': fbp.NodeLouvainPartitioner, 'para': {'num_clients': 10}}
}
task1 = './my_louvain'

flgo.gen_task(task1_config, task_path=task1)

runner = flgo.init(task1, S2FGL, {'gpu': [0,], 'log_file': True, 'learning_rate': 0.2,
                                    'num_steps': 10, 'batch_size': 128, 'num_rounds': 200, 'proportion': 1.0, 'train_holdout': 0.4,
                                    'local_test': True, 'eval_interval': 1, 'seed': 0})

runner.run()

