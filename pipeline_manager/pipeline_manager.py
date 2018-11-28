# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

from tqdm import tqdm
from pathlib import Path
from shutil import rmtree
from psutil import cpu_count
from datetime import datetime
from typing import Union, Dict
from os.path import join, isdir
from copy import copy, deepcopy
from multiprocessing import Pool

from pipeline_manager.pipegen import PipeGen
from pipeline_manager.observer import Observer
from pipeline_manager.utils import get_num_gpu
from pipeline_manager.utils import results_visualization, get_available_gpus

from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.prints import RedirectedPrints
from deeppavlov.core.data.data_fitting_iterator import DataFittingIterator
from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.commands.train import read_data_by_config, get_iterator_from_config


def unpack_args(func):
    """
    Decorator that unpacks input arguments from tuple or dict.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(args):
        if isinstance(args, dict):
            return func(**args)
        else:
            return func(*args)
    return wrapper


class PipelineManager:
    """
    The class implements the functions of automatic experiment management. The class accepts a config in the input in
    which the structure of the experiments is described, and additional parameters, which are class attributes.
    Based on this information, a list of deeppavlov configs is created. Experiments can be run sequentially or in
    parallel, both on video cards and on the processor. A special class is responsible for describing and logging
    experiments, their execution time and results. After passing all the experiments based on the logs, a small report
    is created in the form of a xlsx table, and histogram with metrics info. When you start the experiment, you can
    also search for optimal hyperparameters, "grid" and "random" search is available.

    Running a large number of experiments, especially with large neural models, may take a large amount of time, so a
    special test was added to check the correctness of the joints of individual blocks in all pipelines, or another
    errors. During the test, all pipelines are trained on a small piece of the original dataset, if the test passed
    without errors, you can not worry about the experiment, and then a normal experiments is automatically started.
    The test starts automatically, nothing else needs to be done, but it can also be turned off. In this case, the
    experiment will start immediately. Test supports multiprocessing.

    Also you can save checkpoints for all pipelines, or only the best.

    Write about save best

    Args:
        config_path: path to config file, or config dict.

    Attributes:
        exp_name: str, name of the experiment.
        date: str, date of the experiment.
        info: dict with some additional information that you want to add to the log, the content of the dictionary
              does not affect the algorithm
        root: root path, the root path where the report will be generated and saved checkpoints
        plot: boolean trigger, which determines whether to draw a graph of results or not
        save_best: boolean trigger, which determines whether to save all models or only best model
        do_test: boolean trigger, which determines whether to run an experiment test on a small piece of data,
                 before running a full-scale experiment
        search_type: string parameter defining the type of hyperparams search, can be "grid" or "random"
        sample_num: determines the number of generated pipelines, if parameter search_type == "random"
        target_metric: The metric name on the basis of which the results will be sorted when the report
                       is generated. The default value is None, in this case the target metric is taken the
                       first name from those names that are specified in the config file. If the specified metric
                       is not contained in DeepPavlov will be called error.
        multiprocessing: boolean trigger, determining the run mode of the experiment.
        max_num_workers_: upper limit on the number of workers if experiment running in multiprocessing mode
        use_all_gpus: boolean trigger, if True the pipeline manager automatically considers all available to the user
                      graphics cards (CUDA_VISIBLE_DEVICES is is taken into account). And selects as available only
                      those that meet the memory criterion. If the memory of a video card is occupied by more than
                      "X" percent, then the video card is considered inaccessible, and when the experiment is started,
                      the models will not start on it. For the value of the parameter "X" is responsible
                      "memory_fraction" attribute. Parameters "use_all_gpus" and "use_multi_gpus" can not be not
                      None simultaneously.
        use_multi_gpus: None or List[ints], list with numbers of video cards available for use. All cards from the list
                        are checked for availability by memory criterion.If the memory of a video card is occupied by
                        more than "X" percent, then the video card is considered inaccessible, and when the experiment
                        is started, the models will not start on it. For the value of the parameter "X" is responsible
                        "memory_fraction" attribute. If part of the video cards are busy, then only the remaining cards
                        from the presented list will be used. If all of the presented video cards are busy, an error
                        message will appear. If "use_multi_gpus" if not None, then "use_all_gpus" must be False.
        memory_fraction: the parameter determines the criterion of whether the gpu card is free or not.
                         If memory_fraction == 1.0 only those cards whose memory is completely free will be
                         considered as available. If memory_fraction == 0.5 cards with no more than half of the memory
                         will be considered as available.
        available_gpu: list with numbers of available gpu cards
        save_path: path to the save folder
        observer: A special class that collects auxiliary statistics and results during training, and stores all
                the collected data in a separate log.
        pipeline_generator: A special class that generates configs for training.
        gen_len: amount of pipelines in experiment

    """
    def __init__(self, config_path: Union[str, Dict, Path]):
        """
        Initialize observer, read input args, builds a directory tree, initialize date, start test of experiment on
        tiny data.
        """
        if isinstance(config_path, str):
            self.exp_config = read_json(config_path)
        elif isinstance(config_path, Path):
            self.exp_config = read_json(config_path)
        else:
            self.exp_config = config_path

        self.exp_name = self.exp_config['enumerate'].get('exp_name', 'experiment')
        self.date = self.exp_config['enumerate'].get('date', datetime.now().strftime('%Y-%m-%d'))
        self.info = self.exp_config['enumerate'].get('info')
        self.root = self.exp_config['enumerate'].get('root', 'download/experiments/')
        self.plot = self.exp_config['enumerate'].get('plot', False)
        self.save_best = self.exp_config['enumerate'].get('save_best', False)
        self.do_test = self.exp_config['enumerate'].get('do_test', False)

        # self.cross_validation = self.exp_config['enumerate'].get('cross_val', False)
        # self.k_fold = self.exp_config['enumerate'].get('k_fold', 5)

        self.search_type = self.exp_config['enumerate'].get('search_type', 'random')
        self.sample_num = self.exp_config['enumerate'].get('sample_num', 10)
        self.target_metric = self.exp_config['enumerate'].get('target_metric')
        self.multiprocessing = self.exp_config['enumerate'].get('multiprocessing', True)
        self.max_num_workers_ = self.exp_config['enumerate'].get('max_num_workers')
        self.use_all_gpus = self.exp_config['enumerate'].get('use_all_gpus', False)
        self.use_multi_gpus = self.exp_config['enumerate'].get('use_multi_gpus')
        self.memory_fraction = self.exp_config['enumerate'].get('gpu_memory_fraction', 1.0)
        self.max_num_workers = None
        self.available_gpu = None

        # create the observer
        self.save_path = join(self.root, self.date, self.exp_name, 'checkpoints')
        self.observer = Observer(self.exp_name, self.root, self.info, self.date, self.plot)
        # create the pipeline generator
        self.pipeline_generator = PipeGen(self.exp_config, self.save_path, self.search_type, self.sample_num, False)
        self.gen_len = self.pipeline_generator.length
        # write train data in observer
        self.observer.log['experiment_info']['number_of_pipes'] = self.gen_len
        if self.target_metric:
            self.observer.log['experiment_info']['target_metric'] = self.target_metric
        else:
            self.observer.log['experiment_info']['metrics'] = copy(self.exp_config['train']['metrics'])
            if isinstance(self.exp_config['train']['metrics'][0], dict):
                self.observer.log['experiment_info']['target_metric'] = \
                    copy(self.exp_config['train']['metrics'][0]['name'])
            else:
                self.observer.log['experiment_info']['target_metric'] = copy(self.exp_config['train']['metrics'][0])

        # multiprocessing
        if self.use_multi_gpus and self.use_all_gpus:
            raise ValueError("Parameters 'use_all_gpus' and 'use_multi_gpus' can not simultaneously be not None.")

        if self.multiprocessing:
            self.prepare_multiprocess()

        # write time of experiment start
        self.start_exp = time.time()
        # start test
        if self.do_test:
            self.test()

    def prepare_multiprocess(self):
        """
        Calculates the number of workers and the set of available video cards, if gpu is used, based on init attributes.
        """
        cpu_num = cpu_count()
        gpu_num = get_num_gpu()

        try:
            visible_gpu = [int(q) for q in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
        except KeyError:
            visible_gpu = []

        if self.max_num_workers_ is not None and self.max_num_workers_ < 1:
            raise ConfigError("The number of workers must be at least equal to one. "
                              "Please check 'max_num_workers' parameter in config.")

        if self.use_all_gpus:
            if self.max_num_workers_ is None:
                self.available_gpu = get_available_gpus(gpu_fraction=self.memory_fraction)
                if len(visible_gpu) != 0:
                    self.available_gpu = list(set(self.available_gpu) & set(visible_gpu))

                if len(self.available_gpu) == 0:
                    raise ValueError("GPU with numbers: ({}) are busy.".format(set(visible_gpu)))
                elif len(self.available_gpu) < len(visible_gpu):
                    print("PipelineManagerWarning: 'CUDA_VISIBLE_DEVICES' = ({0}), "
                          "but only {1} are available.".format(visible_gpu, self.available_gpu))

                if int(cpu_num * 0.7) > len(self.available_gpu):
                    self.max_num_workers = len(self.available_gpu)
                else:
                    self.max_num_workers = int(cpu_num * 0.7)
            else:
                if self.max_num_workers_ > gpu_num:
                    self.max_num_workers = gpu_num
                    self.available_gpu = get_available_gpus(gpu_fraction=self.memory_fraction)
                    if len(visible_gpu) != 0:
                        self.available_gpu = list(set(self.available_gpu) & set(visible_gpu))

                    if len(self.available_gpu) == 0:
                        raise ValueError("GPU with numbers: ({}) are busy.".format(set(visible_gpu)))
                else:
                    self.available_gpu = get_available_gpus(num_gpus=self.max_num_workers_,
                                                            gpu_fraction=self.memory_fraction)
                    if len(visible_gpu) != 0:
                        self.available_gpu = list(set(self.available_gpu) & set(visible_gpu))

                    if len(self.available_gpu) == 0:
                        raise ValueError("GPU with numbers: ({}) are busy.".format(set(visible_gpu)))

                    self.max_num_workers = len(self.available_gpu)

        elif self.use_multi_gpus:
            if len(visible_gpu) != 0:
                self.use_multi_gpus = list(set(self.use_multi_gpus) & set(visible_gpu))

            if len(self.use_multi_gpus) == 0:
                raise ValueError("GPU numbers in 'use_multi_gpus' and 'CUDA_VISIBLE_DEVICES' "
                                 "has not intersections".format(set(visible_gpu)))

            self.available_gpu = get_available_gpus(gpu_select=self.use_multi_gpus, gpu_fraction=self.memory_fraction)

            if len(self.available_gpu) == 0:
                raise ValueError("All GPU from 'use_multi_gpus' are busy. "
                                 "GPU numbers ({});".format(self.use_multi_gpus))

            if not self.max_num_workers_:
                self.max_num_workers = len(self.available_gpu)
            else:
                if self.max_num_workers_ > len(self.available_gpu):
                    self.max_num_workers = len(self.available_gpu)
                else:
                    self.max_num_workers = self.max_num_workers_
                    self.available_gpu = self.available_gpu[0:self.max_num_workers_]

        else:
            if self.max_num_workers_ is not None and self.max_num_workers_ > cpu_num:
                print("PipelineManagerWarning: parameter 'max_num_workers'={0}, "
                      "but amounts of cpu is {1}. The {1} will be assigned to 'max_num_workers' "
                      "as default.".format(self.max_num_workers_, cpu_num))
                self.max_num_workers_ = cpu_num

            self.max_num_workers = self.max_num_workers_

    @staticmethod
    @unpack_args
    def train_pipe(pipe, i, observer, gpu_ind=None):
        """
        Start learning single pipeline. Observer write all info in log file.

        Args:
            pipe: config dict of pipeline
            i:  number of pipeline
            observer: link to observer object
            gpu_ind: number of gpu to use (if multiprocessing is True)

        """
        # modify project environment
        if gpu_ind:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ind)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

        observer.pipe_ind = i + 1
        observer.pipe_conf = copy(pipe['chainer']['pipe'])
        dataset_name = copy(pipe['dataset_reader']['data_path'])
        observer.dataset = copy(pipe['dataset_reader']['data_path'])
        observer.batch_size = copy(pipe['train'].get('batch_size', "None"))

        # start pipeline time
        pipe_start = time.time()
        save_path = join(observer.save_path, dataset_name, "pipe_{}".format(i + 1))
        if not isdir(save_path):
            os.makedirs(save_path)

        # run pipeline train with redirected output flow
        with RedirectedPrints(new_target=open(join(save_path, "out.txt"), "w")):
            results = train_evaluate_model_from_config(pipe, to_train=True, to_validate=True)

        # add results and pipe time to log
        observer.pipe_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - pipe_start))
        observer.pipe_res = results

        # update logger
        observer.update_log()

        # save config in checkpoint folder
        observer.save_config(pipe, dataset_name, i + 1)
        return None

    def gpu_gen(self, gpu=False):
        """
        Create generator that returning tuple of args fore self.train_pipe method.

        Args:
            gpu: boolean trigger, determine to use gpu or not

        """
        if gpu:
            for i, pipe_conf in enumerate(self.pipeline_generator()):
                gpu_ind = i - (i // len(self.available_gpu)) * len(self.available_gpu)
                yield (deepcopy(pipe_conf), i, self.observer, gpu_ind)
        else:
            for i, pipe_conf in enumerate(self.pipeline_generator()):
                yield (deepcopy(pipe_conf), i, self.observer)

    def _run(self):
        """
        Run the experiment. Creates a report after the experiments.
        """
        # Start generating pipelines configs
        print('[ Experiment start - {0} pipes, will be run]'.format(self.gen_len))

        if self.multiprocessing:
            # start multiprocessing
            workers = Pool(self.max_num_workers)

            if self.available_gpu is None:
                x = list(tqdm(workers.imap_unordered(self.train_pipe, [x for x in self.gpu_gen(gpu=False)]),
                              total=self.gen_len))
                workers.close()
                workers.join()
            else:
                x = list(tqdm(workers.imap_unordered(self.train_pipe, [x for x in self.gpu_gen(gpu=True)]),
                              total=self.gen_len))
                workers.close()
                workers.join()
            del x
        else:
            for i, pipe in enumerate(tqdm(self.pipeline_generator(), total=self.gen_len)):
                if self.available_gpu is None:
                    self.train_pipe((pipe, i, self.observer))
                else:
                    gpu_ind = self.available_gpu[0]
                    self.train_pipe((pipe, i, self.observer, gpu_ind))

        # save log
        self.observer.exp_time(time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start_exp)))
        # delete all checkpoints and save only best pipe
        if self.save_best:
            self.observer.save_best_pipe()

        print("[ End of experiment ]")
        # visualization of results
        print("[ Create an experiment report ... ]")
        results_visualization(join(self.root, self.date, self.exp_name), self.plot)
        print("[ Report created ]")
        return None

    def run(self):
        try:
            self._run()
        except KeyboardInterrupt:
            # save log
            self.observer.exp_time(time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start_exp)))

            print("[ The experiment was interrupt]")
            # visualization of results
            print("[ Create an intermediate report ... ]")
            results_visualization(join(self.root, self.date, self.exp_name), False)
            print("[ The intermediate report was created ]")

        return None

    def _test(self):
        """
        Run a test experiment on a small piece of data. The test supports multiprocessing.
        """
        def gpu_gen(pipe_gen, available_gpu, gpu=False):
            if gpu:
                for j, pipe_conf in enumerate(pipe_gen()):
                    gpu_ind_ = j - (j // len(available_gpu)) * len(available_gpu)
                    yield (j, deepcopy(pipe_conf), gpu_ind_)
            else:
                for j, pipe_conf in enumerate(pipe_gen()):
                    yield (j, deepcopy(pipe_conf))

        # create the pipeline generator
        pipeline_generator = PipeGen(self.exp_config, self.save_path, self.search_type, self.sample_num, True)
        len_gen = pipeline_generator.length

        # Start generating pipelines configs
        print('[ Test start - {0} pipes, will be run]'.format(len_gen))
        if self.multiprocessing:
            # start multiprocessing
            workers = Pool(self.max_num_workers)

            if self.available_gpu:
                x = list(tqdm(workers.imap_unordered(self.test_pipe,
                                                     [x for x in gpu_gen(pipeline_generator,
                                                                         self.available_gpu,
                                                                         gpu=True)]),
                              total=len_gen))
                workers.close()
                workers.join()
            else:
                x = list(tqdm(workers.imap_unordered(self.test_pipe,
                                                     [x for x in gpu_gen(pipeline_generator,
                                                                         self.available_gpu,
                                                                         gpu=False)]),
                              total=len_gen))
                workers.close()
                workers.join()
            del x
        else:
            for i, pipe in enumerate(tqdm(pipeline_generator(), total=len_gen)):
                if self.available_gpu:
                    gpu_ind = i - (i // len(self.available_gpu)) * len(self.available_gpu)
                    self.test_pipe((i, pipe, gpu_ind))
                else:
                    self.test_pipe((i, pipe))

        # del all tmp files in save path
        rmtree(join(self.save_path, "tmp"))
        print('[ The test was successful ]')
        return None

    def test(self):
        try:
            self._test()
        except KeyboardInterrupt:
            # del all tmp files in save path
            rmtree(join(self.save_path, "tmp"))
            print('[ The test was interrupt ]')
            return None

    @staticmethod
    @unpack_args
    def test_pipe(ind, pipe_conf, gpu_ind=None):
        """
        Start testing single pipeline.

        Args:
            ind: pipeline number
            pipe_conf: pipeline config as dict
            gpu_ind: number of gpu card

        Returns:
            None

        """
        def test_dataset_reader_and_iterator(config, i):
            """
            Creating a test iterator with small peace of train dataset. Config and data validation.

            Args:
                config: pipeline config as dict
                i: number of pipeline

            Returns:
                iterator

            """
            # create and test data generator and data iterator
            dataset_composition_ = dict(train=False, valid=False, test=False)
            data = read_data_by_config(config)
            if i == 0:
                for dtype in dataset_composition_.keys():
                    if len(data.get(dtype, [])) != 0:
                        dataset_composition_[dtype] = True
            else:
                for dtype in dataset_composition_.keys():
                    if len(data.get(dtype, [])) == 0 and dataset_composition_[dtype]:
                        raise ConfigError("The file structure in the {0} dataset differs "
                                          "from the rest datasets.".format(config['dataset_reader']['data_path']))

            iterator = get_iterator_from_config(config, data)

            if isinstance(iterator, DataFittingIterator):
                raise ConfigError("Instance of a class 'DataFittingIterator' is not supported.")
            else:
                if config.get('train', None):
                    if config['train']['test_best'] and len(iterator.data['test']) == 0:
                        raise ConfigError(
                            "The 'test' part of dataset is empty, but 'test_best' in train config is 'True'."
                            " Please check the dataset_iterator config.")

                    if (config['train']['validate_best'] or config['train'].get('val_every_n_epochs', False) > 0) and \
                            len(iterator.data['valid']) == 0:
                        raise ConfigError(
                            "The 'valid' part of dataset is empty, but 'valid_best' in train config is 'True'"
                            " or 'val_every_n_epochs' > 0. Please check the dataset_iterator config.")
                else:
                    if len(iterator.data['test']) == 0:
                        raise ConfigError("The 'test' part of dataset is empty as a 'train' part of config file, "
                                          "but default value of 'test_best' is 'True'. "
                                          "Please check the dataset_iterator config.")

            # get a tiny data from dataset
            if len(iterator.data['train']) <= 100:
                print("!!!!!!!!!!!!! WARNING !!!!!!!!!!!!! Length of 'train' part dataset <= 100. "
                      "Please check the dataset_iterator config")
                tiny_train = copy(iterator.data['train'])
            else:
                tiny_train = copy(iterator.data['train'][:10])
            iterator.train = tiny_train

            if len(iterator.data['valid']) <= 20:
                tiny_valid = copy(iterator.data['valid'])
            else:
                tiny_valid = copy(iterator.data['valid'][:5])
            iterator.valid = tiny_valid

            if len(iterator.data['test']) <= 20:
                tiny_test = copy(iterator.data['test'])
            else:
                tiny_test = copy(iterator.data['test'][:5])
            iterator.test = tiny_test

            iterator.data = {'train': tiny_train,
                             'valid': tiny_valid,
                             'test': tiny_test,
                             'all': tiny_train + tiny_valid + tiny_test}

            return iterator

        # modify project environment
        if gpu_ind:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ind)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

        data_iterator_i = test_dataset_reader_and_iterator(pipe_conf, ind)
        results = train_evaluate_model_from_config(pipe_conf,
                                                   iterator=data_iterator_i,
                                                   to_train=True,
                                                   to_validate=False)
        del results
        return None