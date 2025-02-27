# -*- coding: utf-8 -*-
# Copyright (c) 2019 Uber Technologies, Inc.
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
# ==============================================================================
import logging
import os.path
import uuid
import shutil
from unittest.mock import patch

import pytest

import ray
from ray.tune.sync_client import get_sync_client

from ludwig.backend.ray import RayBackend

from ludwig.hyperopt.execution import (
    RayTuneExecutor, _get_relative_checkpoints_dir_parts)
from ludwig.hyperopt.results import RayTuneResults
from ludwig.hyperopt.run import hyperopt
from ludwig.hyperopt.sampling import (get_build_hyperopt_sampler)
from ludwig.hyperopt.run import update_hyperopt_params_with_defaults
from ludwig.utils.defaults import merge_with_defaults, ACCURACY
from tests.integration_tests.utils import create_data_set_to_use
from tests.integration_tests.utils import binary_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import spawn
from tests.integration_tests.utils import numerical_feature

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)

# Ray mocks

# Dummy sync templates
LOCAL_SYNC_TEMPLATE = "echo {source}/ {target}/"
LOCAL_DELETE_TEMPLATE = "echo {target}"

logger = logging.getLogger(__name__)


def mock_storage_client(path):
    """Mocks storage client that treats a local dir as durable storage."""
    client = get_sync_client(LOCAL_SYNC_TEMPLATE, LOCAL_DELETE_TEMPLATE)
    os.makedirs(path, exist_ok=True)
    client.set_logdir(path)
    return client


HYPEROPT_CONFIG = {
    "parameters": {
        "training.learning_rate": {
            "space": "loguniform",
            "lower": 0.001,
            "upper": 0.1,
        },
        "combiner.num_fc_layers": {
            "space": "randint",
            "lower": 2,
            "upper": 6
        },
        "combiner.num_steps": {
            "space": "grid_search",
            "values": [3,4,5]
        },
    },
    "goal": "minimize"
}


SAMPLERS = [
    {"type": "ray", "num_samples": 2},
    {
        "type": "ray",
        "num_samples": 1,
        "scheduler": {
            "type": "async_hyperband",
            "time_attr": "training_iteration",
            "reduction_factor": 2,
            "dynamic_resource_allocation": True,
        },
    },
    {
        "type": "ray",
        "search_alg": {
            "type": "bohb"
        },
        "scheduler": {
            "type": "hb_bohb",
            "time_attr": "training_iteration",
            "reduction_factor": 4,
        },
        "num_samples": 3
    },
]

EXECUTORS = [
    {"type": "ray"},
]


def _get_config(sampler, executor):
    input_features = [
        numerical_feature(), numerical_feature()
    ]
    output_features = [
        binary_feature()
    ]

    return {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "num_fc_layers": 2},
        "training": {"epochs": 2, "learning_rate": 0.001},
        "hyperopt": {
            **HYPEROPT_CONFIG,
            "executor": executor,
            "sampler": sampler,
        },
    }


class MockRayTuneExecutor(RayTuneExecutor):
    def _get_sync_client_and_remote_checkpoint_dir(self, trial_dir):
        remote_checkpoint_dir = os.path.join(
            self.mock_path, *_get_relative_checkpoints_dir_parts(trial_dir))
        return mock_storage_client(remote_checkpoint_dir), remote_checkpoint_dir


@pytest.fixture
def ray_start_4_cpus():
    address_info = ray.init(num_cpus=4)
    try:
        yield address_info
    finally:
        ray.shutdown()

@pytest.fixture
def ray_mock_dir():
    path = os.path.join(ray._private.utils.get_user_temp_dir(),
                        f"mock-client-{uuid.uuid4().hex[:4]}") + os.sep
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)

@spawn
def run_hyperopt_executor(
    sampler, executor, csv_filename, ray_mock_dir,
    validate_output_feature=False,
    validation_metric=None,
):
    config = _get_config(sampler, executor)

    csv_filename = os.path.join(ray_mock_dir, 'dataset.csv')
    dataset_csv = generate_data(
        config['input_features'], config['output_features'], csv_filename, num_examples=100)
    dataset_parquet = create_data_set_to_use('parquet', dataset_csv)

    config = merge_with_defaults(config)

    hyperopt_config = config["hyperopt"]

    if validate_output_feature:
        hyperopt_config['output_feature'] = config['output_features'][0]['name']
    if validation_metric:
        hyperopt_config['validation_metric'] = validation_metric

    update_hyperopt_params_with_defaults(hyperopt_config)

    parameters = hyperopt_config["parameters"]
    if sampler.get("search_alg", {}).get("type", "") == 'bohb':
        # bohb does not support grid_search search space
        del parameters['combiner.num_steps']

    split = hyperopt_config["split"]
    output_feature = hyperopt_config["output_feature"]
    metric = hyperopt_config["metric"]
    goal = hyperopt_config["goal"]

    hyperopt_sampler = get_build_hyperopt_sampler(
        sampler["type"])(goal, parameters, **sampler)

    hyperopt_executor = MockRayTuneExecutor(
        hyperopt_sampler, output_feature, metric, split, **executor)
    hyperopt_executor.mock_path = os.path.join(ray_mock_dir, "bucket")

    hyperopt_executor.execute(
        config,
        dataset=dataset_parquet,
        backend=RayBackend(processor={'parallelism': 4,}),
        output_directory=ray_mock_dir,
        skip_save_processed_input=True,
        skip_save_unprocessed_output=True
    )


@pytest.mark.distributed
@pytest.mark.parametrize('sampler', SAMPLERS)
@pytest.mark.parametrize('executor', EXECUTORS)
def test_hyperopt_executor(sampler, executor, csv_filename, ray_start_4_cpus, ray_mock_dir):
    run_hyperopt_executor(sampler, executor, csv_filename, ray_mock_dir)


@pytest.mark.distributed
def test_hyperopt_executor_with_metric(csv_filename, ray_start_4_cpus, ray_mock_dir):
    run_hyperopt_executor({"type": "ray", "num_samples": 2},
                          {"type": "ray"},
                          csv_filename,
                          ray_mock_dir,
                          validate_output_feature=True,
                          validation_metric=ACCURACY)


@pytest.mark.distributed
@patch("ludwig.hyperopt.execution.RayTuneExecutor", MockRayTuneExecutor)
def test_hyperopt_run_hyperopt(csv_filename, ray_start_4_cpus, ray_mock_dir):
    input_features = [
        numerical_feature(), numerical_feature()
    ]
    output_features = [
        binary_feature()
    ]

    csv_filename = os.path.join(ray_mock_dir, 'dataset.csv')
    dataset_csv = generate_data(
        input_features, output_features, csv_filename, num_examples=100)
    dataset_parquet = create_data_set_to_use('parquet', dataset_csv)

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "num_fc_layers": 2},
        "training": {"epochs": 4, "learning_rate": 0.001}
    }

    output_feature_name = output_features[0]['name']

    hyperopt_configs = {
        "parameters": {
            "training.learning_rate": {
                "space": "loguniform",
                "lower": 0.001,
                "upper": 0.1,
            },
            output_feature_name + ".fc_size": {
                "space": "randint",
                "lower": 32,
                "upper": 256
            },
            output_feature_name + ".num_fc_layers": {
                "space": "randint",
                "lower": 2,
                "upper": 6
            }
        },
        "goal": "minimize",
        'output_feature': output_feature_name,
        'validation_metrics': 'loss',
        'executor': {'type': 'ray'},
        'sampler': {'type': 'ray', 'num_samples': 2},
        'backend': {'type': 'ray', 'processor': {'parallelism': 4}}
    }

    # add hyperopt parameter space to the config
    config['hyperopt'] = hyperopt_configs
    run_hyperopt(config, dataset_parquet, ray_mock_dir)

@spawn
def run_hyperopt(
        config, rel_path, out_dir,
        experiment_name='ray_hyperopt',
        callbacks=None,
):
    hyperopt_results = hyperopt(
        config,
        dataset=rel_path,
        output_directory=out_dir,
        experiment_name=experiment_name,
        callbacks=callbacks,
    )

    # check for return results
    assert isinstance(hyperopt_results, RayTuneResults)

    # check for existence of the hyperopt statistics file
    assert os.path.isfile(
        os.path.join(out_dir, 'hyperopt_statistics.json')
    )
