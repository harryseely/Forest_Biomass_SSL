import os
from ray import train, tune
from ray.tune import CLIReporter

from utils.train_model import train_fn



def ray_parallel(search_space,
                 static_cfg,
                 apply_to_test_set=True,
                 resources_per_trial=None,
                 n_concurrent_trials=1,
                 time_budget_s=60 * 60,
                 num_samples=-1):
    """
    Trains multiple models with different configurations using Ray Tune.

    :param search_space: dictionary containing hyperparameters to tune
    :param static_cfg: dictionary containing hyperparameters that are not being tuned (i.e., static config)
    :param apply_to_test_set: bool to determine if model is applied to test dataset after training
    :param resources_per_trial: dictionary containing the resources to use per trial (e.g., {"cpu": 1, "gpu": 1})
    :param n_concurrent_trials: number of concurrent trials to run
    :param time_budget_s: time in seconds to run the trials (-1 for infinite)
    :param num_samples: number of runs to perform (-1 for infinite)

    :return:
    """

    #Turn off ray logging
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

    # Establish resources per trial
    if resources_per_trial is None:
        resources_per_trial = {"cpu": 1, "gpu": 1}

    # Set up how progress is printed
    reporter = CLIReporter(
        parameter_columns=list(search_space.keys()),
        metric_columns=["val_loss", "train_loss", "training_iteration"],
    )

    # Insert parameters for trainable function
    trainable_with_params = tune.with_parameters(train_fn, static_cfg=static_cfg, apply_to_test_set=apply_to_test_set)

    tuner = tune.Tuner(
        tune.with_resources(trainable_with_params, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            max_concurrent_trials=n_concurrent_trials,
            scheduler=None,
            num_samples=num_samples,
            time_budget_s=time_budget_s,
            trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
        ),
        run_config=train.RunConfig(
            progress_reporter=reporter,

        ),
        param_space=search_space,
    )

    tuner.fit()
