import os
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.maml import MAMLConfig
from gymnasium.wrappers import TimeLimit
from ray.air.config import RunConfig
from ray.air import CheckpointConfig


from .maml_ppo_trainer import LoadMAMLWeightsCallback

def meta_train(
    env_name="CustomMetaEnv-v0",
    env_class=None,
    env_fn=None,
    env_config={},
    env_limit=None,
    rollout_config=None,    # A dict for rollout parameters (e.g., num_rollout_workers, rollout_fragment_length, batch_mode)
    training_config=None,   # A dict for training parameters (e.g., train_batch_size, lr, inner_adaptation_steps, maml_optimizer_steps)
    max_training_iters=300,  # Maximum training iterations (outer loop updates)
    framework="torch",
    save_checkpoint=True,
    checkpoint_dir="checkpoints/"
):
    ray.init(ignore_reinit_error=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if env_class:
        if env_limit:
            register_env(
            env_name,
            lambda env_cfg: TimeLimit(env_class(**config), max_episode_steps=env_limit),
            )
        else:
            register_env(env_name, lambda config: env_class(**config))
    elif env_fn:
        register_env(env_name, env_fn)

    maml_config_obj = (
        MAMLConfig()
        .environment(env=env_name, env_config=env_config)
        .framework(framework)
    )
    if rollout_config is not None:
        maml_config_obj = maml_config_obj.rollouts(**rollout_config)
    if training_config is not None:
        maml_config_obj = maml_config_obj.training(**training_config)

    maml_config_dict = maml_config_obj.to_dict()

    # Create a Tuner object to run a single fixed trial using the specified configuration.
    tuner = tune.Tuner(
        "MAML",
        param_space=maml_config_dict,
        run_config=tune.RunConfig(
            stop={"training_iteration": max_training_iters},
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=True,
            ),
            verbose=1,
            local_dir = os.path.join(os.getcwd(), "ray_results")
        ),
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            num_samples=1,  # Single trial.
        ),
    )

    # Run the training loop.
    results = tuner.fit()
    best_trial = results.get_best_result(metric="episode_reward_mean", mode="max")
    best_checkpoint = best_trial.checkpoint if save_checkpoint else None

    print(f"Best trial found at: {best_checkpoint}")

    ray.shutdown()
    return tuner, results, best_checkpoint


def meta_test(
    maml_checkpoint_path=None,        # MAML checkpoint path; if None, PPO trains from scratch.
    env_name="CustomMetaEnv-v0",  
    env_class=None,              # Optional: custom env class.
    env_fn=None,                 # Optional: custom env function.
    env_config={},  
    total_timesteps=1_000_000,  
    eval_interval=2048,          # Evaluate (and update) every 2048 timesteps.
    num_rollout_workers=2,  
    framework="torch",
    checkpoint_dir="ppo_checkpoints/",
    num_samples=1                # Run a single trial instance.
):
    # Initialize Ray.
    ray.init(ignore_reinit_error=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Register your environment if an env class or env function is provided.
    if env_class:
        register_env(env_name, lambda config: env_class(**config))
    elif env_fn:
        register_env(env_name, env_fn)
    
    # Calculate rollout_fragment_length so that the total collected timesteps per update ~ eval_interval.
    # total_workers = num_rollout_workers + 1  # local worker + remote rollout workers
    # rollout_fragment_length = eval_interval // total_workers
    
    # Create the PPO configuration.
    config = (
        PPOConfig()
        .environment(env=env_name, env_config=env_config)
        .rollouts(num_rollout_workers=num_rollout_workers,
                  rollout_fragment_length="auto")
        .training(train_batch_size=eval_interval, lr=3e-4, gamma=0.99)
        .framework(framework)
        .evaluation(
            evaluation_interval=1,         # Evaluate after each training iteration.
            evaluation_duration=eval_interval,  # Run evaluation for eval_interval timesteps.
            evaluation_duration_unit="timesteps",
            evaluation_num_workers=1,
        )
        .update_from_dict({"maml_checkpoint_path": "/path/to/weights"})
        .callbacks(LoadMAMLWeightsCallback)
        .resources(num_gpus=1)
        .to_dict()
    )

    # Create the Tuner object.
    tuner = tune.Tuner(
        'PPO',  # Use our custom trainer class.
        param_space=config,
        run_config=RunConfig(
            stop={"timesteps_total": total_timesteps},
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=1,  # Save a checkpoint at every evaluation phase.
                checkpoint_at_end=True,
            ),
            verbose=1,
            local_dir = os.path.join(os.getcwd(), "ray_results")
            
        ),
        tune_config=tune.TuneConfig(
            num_samples=num_samples,  # Only one trial instance.
        ),
    )
    
    # Run the training.
    results = tuner.fit()
    
    # Since we only run one instance, get_best_result returns that single trial.
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    final_checkpoint = best_result.checkpoint
    print(f"Final checkpoint saved at: {final_checkpoint}")
    
    ray.shutdown()
    return tuner, best_result, final_checkpoint