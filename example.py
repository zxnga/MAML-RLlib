import argparse
from ray.rllib.examples.env.cartpole_mass import CartPoleMassEnv
from src.maml import meta_train, meta_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run meta-test with configurable parameters.")

    parser.add_argument("--maml_checkpoint_path", type=str, default=None, help="Path to MAML checkpoint")
    parser.add_argument("--env_name", type=str, default="CustomMetaEnv-v0", help="Environment name")
    parser.add_argument("--total_timesteps", type=int, default=10000, help="Total number of timesteps")
    parser.add_argument("--eval_interval", type=int, default=2048, help="Evaluation interval in timesteps")
    parser.add_argument("--num_rollout_workers", type=int, default=2, help="Number of rollout workers")
    parser.add_argument("--framework", type=str, default="torch", help="Deep learning framework to use")
    parser.add_argument("--checkpoint_dir", type=str, default="ppo_checkpoints/", help="Directory to save checkpoints")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples or trials to run")

    args = parser.parse_args()

    meta_test(
        maml_checkpoint_path=args.maml_checkpoint_path,
        env_name=args.env_name,
        env_class=CartPoleMassEnv,
        env_fn=None,
        env_config={},
        total_timesteps=args.total_timesteps,
        eval_interval=args.eval_interval,
        num_rollout_workers=args.num_rollout_workers,
        framework=args.framework,
        checkpoint_dir=args.checkpoint_dir,
        num_samples=args.num_samples
    )
