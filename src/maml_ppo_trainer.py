from ray.rllib.algorithms.maml import MAMLConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
                  
class LoadMAMLWeightsCallback(DefaultCallbacks):
    def on_training_start(self, *, trainer, **kwargs):
        maml_checkpoint_path = trainer.config.get("maml_checkpoint_path")
        if maml_checkpoint_path:
            print(f"Loading MAML weights from: {maml_checkpoint_path}")
            maml_trainer = (
                MAMLConfig()
                .environment(
                    env=trainer.config["env"],
                    env_config=trainer.config.get("env_config", {})
                )
                .framework(trainer.config.get("framework", "torch"))
                .build()
            )
            maml_trainer.restore(maml_checkpoint_path)
            maml_weights = maml_trainer.get_weights()
            trainer.set_weights(maml_weights)
            print("MAML weights loaded into trainer.")
        else:
            print("No MAML pretrained weights provided. Using random initialization.")
