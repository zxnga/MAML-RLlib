# MAML PPO meta-training and meta testing.

1. Meta-training on a given list of task or using a task generator
- using `src.maml.meta_train`
- env has to be serializable
2. Meta-testing on hidden task
- using `src.maml.meta_test`
- can load meta-trained weights or perform single agent RL training