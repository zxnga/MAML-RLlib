import gymnasium
import numpy as np

class TaskEnvWrapper(gymnasium.Env):
    def __init__(self, base_env, base_task_info, task_generator):
        """
        Wrap an existing environment and add meta-learning functionality.
        
        Args:
            base_env (gym.Env): The underlying environment instance.
            base_task_info: Additional info associated with the task.
            task_generator (TaskGenerator): An instance of your TaskGenerator.
        """
        self.task = base_env
        self.task_info = base_task_info
        self.task_generator = task_generator

        # Forward the action and observation spaces from the base env.
        self.action_space = self.task.action_space
        self.observation_space = self.task.observation_space

    def _ensure_writable_arrays(self):
        """
        Check if the underlying environment has the problematic array and,
        if so, make a writable copy.
        """
        if hasattr(self.task, '_CityLearnEnv__energy_from_cooling_device'):
            arr = self.task._CityLearnEnv__energy_from_cooling_device
            # Make a writable copy
            self.task._CityLearnEnv__energy_from_cooling_device = np.copy(arr)

    def step(self, action):
        # Optionally ensure writable state before stepping.
        self._ensure_writable_arrays()
        return self.task.step(action)

    def reset(self, **kwargs):
        # Ensure that before resetting the environment, we fix the array.
        self._ensure_writable_arrays()
        return self.task.reset(**kwargs)

    def sample_tasks(self, n_tasks=10):
        """
        Generate a list of tasks. Each task is a tuple (env, info).
        
        Args:
            n_tasks (int): Number of tasks to sample.
        
        Returns:
            List of tuples containing (env, info).
        """
        tasks = []
        for i in range(n_tasks):
            task, info, _ = self.task_generator.get_task(meta_step=i)
            tasks.append((task, info))
        return tasks

    def set_task(self, task_and_info):
        """
        Set a new task for the environment. The MAML algorithm calls this method
        to update the environment with a new task.
        """
        self.task, self.task_info = task_and_info
        # Optionally call reset if needed.
        # self.reset()
        
def env_creator(config):
    assert config.get("task_callable") is not None
    assert config.get("task_params") is not None
    
    task_gen = TaskGenerator(
        task_callable=config.get("task_callable"),
        task_callable_params=config.get("task_params", {}),  # Replace with your params
        revisit_start = 50,
    )
    # Create the first task/environment instance.
    base_env, base_info, meta = task_gen.get_task(meta_step=0)
    
    # Wrap it with our TaskEnvWrapper that provides sample_tasks.
    wrapped_env = TaskEnvWrapper(base_env, base_info, task_gen)
    return wrapped_env