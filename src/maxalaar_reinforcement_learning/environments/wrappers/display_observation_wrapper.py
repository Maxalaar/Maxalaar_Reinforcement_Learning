import numpy as np
import cv2
from gymnasium.core import Wrapper


class DisplayObservationWrapper(Wrapper):
    def __init__(self, env, observation_scale_factor=1.0, fps=10, channel_layout="CHW"):
        """
        Displays observations in an OpenCV window
        Args:
            env: Environment to wrap
            observation_scale_factor: Display resizing factor
            fps: Frames per second for display
            channel_layout: Channel organization ('CHW' or 'HWC')
        """
        super().__init__(env)
        self.observation_scale_factor = observation_scale_factor
        self.fps = fps
        self.channel_layout = channel_layout.upper()
        # Handling the environment name
        self.environment_name = "Unknown"
        if hasattr(env, "spec") and env.spec is not None:
            self.environment_name = env.spec.id
        elif hasattr(env, "__class__"):
            self.environment_name = env.__class__.__name__
        # Window setup
        self.window_name = f"Observation: {self.environment_name}"
        self._window_initialized = False
        self._closed = False

    def _process_observation(self, observation):
        """Converts and formats the observation for display"""
        # Convert to uint8 if necessary
        if observation.dtype != np.uint8:
            observation = np.clip(observation, 0, 1)
            observation = (255 * observation).astype(np.uint8)
        # Reorganize channels if necessary
        if self.channel_layout == "HWC" and observation.ndim == 3:
            observation = np.transpose(observation, (2, 0, 1))
        elif observation.ndim == 2:
            observation = observation[np.newaxis, ...]
        return observation

    def _create_channel_grid(self, observation):
        """Creates a grid of channels with borders"""
        C, H, W = observation.shape
        # Grid setup
        n_cols = min(3, C)
        n_rows = (C + n_cols - 1) // n_cols
        border = 2
        # Create grid with white borders
        grid = np.full((
            n_rows * (H + border) - border,
            n_cols * (W + border) - border
        ), 255, dtype=np.uint8)
        # Place channels in the grid
        for i in range(C):
            r, c = i // n_cols, i % n_cols
            y = r * (H + border)
            x = c * (W + border)
            grid[y:y + H, x:x + W] = observation[i]
        return grid

    def _display(self, grid):
        """Displays the image in an OpenCV window"""
        # Resize
        if self.observation_scale_factor != 1.0:
            new_w = int(grid.shape[1] * self.observation_scale_factor)
            new_h = int(grid.shape[0] * self.observation_scale_factor)
            grid = cv2.resize(grid, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # Display
        cv2.imshow(self.window_name, grid)
        cv2.waitKey(int(1000 / self.fps))
        # Manual closing with ESC
        if cv2.waitKey(1) & 0xFF == 27:
            self.close()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not self._closed:
            processed = self._process_observation(obs)
            grid = self._create_channel_grid(processed)
            self._display(grid)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if not self._closed:
            processed = self._process_observation(obs)
            grid = self._create_channel_grid(processed)
            self._display(grid)
        return obs, info

    def close(self):
        if not self._closed:
            cv2.destroyWindow(self.window_name)
            self._closed = True
        return super().close()

    def __del__(self):
        self.close()
