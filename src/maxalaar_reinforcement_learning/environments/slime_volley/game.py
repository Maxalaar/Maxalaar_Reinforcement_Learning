from typing import Optional

import numpy as np

from maxalaar_reinforcement_learning.environments.slime_volley.agent import Agent
from maxalaar_reinforcement_learning.environments.slime_volley.slime_volley_utilities import Wall, Particle, DelayScreen, create_canvas


class Game:
    def __init__(self, environment, np_random=np.random):
        self.environment = environment
        self.ball = None
        self.ground = None
        self.fence = None
        self.fenceStub = None
        self.agent_left: Optional[Agent] = None
        self.agent_right: Optional[Agent] = None
        self.delayScreen = None
        self.np_random = np_random
        self.reset()

    def reset(self):
        self.ground = Wall(self.environment, 0, 0.75, self.environment.REF_W, self.environment.REF_U, self.environment.GROUND_COLOR)
        self.fence = Wall(self.environment, 0, 0.75 + self.environment.REF_WALL_HEIGHT / 2, self.environment.REF_WALL_WIDTH, self.environment.REF_WALL_HEIGHT - 1.5, self.environment.FENCE_COLOR)
        self.fenceStub = Particle(self.environment, 0, self.environment.REF_WALL_HEIGHT, 0, 0, self.environment.REF_WALL_WIDTH / 2, self.environment.FENCE_COLOR)
        ball_vx = self.np_random.uniform(low=-20, high=20)
        ball_vy = self.np_random.uniform(low=10, high=25)
        self.ball = Particle(self.environment, 0, self.environment.REF_W / 4, ball_vx, ball_vy, 0.5, self.environment.BALL_COLOR)
        self.agent_left = Agent(self.environment, -1, -self.environment.REF_W / 4, 1.5, self.environment.AGENT_LEFT_COLOR)
        self.agent_right = Agent(self.environment, 1, self.environment.REF_W / 4, 1.5, self.environment.AGENT_RIGHT_COLOR)
        self.agent_left.updateState(self.ball, self.agent_right)
        self.agent_right.updateState(self.ball, self.agent_left)
        self.delayScreen = DelayScreen(self.environment)

    def newMatch(self):
        ball_vx = self.np_random.uniform(low=-20, high=20)
        ball_vy = self.np_random.uniform(low=10, high=25)
        self.ball = Particle(self.environment, 0, self.environment.REF_W / 4, ball_vx, ball_vy, 0.5, self.environment.BALL_COLOR)
        self.delayScreen.reset()

    def step(self):
        self.betweenGameControl()
        self.agent_left.update()
        self.agent_right.update()

        if self.delayScreen.status():
            self.ball.applyAcceleration(0, self.environment.GRAVITY)
            self.ball.limitSpeed(0, self.environment.MAX_BALL_SPEED)
            self.ball.move()

        if self.ball.isColliding(self.agent_left):
            self.ball.bounce(self.agent_left)
        if self.ball.isColliding(self.agent_right):
            self.ball.bounce(self.agent_right)
        if self.ball.isColliding(self.fenceStub):
            self.ball.bounce(self.fenceStub)

        result = -self.ball.checkEdges()
        reward = 0
        reward += 0.5 / self.environment.t_limit

        if result != 0:
            self.newMatch()
            if result < 0:
                self.agent_left.emotion = "happy"
                self.agent_right.emotion = "sad"
                self.agent_right.life -= 1
                reward += +1
            else:
                self.agent_left.emotion = "sad"
                self.agent_right.emotion = "happy"
                self.agent_left.life -= 1
                reward += -1
        else:
            self.agent_left.updateState(self.ball, self.agent_right)
            self.agent_right.updateState(self.ball, self.agent_left)

        return reward

    def display(self, canvas):
        canvas = create_canvas(self.environment, canvas, self.environment.BACKGROUND_COLOR)
        self.fence.display(canvas)
        self.fenceStub.display(canvas)
        self.agent_left.display(canvas, self.ball.x, self.ball.y)
        self.agent_right.display(canvas, self.ball.x, self.ball.y)
        self.ball.display(canvas)
        self.ground.display(canvas)
        return canvas

    def betweenGameControl(self):
        if self.delayScreen.life > 0:
            pass