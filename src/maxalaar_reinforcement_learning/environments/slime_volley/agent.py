import math

from maxalaar_reinforcement_learning.environments.slime_volley.relative_state import RelativeState
from maxalaar_reinforcement_learning.environments.slime_volley.slime_volley_utilities import toX, toY, toP, half_circle, circle


class Agent:
    def __init__(self, env, dir, x, y, c):
        self.env = env
        self.dir = dir
        self.x = x
        self.y = y
        self.r = 1.5
        self.c = c
        self.vx = 0
        self.vy = 0
        self.desired_vx = 0
        self.desired_vy = 0
        self.state = RelativeState()
        self.emotion = "happy"
        self.life = env.MAXLIVES

    def lives(self):
        return self.life

    def setAction(self, action):
        forward = action[0] > 0
        backward = action[1] > 0
        jump = action[2] > 0
        self.desired_vx = 0
        self.desired_vy = 0
        if forward and not backward:
            self.desired_vx = -self.env.PLAYER_SPEED_X
        if backward and not forward:
            self.desired_vx = self.env.PLAYER_SPEED_X
        if jump:
            self.desired_vy = self.env.PLAYER_SPEED_Y

    def move(self):
        self.x += self.vx * self.env.TIMESTEP
        self.y += self.vy * self.env.TIMESTEP

    def step(self):
        self.x += self.vx * self.env.TIMESTEP
        self.y += self.vy * self.env.TIMESTEP

    def update(self):
        self.vy += self.env.GRAVITY * self.env.TIMESTEP

        if self.y <= self.env.REF_U + self.env.NUDGE * self.env.TIMESTEP:
            self.vy = self.desired_vy

        self.vx = self.desired_vx * self.dir

        self.move()

        if self.y <= self.env.REF_U:
            self.y = self.env.REF_U
            self.vy = 0

        if self.x * self.dir <= (self.env.REF_WALL_WIDTH / 2 + self.r):
            self.vx = 0
            self.x = self.dir * (self.env.REF_WALL_WIDTH / 2 + self.r)

        if self.x * self.dir >= (self.env.REF_W / 2 - self.r):
            self.vx = 0
            self.x = self.dir * (self.env.REF_W / 2 - self.r)

    def updateState(self, ball, opponent):
        self.state.x = self.x * self.dir
        self.state.y = self.y
        self.state.vx = self.vx * self.dir
        self.state.vy = self.vy
        self.state.bx = ball.x * self.dir
        self.state.by = ball.y
        self.state.bvx = ball.vx * self.dir
        self.state.bvy = ball.vy
        self.state.ox = opponent.x * (-self.dir)
        self.state.oy = opponent.y
        self.state.ovx = opponent.vx * (-self.dir)
        self.state.ovy = opponent.vy

    def getObservation(self):
        return self.state.getObservation()

    def display(self, canvas, bx, by):
        x = toX(self.env, self.x)
        y = toY(self.env, self.y)
        r = toP(self.env, self.r)

        angle = math.pi * 60 / 180 if self.dir == -1 else math.pi * 120 / 180
        eyeX, eyeY = 0, 0

        half_circle(self.env, canvas, x, y, r, self.c)

        c = math.cos(angle)
        s = math.sin(angle)
        ballX = toX(self.env, bx) - (x + 0.6 * r * c)
        ballY = toY(self.env, by) - (y + 0.6 * r * s)

        dist = math.sqrt(ballX ** 2 + ballY ** 2)
        if dist > 0:
            eyeX = ballX / dist
            eyeY = ballY / dist

        eye_pos_x = x + 0.6 * r * c + eyeX * 0.15 * r
        eye_pos_y = y + 0.6 * r * s + eyeY * 0.15 * r
        circle(self.env, canvas, eye_pos_x, eye_pos_y, r * 0.3, (255, 255, 255))
        circle(self.env, canvas, eye_pos_x + eyeX * 0.15 * r, eye_pos_y + eyeY * 0.15 * r, r * 0.1, (0, 0, 0))

        for i in range(1, self.life):
            coin_x = toX(self.env, self.dir * (self.env.REF_W / 2 + 0.5 - i * 2.0))
            coin_y = toY(self.env, 1.5)
            circle(self.env, canvas, coin_x, self.env.WINDOW_HEIGHT - coin_y, toP(self.env, 0.5), self.env.COIN_COLOR)

        return canvas