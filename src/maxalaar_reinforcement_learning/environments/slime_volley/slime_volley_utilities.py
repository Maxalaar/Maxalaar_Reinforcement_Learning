import math

import pygame


class DelayScreen:
    """ initially the ball is held still for INIT_DELAY_FRAMES(30) frames """

    def __init__(self, env, life=None):
        self.env = env
        self.life = life if life is not None else env.INIT_DELAY_FRAMES

    def reset(self, life=None):
        self.life = life if life is not None else self.env.INIT_DELAY_FRAMES

    def status(self):
        if self.life == 0:
            return True
        self.life -= 1
        return False

def toX(env, x):
    return (x + env.REF_W / 2) * env.FACTOR

def toP(env, x):
    return x * env.FACTOR

def toY(env, y):
    return y * env.FACTOR

def create_canvas(env, canvas, c):
    canvas.fill(c)
    return canvas

def rect(env, canvas, x, y, width, height, color):
    if height < 0:
        pygame_height = -height
        pygame_y = env.WINDOW_HEIGHT - y
    else:
        pygame_height = height
        pygame_y = env.WINDOW_HEIGHT - y - height
    pygame.draw.rect(canvas, color, (int(x), int(pygame_y), int(width), int(pygame_height)))

def half_circle(env, canvas, x, y, r, color):
    res = 20
    points = []
    for i in range(res + 1):
        ang = math.pi - math.pi * i / res
        px = x + math.cos(ang) * r
        py = y + math.sin(ang) * r
        points.append((int(px), int(env.WINDOW_HEIGHT - py)))
    pygame.draw.polygon(canvas, color, points)

def circle(env, canvas, x, y, r, color):
    pygame.draw.circle(canvas, color, (int(x), int(env.WINDOW_HEIGHT - y)), int(r))

class Particle:
    def __init__(self, env, x, y, vx, vy, r, c):
        self.env = env
        self.x = x
        self.y = y
        self.prev_x = self.x
        self.prev_y = self.y
        self.vx = vx
        self.vy = vy
        self.r = r
        self.c = c

    def display(self, canvas):
        circle(self.env, canvas, toX(self.env, self.x), toY(self.env, self.y), toP(self.env, self.r), self.c)
        return canvas

    def move(self):
        self.prev_x = self.x
        self.prev_y = self.y
        self.x += self.vx * self.env.TIMESTEP
        self.y += self.vy * self.env.TIMESTEP

    def applyAcceleration(self, ax, ay):
        self.vx += ax * self.env.TIMESTEP
        self.vy += ay * self.env.TIMESTEP

    def checkEdges(self):
        if self.x <= (self.r - self.env.REF_W / 2):
            self.vx *= -self.env.FRICTION
            self.x = self.r - self.env.REF_W / 2 + self.env.NUDGE * self.env.TIMESTEP

        if self.x >= (self.env.REF_W / 2 - self.r):
            self.vx *= -self.env.FRICTION
            self.x = self.env.REF_W / 2 - self.r - self.env.NUDGE * self.env.TIMESTEP

        if self.y <= (self.r + self.env.REF_U):
            self.vy *= -self.env.FRICTION
            self.y = self.r + self.env.REF_U + self.env.NUDGE * self.env.TIMESTEP
            if self.x <= 0:
                return -1
            else:
                return 1
        if self.y >= (self.env.REF_H - self.r):
            self.vy *= -self.env.FRICTION
            self.y = self.env.REF_H - self.r - self.env.NUDGE * self.env.TIMESTEP
        # fence:
        if (self.x <= (self.env.REF_WALL_WIDTH / 2 + self.r)) and (self.prev_x > (self.env.REF_WALL_WIDTH / 2 + self.r)) and (self.y <= self.env.REF_WALL_HEIGHT):
            self.vx *= -self.env.FRICTION
            self.x = self.env.REF_WALL_WIDTH / 2 + self.r + self.env.NUDGE * self.env.TIMESTEP

        if (self.x >= (-self.env.REF_WALL_WIDTH / 2 - self.r)) and (self.prev_x < (-self.env.REF_WALL_WIDTH / 2 - self.r)) and (self.y <= self.env.REF_WALL_HEIGHT):
            self.vx *= -self.env.FRICTION
            self.x = -self.env.REF_WALL_WIDTH / 2 - self.r - self.env.NUDGE * self.env.TIMESTEP
        return 0

    def getDist2(self, p):
        dy = p.y - self.y
        dx = p.x - self.x
        return dx * dx + dy * dy

    def isColliding(self, p):
        r = self.r + p.r
        return r * r > self.getDist2(p)

    def bounce(self, p):
        abx = self.x - p.x
        aby = self.y - p.y
        abd = math.sqrt(abx * abx + aby * aby)
        abx /= abd
        aby /= abd
        nx = abx
        ny = aby
        abx *= self.env.NUDGE
        aby *= self.env.NUDGE
        while self.isColliding(p):
            self.x += abx
            self.y += aby
        ux = self.vx - p.vx
        uy = self.vy - p.vy
        un = ux * nx + uy * ny
        unx = nx * (un * 2.)
        uny = ny * (un * 2.)
        ux -= unx
        uy -= uny
        self.vx = ux + p.vx
        self.vy = uy + p.vy

    def limitSpeed(self, minSpeed, maxSpeed):
        mag2 = self.vx * self.vx + self.vy * self.vy
        if mag2 > (maxSpeed * maxSpeed):
            mag = math.sqrt(mag2)
            self.vx /= mag
            self.vy /= mag
            self.vx *= maxSpeed
            self.vy *= maxSpeed
        if mag2 < (minSpeed * minSpeed):
            mag = math.sqrt(mag2)
            self.vx /= mag
            self.vy /= mag
            self.vx *= minSpeed
            self.vy *= minSpeed

class Wall:
    def __init__(self, env, x, y, w, h, c):
        self.env = env
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.c = c

    def display(self, canvas):
        rect(self.env, canvas, toX(self.env, self.x - self.w / 2), toY(self.env, self.y - self.h / 2), toP(self.env, self.w), toP(self.env, self.h), self.c)
        return canvas