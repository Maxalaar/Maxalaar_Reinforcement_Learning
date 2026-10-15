import numpy as np
import pygame
import math

from maxalaar_reinforcement_learning.environments.pong_survivor.ball import Ball


def calculate_angle(x1, y1, x2, y2):
    return math.atan2(y2 - y1, x2 - x1)


def draw_arrow(surface, color, x1, y1, x2, y2, arrow_width):
    pygame.draw.line(surface, color, (x1, y1), (x2, y2))
    angle = calculate_angle(x1, y1, x2, y2)
    pygame.draw.polygon(surface, color, [
        (x2, y2),
        (x2 - arrow_width * math.cos(angle - math.pi / 6), y2 - arrow_width * math.sin(angle - math.pi / 6)),
        (x2 - arrow_width * math.cos(angle + math.pi / 6), y2 - arrow_width * math.sin(angle + math.pi / 6))
    ])


class RenderEnvironment:
    def __init__(self, environment):
        pygame.init()
        self.environment = environment

        # Window Parameters
        self.window_width = 800
        self.window_size_coefficient = self.window_width / environment.play_area[0]
        self.window_size = (
            int(environment.play_area[0] * self.window_size_coefficient),
            int(environment.play_area[1] * self.window_size_coefficient)
        )

        # Couleurs
        self.background_color = (255, 255, 255)
        self.ball_color = (0, 0, 0)
        self.paddle_color = (255, 0, 0)
        self.text_color = (0, 0, 0)
        self.arrow_color = (0, 0, 255)

        # Paramètres visuels
        self.ball_base_size = 2
        self.ball_size = int(self.ball_base_size * self.window_size_coefficient)
        self.paddle_height = 20
        self.font_size = 50
        self.arrow_head_ratio = 0.2

        # État du rendu
        self.human_render_is_init = False
        self.window = None
        self.clock = None
        self.canvas = pygame.Surface((self.window_size[0], self.window_size[0]))

    def render(self):
        self._render_frame()

        if self.environment.render_mode == 'human':
            if not self.human_render_is_init:
                if not pygame.get_init():
                    pygame.display.init()
                self.window = pygame.display.set_mode(self.window_size)
                self.clock = pygame.time.Clock()
                pygame.display.set_caption('Pong Survivor')
                self.human_render_is_init = True

            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(120)

        elif self.environment.render_mode == 'rgb_array':
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))

    def _render_frame(self):
        self.canvas.fill(self.background_color)

        for ball in self.environment.balls:
            self._draw_ball(ball)

        for paddle in self.environment.paddles:
            self._draw_paddle(paddle)

    def _draw_ball(self, ball: Ball):
        pygame.draw.rect(
            self.canvas,
            self.ball_color,
            pygame.Rect(
                ball.position * self.window_size_coefficient - self.ball_size / 2,
                (self.ball_size, self.ball_size),
            ),
        )

        if self.environment.display_number:
            font = pygame.font.Font(None, self.font_size)
            text = str(ball.matriculation)
            text_surface = font.render(text, True, self.text_color)
            text_rect = text_surface.get_rect()
            text_rect.center = ball.position * self.window_size_coefficient - self.ball_size
            self.canvas.blit(text_surface, text_rect)

        if self.environment.display_arrows:
            velocity_norm = np.linalg.norm(ball.velocity)
            velocity = ball.velocity / velocity_norm * self.environment.arrow_size
            draw_arrow(
                self.canvas,
                self.arrow_color,
                ball.position[0] * self.window_size_coefficient,
                ball.position[1] * self.window_size_coefficient,
                ball.position[0] * self.window_size_coefficient + velocity[0],
                ball.position[1] * self.window_size_coefficient + velocity[1],
                self.environment.arrow_size * self.arrow_head_ratio
            )

    def _draw_paddle(self, paddle):
        pygame.draw.rect(
            self.canvas,
            self.paddle_color,
            pygame.Rect(
                ((paddle.position[0] - paddle.size / 2) * self.window_size_coefficient,
                 (paddle.position[1] * self.window_size_coefficient) - self.paddle_height),
                (paddle.size * self.window_size_coefficient, self.paddle_height),
            ),
        )