import pygame
import random
import time


class PygameUtils:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((480, 240))
        pygame.display.set_caption('Command Controller')

        self.commands = [0.0, 0.0, 0.0, 0.0]
        self.max_linear_speed = 0.75
        self.max_angular_speed = 1.0
        self.range_pos_z = (0.1531942, 0.3531942)
        self.acceleration = 0.05
        self.deceleration = 0.05
        self.automation_command = False
        self.automation_command_lateral = True
        self.automation_command_back_and_forth = True
        self.start_key_time = time.time()
        self.key_duration = 3
        self.next_key_time = self.start_key_time
        self.current_key = None

    def draw_keyboard(self, keys):
        self.screen.fill((0, 0, 0))

        key_positions = {
            pygame.K_UP: (75, 50, 50, 50),
            pygame.K_DOWN: (75, 150, 50, 50),
            pygame.K_LEFT: (25, 100, 50, 50),
            pygame.K_RIGHT: (125, 100, 50, 50),
            pygame.K_w: (275, 50, 50, 50),
            pygame.K_s: (275, 150, 50, 50),
        }

        for key, pos in key_positions.items():
            color = (255, 0, 0) if keys[key] else (0, 255, 0)
            pygame.draw.rect(self.screen, color, pos)

    def simulate_key_press(self):
        current_time = time.time()
        if current_time >= self.next_key_time:
            self.next_key_time = current_time + self.key_duration
            if self.automation_command_lateral and not self.automation_command_back_and_forth:
                self.current_key = random.choice([pygame.K_LEFT, pygame.K_RIGHT])
            elif self.automation_command_back_and_forth and not self.automation_command_lateral:
                self.current_key = random.choice([pygame.K_UP, pygame.K_DOWN])
            else:
                self.current_key = random.choice([pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT])

        keys = {pygame.K_UP: 0, pygame.K_DOWN: 0, pygame.K_LEFT: 0, pygame.K_RIGHT: 0}
        if self.current_key is not None:
            keys[self.current_key] = 1

        return keys

    def update_commands(self, automation=False):
        pygame.event.pump()

        if automation:
            keys = self.simulate_key_press()
        else:
            keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            self.commands[0] += self.acceleration
            self.commands[0] = min(self.commands[0], self.max_linear_speed)
        elif keys[pygame.K_DOWN]:
            self.commands[0] -= self.acceleration
            self.commands[0] = max(self.commands[0], -self.max_linear_speed)
        else:
            self.commands[0] = self._adjust_speed(self.commands[0], self.deceleration)

        if keys[pygame.K_LEFT]:
            self.commands[2] += self.acceleration
            self.commands[2] = min(self.commands[2], self.max_angular_speed)
        elif keys[pygame.K_RIGHT]:
            self.commands[2] -= self.acceleration
            self.commands[2] = max(self.commands[2], -self.max_angular_speed)
        else:
            self.commands[2] = self._adjust_speed(self.commands[2], self.deceleration)

        if keys[pygame.K_w]:
            self.commands[3] += self.acceleration * 0.01
            self.commands[3] = min(self.commands[3], self.range_pos_z[1])
        elif keys[pygame.K_s]:
            self.commands[3] -= self.acceleration * 0.01
            self.commands[3] = max(self.commands[3], self.range_pos_z[0])

        self.draw_keyboard(keys)

    def _adjust_speed(self, speed, change_rate):
        if speed > 0:
            speed -= change_rate
            speed = max(speed, 0)
        elif speed < 0:
            speed += change_rate
            speed = min(speed, 0)
        return speed

    def run_gui(self, mujoco_thread):
        try:
            clock = pygame.time.Clock()
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        mujoco_thread.join()
                        exit()
                pygame.display.flip()
                clock.tick(100)
        except KeyboardInterrupt:
            pygame.quit()
            mujoco_thread.join()

    def close_gui(self):
        pygame.quit()