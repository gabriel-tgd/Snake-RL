"""
snake_game.py — Environnement Snake pour Reinforcement Learning.

Logique du jeu + rendu pygame. Lancé seul (python snake_game.py), une IA
heuristique (pathfinding BFS) pilote le serpent en mode démo — utile pour
voir le jeu tourner sans dépendre d'un modèle entraîné.

Interface RL (inchangée) :
    env = SnakeGame(render=False)
    state = env.reset()
    state, reward, done = env.step(action)
"""

import pygame
import pygame.gfxdraw
import random
from collections import namedtuple, deque
from enum import Enum

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

BLOCK_SIZE = 30
GRID_WIDTH = 20
GRID_HEIGHT = 20
SPEED = 15
HEADER_HEIGHT = 55
BORDER = 10

# Palette sombre, dégradé violet/bleu
BG_BOARD = (22, 24, 34)
GRID_LINE = (32, 35, 48)
HEADER_BG = (16, 17, 26)
MAP_BORDER = (124, 92, 255)
TEXT_DARK = (240, 240, 245)
MUTED = (150, 150, 165)
RED = (255, 196, 0)
SNAKE_HEAD = (140, 110, 255)
SNAKE_BODY = (90, 200, 245)
ACCENT = (140, 110, 255)

Point = namedtuple("Point", ["x", "y"])


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


_CLOCKWISE = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]

_DELTA = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}

# ---------------------------------------------------------------------------
# Pathfinding pour le serpent de démo (IA heuristique, pas le modèle entraîné)
# ---------------------------------------------------------------------------

def _in_bounds(p):
    return 0 <= p.x < GRID_WIDTH and 0 <= p.y < GRID_HEIGHT


def _neighbors(p):
    for d, (dx, dy) in _DELTA.items():
        yield d, Point(p.x + dx, p.y + dy)


def bfs_first_step(start, target, blocked):
    """Plus court chemin (BFS). Retourne la Direction du PREMIER pas, ou None."""
    if start == target:
        return None
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == target:
            break
        for _, nxt in _neighbors(cur):
            if _in_bounds(nxt) and nxt not in blocked and nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    if target not in prev:
        return None
    cur = target
    while prev[cur] != start:
        cur = prev[cur]
    dx, dy = cur.x - start.x, cur.y - start.y
    for d, (ddx, ddy) in _DELTA.items():
        if (ddx, ddy) == (dx, dy):
            return d
    return None


def free_space(start, blocked, cap=120):
    """Flood fill : nombre de cases libres accessibles depuis start (borné)."""
    if not _in_bounds(start) or start in blocked:
        return 0
    seen = {start}
    q = deque([start])
    count = 0
    while q and count < cap:
        cur = q.popleft()
        count += 1
        for _, nxt in _neighbors(cur):
            if _in_bounds(nxt) and nxt not in blocked and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return count


def plan_direction(head, direction, food, blocked):
    """Cerveau de l'IA de démo :
       1) BFS vers la pomme (plus court chemin)
       2) sinon : direction sûre laissant le plus d'espace libre."""
    if food is None:
        return None
    d = bfs_first_step(head, food, blocked)
    if d is not None:
        return d
    best, best_space = None, -1
    candidates = [direction] + [x for x in _DELTA if x != direction]
    for cand in candidates:
        nxt = Point(head.x + _DELTA[cand][0], head.y + _DELTA[cand][1])
        if _in_bounds(nxt) and nxt not in blocked:
            sp = free_space(nxt, blocked)
            if sp > best_space:
                best, best_space = cand, sp
    return best


def _dir_to_action(current, desired):
    """Convertit une direction absolue en action relative (0/1/2)."""
    i = _CLOCKWISE.index(current)
    if _CLOCKWISE[(i + 1) % 4] == desired:
        return 1
    if _CLOCKWISE[(i - 1) % 4] == desired:
        return 2
    return 0


def _draw_food(display, food, board_x, board_y):
    fx = board_x + food.x * BLOCK_SIZE + BLOCK_SIZE // 2
    fy = board_y + food.y * BLOCK_SIZE + BLOCK_SIZE // 2
    r = BLOCK_SIZE // 2 - 3
    pygame.gfxdraw.aacircle(display, fx, fy, r, RED)
    pygame.gfxdraw.filled_circle(display, fx, fy, r, RED)


def _draw_snake(display, snake, board_x, board_y, body_color, head_color):
    radius = int(BLOCK_SIZE * 0.42)
    for i in range(len(snake) - 1, 0, -1):
        p_curr, p_prev = snake[i], snake[i - 1]
        c1 = (board_x + p_curr.x * BLOCK_SIZE + BLOCK_SIZE // 2,
              board_y + p_curr.y * BLOCK_SIZE + BLOCK_SIZE // 2)
        c2 = (board_x + p_prev.x * BLOCK_SIZE + BLOCK_SIZE // 2,
              board_y + p_prev.y * BLOCK_SIZE + BLOCK_SIZE // 2)
        x1 = min(c1[0], c2[0]) - radius
        y1 = min(c1[1], c2[1]) - radius
        x2 = max(c1[0], c2[0]) + radius
        y2 = max(c1[1], c2[1]) + radius
        pygame.draw.rect(display, body_color,
                         pygame.Rect(x1, y1, x2 - x1, y2 - y1),
                         border_radius=radius)
        pygame.draw.circle(display, body_color, c1, radius)
    head_center = (board_x + snake[0].x * BLOCK_SIZE + BLOCK_SIZE // 2,
                   board_y + snake[0].y * BLOCK_SIZE + BLOCK_SIZE // 2)
    pygame.draw.circle(display, head_color, head_center, radius)


# ---------------------------------------------------------------------------
# Environnement RL (API inchangée)
# ---------------------------------------------------------------------------

class SnakeGame:
    def __init__(self, render=True, trap_cap=None):
        self.trap_cap = trap_cap
        self.render_enabled = render
        self.high_score = 0
        if self.render_enabled:
            pygame.init()
            self.font_score = pygame.font.SysFont("segoeui", 22, bold=True)
            self.font_small = pygame.font.SysFont("segoeui", 15)
            self.window_width = GRID_WIDTH * BLOCK_SIZE + 2 * BORDER
            self.window_height = HEADER_HEIGHT + GRID_HEIGHT * BLOCK_SIZE + 2 * BORDER
            self.display = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
            pygame.display.set_caption("Snake RL")
            self.clock = pygame.time.Clock()
            self._bg = self._build_background()

        self.reset()

    def _build_background(self):
        w, h = GRID_WIDTH * BLOCK_SIZE, GRID_HEIGHT * BLOCK_SIZE
        surf = pygame.Surface((w, h))
        surf.fill(BG_BOARD)
        for x in range(0, w, BLOCK_SIZE):
            pygame.draw.line(surf, GRID_LINE, (x, 0), (x, h))
        for y in range(0, h, BLOCK_SIZE):
            pygame.draw.line(surf, GRID_LINE, (0, y), (w, y))
        return surf

    def reset(self):
        if hasattr(self, "score") and self.score > self.high_score:
            self.high_score = self.score

        self.direction = Direction.RIGHT
        self.head = Point(GRID_WIDTH // 2, GRID_HEIGHT // 2)
        self.snake = [
            self.head,
            Point(self.head.x - 1, self.head.y),
            Point(self.head.x - 2, self.head.y),
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        return self._get_state()

    def _place_food(self):
        while True:
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    def step(self, action, handle_events=True):
        """
        action : 0 = tout droit, 1 = droite, 2 = gauche
        handle_events=False -> ne consomme PAS les événements pygame
        """
        self.frame_iteration += 1

        if self.render_enabled and handle_events:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        old_head = self.head  # position avant le mouvement, pour le reward shaping
        old_distance = abs(old_head.x - self.food.x) + abs(old_head.y - self.food.y)

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        done = False
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            done = True
            reward = -10
            return self._get_state(), reward, done

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

            # Reward shaping : petit bonus/malus selon rapprochement de la
            # pomme (bien plus petit que +10/-10, pour ne jamais dominer la
            # décision face à un vrai événement comme manger/mourir)
            new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            if new_distance < old_distance:
                reward = 1
            elif new_distance > old_distance:
                reward = -1

        if self.render_enabled:
            self._render_frame()
            self.clock.tick(SPEED)

        return self._get_state(), reward, done

    def _is_collision(self, point=None):
        if point is None:
            point = self.head
        if point.x < 0 or point.x >= GRID_WIDTH or point.y < 0 or point.y >= GRID_HEIGHT:
            return True
        if point in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        idx = _CLOCKWISE.index(self.direction)
        if action == 0:
            new_direction = _CLOCKWISE[idx]
        elif action == 1:
            new_direction = _CLOCKWISE[(idx + 1) % 4]
        else:
            new_direction = _CLOCKWISE[(idx - 1) % 4]
        self.direction = new_direction

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1
        self.head = Point(x, y)

    # -----------------------------------------------------------------
    # State : 14 valeurs — danger à 1 case ET 2 cases (3 directions
    # relatives), direction actuelle, position relative de la pomme
    # -----------------------------------------------------------------
    def _get_state(self):
        """
        Retourne un tuple de 14 valeurs :
        - danger tout droit / à droite / à gauche, à 1 case ET à 2 cases
        - direction actuelle (one-hot)
        - position relative de la pomme (gauche/droite/haut/bas)
        """
        point_l = Point(self.head.x - 1, self.head.y)
        point_r = Point(self.head.x + 1, self.head.y)
        point_u = Point(self.head.x, self.head.y - 1)
        point_d = Point(self.head.x, self.head.y + 1)

        point_l2 = Point(self.head.x - 2, self.head.y)
        point_r2 = Point(self.head.x + 2, self.head.y)
        point_u2 = Point(self.head.x, self.head.y - 2)
        point_d2 = Point(self.head.x, self.head.y + 2)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger tout droit (1 case)
            (dir_r and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d)),

            # Danger à droite (1 case)
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_d)),

            # Danger à gauche (1 case)
            (dir_d and self._is_collision(point_r)) or
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d)),

            # Danger tout droit (2 cases)
            (dir_r and self._is_collision(point_r2)) or
            (dir_l and self._is_collision(point_l2)) or
            (dir_u and self._is_collision(point_u2)) or
            (dir_d and self._is_collision(point_d2)),

            # Danger à droite (2 cases)
            (dir_u and self._is_collision(point_r2)) or
            (dir_d and self._is_collision(point_l2)) or
            (dir_l and self._is_collision(point_u2)) or
            (dir_r and self._is_collision(point_d2)),

            # Danger à gauche (2 cases)
            (dir_d and self._is_collision(point_r2)) or
            (dir_u and self._is_collision(point_l2)) or
            (dir_r and self._is_collision(point_u2)) or
            (dir_l and self._is_collision(point_d2)),

            # Direction actuelle
            dir_l, dir_r, dir_u, dir_d,

            # Position de la pomme par rapport à la tête
            self.food.x < self.head.x,
            self.food.x > self.head.x,
            self.food.y < self.head.y,
            self.food.y > self.head.y,
        ]
        
        state = state + self._get_trap_flags()
        return [int(x) for x in state]
    
    def _get_trap_flags(self):
        idx = _CLOCKWISE.index(self.direction)
        tout_droit = _CLOCKWISE[(idx + 0) % 4]
        droite = _CLOCKWISE[(idx + 1) % 4]
        gauche = _CLOCKWISE[(idx - 1) % 4]
        blocked = set(self.snake[:-1])
        
        flag = []
        condidates = [tout_droit, droite, gauche]
        if self.trap_cap is None:
            cap = len(self.snake)
        else:
            cap = self.trap_cap
        for d in condidates:
            dx, dy = _DELTA[d]
            nxt = Point(self.head.x + dx, self.head.y + dy)
            space = free_space(nxt, blocked, cap)
            piege = space < len(self.snake)
            flag.append(piege)   
        return flag
            
            

    def _render_frame(self):
        board_x, board_y = BORDER, HEADER_HEIGHT + BORDER

        header = pygame.Rect(0, 0, self.window_width, HEADER_HEIGHT)
        pygame.draw.rect(self.display, HEADER_BG, header)
        pygame.draw.line(self.display, GRID_LINE,
                         (0, HEADER_HEIGHT), (self.window_width, HEADER_HEIGHT), 2)

        score_text = self.font_score.render(f"{self.score}", True, TEXT_DARK)
        self.display.blit(score_text, (16, 12))
        label = self.font_small.render("SCORE", True, MUTED)
        self.display.blit(label, (16 + score_text.get_width() + 8, 19))

        best_text = self.font_small.render(f"Meilleur : {self.high_score}", True, ACCENT)
        self.display.blit(best_text,
                          (self.window_width - best_text.get_width() - 16, 17))

        self.display.blit(self._bg, (board_x, board_y))
        _draw_food(self.display, self.food, board_x, board_y)
        _draw_snake(self.display, self.snake, board_x, board_y,
                    SNAKE_BODY, SNAKE_HEAD)

        map_rect = pygame.Rect(0, HEADER_HEIGHT, self.window_width,
                               GRID_HEIGHT * BLOCK_SIZE + 2 * BORDER)
        pygame.draw.rect(self.display, MAP_BORDER, map_rect, width=BORDER)

        pygame.display.flip()

    def render(self):
        if self.render_enabled:
            self._render_frame()


# ---------------------------------------------------------------------------
# Lancement direct : le serpent est piloté par l'IA heuristique (BFS),
# juste pour voir le jeu tourner sans dépendre d'un modèle entraîné.
# ---------------------------------------------------------------------------

def main():
    env = SnakeGame(render=True)
    running = True
    while running:
        blocked = set(env.snake[:-1])
        want = plan_direction(env.head, env.direction, env.food, blocked)
        action = _dir_to_action(env.direction, want) if want is not None else 0
        _, _, done = env.step(action)
        if done:
            env.reset()


if __name__ == "__main__":
    main()