import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import heapq
import math


class HexGrid:
    def __init__(self, rows=6, cols=10):
        self.rows = rows
        self.cols = cols
        self.hex_radius = 1  # Radius of a hexagon
        self.hex_height = np.sqrt(3) * self.hex_radius  # Height of a hexagon
        self.hex_width = 2 * self.hex_radius  # Width of a hexagon
        self.x_offset = self.hex_width * 3 / 4
        self.coordinates = {}
        self.obstacles = [(0, 2), (2, 3), (3, 2), (4, 3), (4, 1), (6, 1), (6, 2), (7, 1), (8, 4)]  # Obstacle positions
        self.traps = [(1, 4), (3, 4), (2, 1), (5, 2), (6, 4), (8, 3)]  # Trap positions
        self.treasures = [(4, 4), (3, 1), (7, 2), (9, 2)]  # Treasure positions
        self.rewards = [(4, 5), (1, 2), (7, 3), (5, 0)]  # Reward positions
        self.steps_taken = 0
        self.energy_consumed = 0
        self.step_cost = 4
        self.energy_cost = 4
        self.start_pos = (0, 5)
        self.create_hex_grid()

    def create_hex_grid(self):
        for row in range(self.rows):
            for col in range(self.cols):
                x = col * self.x_offset
                y = row * self.hex_height + (col % 2) * (self.hex_height / 2)
                self.coordinates[(col, row)] = (x, y)

    def display_grid(self, player_position=(0, 5)):
        fig, ax = plt.subplots()

        for (col, row), (x, y) in self.coordinates.items():
            hexagon = patches.RegularPolygon((x, y), numVertices=6, radius=self.hex_radius, orientation=np.radians(30),
                                             edgecolor='black', facecolor='white')
            ax.add_patch(hexagon)
            plt.text(x, y, f'{col},{row}', ha='center', va='center', fontsize=8)

        for obstacle in self.obstacles:
            ox, oy = self.coordinates[obstacle]
            obstacle_hex = patches.RegularPolygon((ox, oy), numVertices=6, radius=self.hex_radius,
                                                  orientation=np.radians(30), edgecolor='black', facecolor='darkgrey')
            ax.add_patch(obstacle_hex)

        for trap in self.traps:
            tx, ty = self.coordinates[trap]
            trap_hex = patches.RegularPolygon((tx, ty), numVertices=6, radius=self.hex_radius,
                                              orientation=np.radians(30), edgecolor='black', facecolor='violet')
            ax.add_patch(trap_hex)

        for treasure in self.treasures:
            tx, ty = self.coordinates[treasure]
            treasure_hex = patches.RegularPolygon((tx, ty), numVertices=6, radius=self.hex_radius,
                                                  orientation=np.radians(30), edgecolor='black', facecolor='yellow')
            ax.add_patch(treasure_hex)

        for reward in self.rewards:
            rx, ry = self.coordinates[reward]
            reward_hex = patches.RegularPolygon((rx, ry), numVertices=6, radius=self.hex_radius,
                                                orientation=np.radians(30), edgecolor='black', facecolor='turquoise')
            ax.add_patch(reward_hex)

        if player_position:
            px, py = self.coordinates[player_position]
            player = patches.RegularPolygon((px, py), numVertices=6, radius=self.hex_radius * 0.5,
                                            orientation=np.radians(30), edgecolor='black', facecolor='red')
            ax.add_patch(player)

        # Add text annotations for step cost, energy cost, steps taken, and energy consumed
        textstr = '\n'.join((
            f'Step Cost: {self.step_cost}',
            f'Energy Cost: {self.energy_cost}',
            f'Steps Taken: {self.steps_taken}',
            f'Energy Consumed: {self.energy_consumed}',
            f'Treasures Left: {len(self.treasures)}'))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(1.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        ax.set_xlim(-self.hex_radius, self.cols * self.x_offset + self.hex_radius)
        ax.set_ylim(-self.hex_radius, self.rows * self.hex_height + self.hex_radius)
        ax.set_aspect('equal')
        plt.axis('off')
        plt.show()

    def apply_trap_effect(self, new_position, last_direction, removal):
        if new_position == (8, 3):
            self.energy_cost *= 2
            if removal:
                self.traps.remove((8, 3))
            # print("Trap activated: Energy multiplier increased by 2.")
        elif new_position in [(1, 4), (2, 1)]:
            self.step_cost *= 2
            if removal:
                self.traps.remove(new_position)
            # print("Trap activated: Step multiplier increased by 2.")
        elif new_position in [(6, 4), (5, 2)]:
            col, row = new_position
            direction_effects = {
                'D': (0, -2),
                'U': (0, 2),
                'BL': (-2, -1),
                'TL': (-2, 1),
                'BR': (2, -1),
                'TR': (2, 1)
            }
            dc, dr = direction_effects[last_direction]
            col += dc
            row += dr

            # Check if the new position is valid
            if self.is_valid_position((col, row)):
                if removal:
                    self.traps.remove(new_position)
                new_position = (col, row)
                # print("Trap activated: Player moved two cells away.")
            else:
                # print("Invalid move after trap activation. Reverting to one cell move.")
                # Use single-cell movement direction effects
                col, row = new_position
                direction_effects = {
                    'D': (0, -1),
                    'U': (0, 1),
                    'BL': (-1, -1 if col % 2 == 0 else 0),
                    'TL': (-1, 0),
                    'BR': (1, -1),
                    'TR': (1, 1)
                }
                dc, dr = direction_effects[last_direction]
                col += dc
                row += dr
                if removal:
                    self.traps.remove(new_position)
                new_position = (col, row)

        elif new_position == (3, 4):
            self.treasures = []
            if removal:
                self.traps.remove((3, 4))
            #print("Trap activated: All uncollected treasures removed.")

        return new_position

    def apply_reward_effect(self, new_position, removal):
        if new_position in [(4, 5), (1, 2)]:
            self.energy_cost /= 2
            # print("Reward activated: Energy multiplier decreased by 2.")
        elif new_position in [(7, 3), (5, 0)]:
            self.step_cost /= 2
            # print("Reward activated: Step multiplier decreased by 2.")
        if removal:
            self.rewards.remove(new_position)

    def is_valid_position(self, position):
        col, row = position

        # Check if the position is within bounds
        if not (0 <= col < self.cols and 0 <= row < self.rows):
            return False

        # Check if the position is blocked by an obstacle
        if position in self.obstacles:
            return False

        return True

    def move_player(self, position, direction, removal):
        col, row = position
        last_direction = direction

        if direction == 'D':
            row -= 1
        elif direction == 'U':
            row += 1
        elif direction == 'BL':
            col -= 1
            if col % 2 == 1:
                row -= 1
        elif direction == 'TL':
            col -= 1
            if col % 2 == 0:
                row += 1
        elif direction == 'BR':
            col += 1
            if col % 2 == 1:
                row -= 1
        elif direction == 'TR':
            col += 1
            if col % 2 == 0:
                row += 1

        # Calculate path cost
        steps_cost = self.step_cost
        energy_cost = self.energy_cost * steps_cost

        # Ensure new position is within grid bounds and not an obstacle
        new_position = (col, row)
        if 0 <= col < self.cols and 0 <= row < self.rows:
            if new_position in self.obstacles:
                # print("Invalid move: Blocked by obstacle")
                return position
            else:
                self.steps_taken += steps_cost
                self.energy_consumed += energy_cost
                if new_position in self.traps:
                    new_position = self.apply_trap_effect(new_position, last_direction, removal)
                    # print("Player stepped on a trap!")
                    # print("step_multiplier", self.step_multiplier)
                    # print("energy_multiplier", self.energy_multiplier)
                if new_position in self.treasures:
                    # print("Player found a treasure!")
                    if removal:
                        self.treasures.remove(new_position)
                if new_position in self.rewards:
                    self.apply_reward_effect(new_position, removal)
                    # print("step_multiplier", self.step_multiplier)
                    # print("energy_multiplier", self.energy_multiplier)
                return new_position
        else:
            # print("Invalid move: Out of bounds")
            return position

    def get_neighbours(self, position):
        directions = ['D', 'U', 'BL', 'TL', 'BR', 'TR']
        valid_moves = []

        # Save the current state
        original_steps_taken = self.steps_taken
        original_energy_consumed = self.energy_consumed
        original_step_cost = self.step_cost
        original_energy_cost = self.energy_cost
        original_traps = self.traps.copy()
        original_treasures = self.treasures.copy()
        original_rewards = self.rewards.copy()

        for direction in directions:
            # Save the state before each move
            self.steps_taken = original_steps_taken
            self.energy_consumed = original_energy_consumed
            self.step_cost = original_step_cost
            self.energy_cost = original_energy_cost
            self.traps = original_traps.copy()
            self.treasures = original_treasures.copy()
            self.rewards = original_rewards.copy()

            new_position = self.move_player(position, direction, False)
            if new_position != position:  # Check if the move was valid
                tile_type = "normal"
                if new_position in original_traps:
                    tile_type = "trap"
                elif new_position in original_treasures:
                    tile_type = "treasure"
                elif new_position in original_rewards:
                    tile_type = "reward"

                collected_treasures = len(original_treasures) - len(self.treasures)

                valid_moves.append({
                    'direction': direction,
                    'position': new_position,
                    'tile_type': tile_type,
                    'energy_cost': self.energy_cost,
                    'step_cost': self.step_cost,
                    'collected_treasures': collected_treasures,
                    'energy_consumed': self.energy_consumed,
                    'steps_taken': self.steps_taken
                })

        # Restore the original state
        self.steps_taken = original_steps_taken
        self.energy_consumed = original_energy_consumed
        self.step_cost = original_step_cost
        self.energy_cost = original_energy_cost
        self.traps = original_traps
        self.treasures = original_treasures
        self.rewards = original_rewards

        return valid_moves

    def find_optimal_path_to_collect_treasures(self):
        def heuristic(pos, goals):
            # Using Euclidean distance as the heuristic
            return sum(math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2) for goal in goals)

        start = self.start_pos
        treasures = set(self.treasures)

        frontier = []
        heapq.heappush(frontier, (0, start, [], 0, 0, 4, 4, set(), self.traps.copy(), self.rewards.copy()))

        visited = set()

        while frontier:
            _, current, path, steps_taken, energy_consumed, step_cost, energy_cost, collected_treasures, current_traps, current_rewards = heapq.heappop(
                frontier)

            if collected_treasures == treasures:
                print(f"Found optimal path: {path}")
                print(f"Total Steps Taken: {steps_taken}")
                print(f"Total Energy Consumed: {energy_consumed}")
                return path, steps_taken, energy_consumed

            if (current, step_cost, energy_cost, tuple(current_traps), tuple(current_rewards)) in visited:
                continue

            visited.add((current, step_cost, energy_cost, tuple(current_traps), tuple(current_rewards)))

            for neighbour in self.get_neighbours(current):
                direction = neighbour['direction']
                position = neighbour['position']
                tile_type = neighbour['tile_type']
                next_steps_taken = steps_taken + step_cost
                next_energy_consumed = energy_consumed + energy_cost * step_cost
                next_step_cost = step_cost
                next_energy_cost = energy_cost

                new_collected_treasures = collected_treasures.copy()
                if position in self.treasures:
                    new_collected_treasures.add(position)

                # Create new state copies for traps and rewards
                new_traps = current_traps.copy()
                new_rewards = current_rewards.copy()

                # Apply the effect of the tile for the next move
                if tile_type == 'reward' and position in new_rewards:
                    if position in [(4, 5), (1, 2)]:
                        next_energy_cost /= 2
                    elif position in [(7, 3), (5, 0)]:
                        next_step_cost /= 2
                    new_rewards.remove(position)
                if tile_type == 'trap' and position in new_traps:
                    if position in [(1, 4), (2, 1)]:
                        next_step_cost *= 2
                    elif position in [(8, 3)]:
                        next_energy_cost *= 2
                    new_traps.remove(position)

                #print(f"Moving to {position} via {direction}")
                #print(f"Tile type: {tile_type}")
                #print(f"Steps Taken: {next_steps_taken}")
                #print(f"Energy Consumed: {next_energy_consumed}")
                #print(f"Step Cost: {next_step_cost}")
                #print(f"Energy Cost: {next_energy_cost}")
                #print(f"Collected Treasures: {new_collected_treasures}")

                new_path = path + [direction]
                priority = next_energy_consumed + heuristic(position, treasures - new_collected_treasures)

                heapq.heappush(frontier, (
                    priority, position, new_path, next_steps_taken, next_energy_consumed, next_step_cost,
                    next_energy_cost,
                    new_collected_treasures, new_traps, new_rewards))

        return None, None, None

# Initialize grid
grid = HexGrid()

# Find the optimal path to collect all treasures
path, steps_taken, energy_consumed = grid.find_optimal_path_to_collect_treasures()

"""# Display the result
if path:
    print(f"Optimal Path: {path}")
    print(f"Total Steps Taken: {steps_taken}")
    print(f"Total Energy Consumed: {energy_consumed}")
else:
    print("No path found to collect all treasures")"""

# Visualization
# Initialize starting position
player_position = grid.start_pos

# Display the initial grid with player
grid.display_grid(player_position)

for direction in path:
    player_position = grid.move_player(player_position, direction, True)
    grid.display_grid(player_position)
