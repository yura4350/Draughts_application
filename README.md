# Draughts (Checkers) Game - IB Computer Science HL Internal Assessment

This project is a fully-featured Draughts (Checkers) application developed as part of the IB Computer Science HL Internal Assessment. It includes a graphical user interface, local Player vs. Player (PvP) and Player vs. AI (PvE) modes, a database for storing game and user data, and a Telegram bot for notifications and interactions.

## Features

* **Graphical User Interface:** A clean and interactive game board built with Pygame.
* **Game Modes:**
    * **Player vs. Player (PvP):** Play against another person on the same computer.
    * **Player vs. AI (PvE):** Challenge an intelligent AI opponent.
* **Advanced AI:** The AI opponent uses the **Minimax algorithm with Alpha-Beta Pruning** for efficient and challenging gameplay.
* **Database Integration:**
    * Uses SQLite to store user information (`app_users.db`).
    * Saves and tracks game history and outcomes (`games.db`).
* **Telegram Bot:** A bot to interact with users, potentially for sending game notifications or stats (implementation in `TelegramBot.py`).
* **Game History:** A utility to load and review past games.

## Project Structure

```
.
├── checkers/             # Contains the core game logic and rules for Draughts.
├── minimax/              # Implementation of the Minimax algorithm with alpha-beta pruning for the AI.
├── TelegramBot.py        # Script for the Telegram bot integration.
├── app_users.db          # SQLite database for user data.
├── games.db              # SQLite database for storing game history.
├── load_game_history.py  # Utility script to load and display game history from the database.
├── main.py               # The main entry point to run the application.
├── pygame_part.py        # Handles the Pygame GUI, rendering the board, pieces, and interactions.
└── shared_state.py       # Manages shared state information across different modules.
```

## How to Run the Project

### Prerequisites

* Python 3.x
* Pygame library: `pip install pygame`
* python-telegram-bot library: `pip install python-telegram-bot`

### Running the Application

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    ```bash
    pip install pygame
    pip install python-telegram-bot
    ```

3.  **Run the main script:**
    ```bash
    python main.py
    ```
This will launch the Pygame window where you can start playing the game.

## Technologies Used

* **Language:** Python
* **GUI:** Pygame
* **AI Algorithm:** Minimax with Alpha-Beta Pruning
* **Database:** SQLite
* **Bot Framework:** python-telegram-bot
