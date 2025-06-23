import telebot
from telebot import types
import sqlite3
import hashlib

class TelegramBot:
    def __init__(self, token):
        self.bot = telebot.TeleBot(token)
        self.user_data = {}

        # Registering handlers
        self.bot.message_handler(commands=['start'])(self.start)
        self.bot.callback_query_handler(func=lambda call: True)(self.handle_query)
        self.bot.message_handler(commands=['game_history'])(self.show_game_history)
        self.bot.message_handler(func=lambda message: self.get_state(message) == 'awaiting_username')(
            self.process_username_step)
        self.bot.message_handler(func=lambda message: self.get_state(message) == 'awaiting_password')(
            self.process_password_step)

    def get_state(self, message):
        return self.user_data.get(message.chat.id, {}).get('state')

    def start(self, message):
        self.user_data[message.chat.id] = {'state': 'awaiting_username'}
        self.bot.send_message(message.chat.id, "Please enter your username:")

    def process_username_step(self, message):
        try:
            self.user_data[message.chat.id] = {
                'username': message.text,
                'state': 'awaiting_password'
            }
            self.bot.send_message(message.chat.id, "Now, please enter your password:")
        except Exception as e:
            self.bot.reply_to(message, 'Oops! Something went wrong. '
                                       'Probably the username you inputted has incorrect format '
                                       'or the username you inputted does not exist'
                                       'Please, try again.')

    def process_password_step(self, message):
        try:
            user_id = message.chat.id
            username = self.user_data[user_id]['username']
            password = message.text

            if self.check_credentials(username, password):
                self.user_data[user_id]['logged_in'] = True
                markup = self.create_game_history_button()
                self.bot.send_message(user_id, 'Login successful! Click the button below to view game history.',
                                      reply_markup=markup)
            else:
                self.bot.send_message(user_id,
                                      'Login failed. Please check your username and password '
                                      'and use /start to try again.')

            # Optionally, you can clear the user data here or after showing the history
            # del self.user_data[user_id]

        except Exception as e:
            self.bot.reply_to(message, 'An error occurred. Please try again using /start.')

    def create_game_history_button(self):
        markup = types.InlineKeyboardMarkup()
        button = types.InlineKeyboardButton("Game History", callback_data="game_history")
        markup.add(button)
        return markup

    def handle_query(self, call):
        user_id = call.message.chat.id
        if self.user_data.get(user_id, {}).get('logged_in', False):
            history = self.get_game_history(self.user_data[user_id]['username'])
            if history:
                self.bot.send_message(call.message.chat.id, history, parse_mode='HTML')
            else:
                self.bot.send_message(call.message.chat.id, "No game history found.")
            self.bot.send_message(call.message.chat.id, "Use /start to return to the main menu.")
            del self.user_data[user_id]  # Clear user data after showing history
        else:
            self.bot.send_message(call.message.chat.id, "You need to log in first. Use /start to begin.")

    def check_credentials(self, username, provided_password):
        conn = sqlite3.connect('app_users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT password FROM app_users WHERE username=?', (username,))
        result = cursor.fetchone()
        conn.close()

        if result:
            stored_password = result[0]
            return self.verify_password(stored_password, provided_password)
        else:
            return False

    def verify_password(self, stored_password, provided_password):
        salt = stored_password[:32]
        stored_pwdhash = stored_password[32:]
        pwdhash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
        return pwdhash == stored_pwdhash

    def show_game_history(self, message):
        try:
            history = self.get_game_history()
            if history:
                response = "Game History:\n" + history
            else:
                response = "No game history found."
            self.bot.send_message(message.chat.id, response)
        except Exception as e:
            self.bot.reply_to(message, 'Oops! Something went wrong while retrieving game history.')

    def get_game_history(self, username):
        conn = sqlite3.connect('games.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM games WHERE player1 = ? OR player2 = ?', (username, username))
        games = cursor.fetchall()
        conn.close()

        if not games:
            return None

        # Define column widths
        player_width = 15
        result_width = 8
        date_width = 16  # Adjust as needed for your date format

        # Header and separator using HTML and preformatted text for alignment
        history = "<pre>"
        history += f"{'Player 1':<{player_width}}{'Player 2':<{player_width}}{'Result':<{result_width}}Date\n"
        history += f"{'-' * player_width}{'-' * player_width}{'-' * result_width}{'-' * date_width}\n"

        # Format each game record
        for game in games:
            _, player1, player2, result, game_date = game
            history += f"{player1:<{player_width}}{player2:<{player_width}}{result:<{result_width}}{game_date}\n"

        history += "</pre>"
        return history

    def run(self):
        self.bot.polling(none_stop=True)

bot = TelegramBot("6860179471:AAGyV86i9zzD62E7i3HIr6lAAT8yjoXY5L8")
bot.run()