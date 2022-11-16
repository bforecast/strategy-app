import datetime
import vectorbt as vbt

from utils.portfolio import Portfolio

from telegram.ext import CommandHandler

import toml
import logging

logging.basicConfig(level=logging.INFO)  # enable logging

class MyTelegramBot(vbt.TelegramBot):
    @property
    def custom_handlers(self):
        return (CommandHandler('get', self.get),
                CommandHandler('check', self.check),)

    @property
    def help_message(self):
         return "Type: \n  /get [strategy]_[symbol] to get the portfolio's performance. \
                    \n  /check [yyyy-mm-dd] to check the signals"

    def get(self, update, context):
        chat_id = update.effective_chat.id

        if len(context.args) == 1:
            strategyname = context.args[0].upper()
            portfolio = Portfolio()
            df = portfolio.get(strategyname)
            if len(df) == 0:
                self.send_message(chat_id, "No records")
            else:
                for i, row in df.iterrows():
                    result_str = '**' + strategyname + '**' + ':\n'
                    result_str += 'Annualized:      ' + "{0:.0%}".format(row['annual_return']) + '\n'
                    result_str += 'Lastday Return:  ' + "{0:.1%}".format(row['lastday_return']) + '\n'
                    result_str += 'Sharpe Ratio:    ' + "{0:.2}".format(row['sharpe_ratio']) + '\n'
                    result_str += 'Parameters:      ' + row['param_dict'] + '\n'
                    result_str += 'Update Date:     ' + row['end_date'] + '\n'
                    result_str += 'Description:     ' + row['description']

                    self.send_message(chat_id, result_str)
        else:
            self.send_message(chat_id, "This command requires strategy and symbol.")
            return

    def check(self, update, context):
        chat_id = update.effective_chat.id
        dt = datetime.date.today()
        if len(context.args) == 1:
            try:
                dt = datetime.datetime.strptime(context.args[0], '%Y-%m-%d').date()
            except ValueError as ve:
                print('ValueError Raised:', ve)
                self.send_message(chat_id, 'Date format error. yyyy-dd-mm is prefered.')

        portfolio = Portfolio()
        check_df = portfolio.check_records(dt=dt)
        if len(check_df) == 0:
            self.send_message(chat_id, "No records")
        else:
            self.send_message(chat_id, f"Found {len(check_df)} records on {dt}")

            for i, row in check_df.iterrows():
                symbol_str = str(i+1) + '.' + row['name'] + ' : ' + row['records']
                self.send_message(chat_id, symbol_str)


secrets_dict = toml.load(".streamlit/secrets.toml")
bot = MyTelegramBot(token=secrets_dict['telegram']['token'])
bot.start(in_background=True)

def update_portfolio():
    portfolio = Portfolio()
    print(datetime.datetime.now(), "--Update portfolio.")
    if portfolio.updateAll():
        print(datetime.datetime.now(), "--Update portfolio sucessfully.")
        bot.send_message_to_all("Update portfolios sucessfully.")

        check_df = portfolio.check_records(dt = datetime.date.today())
        # send the notification to telegram users
        if len(check_df) == 0:
            bot.send_message_to_all("No signal found.")
        else:
            bot.send_message_to_all(f"Found {len(check_df)} signal.")
            for i, row in check_df.iterrows():
                symbol_str =str(i+1) + '.' + row['name'] + ' : ' + row['records']
                bot.send_message_to_all(symbol_str)
    else:
        print(datetime.datetime.now(), "--Failed to update portfolio.")

def run_scheduler():
    # add jobs
    sch_manager = vbt.ScheduleManager()
    # Run in 2 minutes after Stock Market A Open
    sch_manager.every('09:32').do(update_portfolio)
    # Run in 2 minutes after Stock Market US Open
    sch_manager.every('22:32').do(update_portfolio) 
    # sch_manager.every().do(update_portfolio)
    sch_manager.start()


update_portfolio()
run_scheduler()







