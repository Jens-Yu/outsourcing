from production import productions_conv_handler, list_productions_by_groupid, settle_conv_handler, \
    balance_conv_handler, view_logs_handler
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram import __version__ as TG_VER
from typing import Final
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ConversationHandler,
    filters, ContextTypes, CallbackContext
)
from openai import OpenAI
import logging
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime, timezone
from enum import Enum, auto
from setting import MONGO_URI, TOKEN, GPT_KEY
from mongodb import users_collection

class UserRole(Enum):
    STAFF = auto()
    MERCHANT = auto()

def role_to_str(role: UserRole) -> str:
    if role == UserRole.STAFF:
        return 'staff'
    elif role == UserRole.MERCHANT:
        return 'merchant'

current_time = datetime.now(timezone.utc)



# 日志配置
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

if GPT_KEY is None:
    raise ValueError("OpenAI API key not found. Set 'GPT_KEY' as an environment variable.")

client = OpenAI(api_key=GPT_KEY)

BOT_USERNAME: Final = '@YangHanBot'

ROLE_SELECTION = 0

async def welcome(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    user_id = user.id
    print(user_id)
    count = await users_collection.count_documents({'user_id': user_id})
    print(f"User ID: {user_id}, Count: {count}")  # 打印出 user_id 和匹配文档的数量

    if count == 0:
        keyboard = [['Staff', 'Merchant']]
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text(
            '欢迎使用本机器人，请问你是员工还是商户?',
            reply_markup=reply_markup
        )
        return ROLE_SELECTION
    else:
        await update.message.reply_text('您已注册.')
        return ConversationHandler.END

# Function to handle role selection
async def role_selection(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    text = update.message.text
    role = UserRole[text.upper()] if text.lower() in [role_to_str(r) for r in UserRole] else None

    if not role:
        await update.message.reply_text(
            '请选择您的角色。',
            reply_markup=ReplyKeyboardRemove()
        )
        return ROLE_SELECTION

    user_data = {
        'user_id': user.id,
        'username': user.username,
        'role': role_to_str(role),
        'registered_at': datetime.now(timezone.utc)
    }

    users_collection.insert_one(user_data)
    await update.message.reply_text(
        f'您已作为{role_to_str(role)}成功注册!',
        reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END

async def help_command(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('需求帮助请联系我。')

async def handle_message(update: Update, context: CallbackContext) -> None:
    text = update.message.text
    response = await ask_gpt(text)
    await update.message.reply_text(response)

async def ask_gpt(question: str, model="gpt-3.5-turbo") -> str:
    try:
        response = client.completions.create(
            model=model,
            prompt=question,
            max_tokens=100,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logger.error(f"Error with OpenAI request: {e}")
        return "抱歉，目前无法生成回复，请稍后再试。"

async def error(update: object, context: CallbackContext) -> None:
    logger.warning(f'Update {update} caused error {context.error}')


def main() -> None:
    application = Application.builder().token(TOKEN).build()

    # Define conversation handler for registration process
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', welcome)],
        states={
            ROLE_SELECTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, role_selection)],
        },
        fallbacks=[MessageHandler(filters.TEXT, welcome)]
    )

    list_productions_handler = CommandHandler('list', list_productions_by_groupid)

    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(list_productions_handler)
    application.add_error_handler(error)
    application.add_handler(productions_conv_handler)
    application.add_handler(settle_conv_handler)
    application.add_handler(balance_conv_handler)
    application.add_handler(view_logs_handler)
    application.run_polling()

if __name__ == '__main__':
    main()
