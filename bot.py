import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from model import ModelHandler

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Инициализация модели
model = ModelHandler()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user = update.effective_user
    await update.message.reply_text(
        f"Привет, {user.first_name}! Я IT-помощник. Задайте мне вопрос о компьютерах, "
        "программах или технических проблемах, и я постараюсь помочь."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = """
Доступные команды:
/start - Начать общение
/help - Показать это сообщение

Просто напишите ваш вопрос, и я постараюсь помочь с техническими проблемами.
"""
    await update.message.reply_text(help_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    user_message = update.message.text
    
    # Показываем статус "печатает"
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, 
        action="typing"
    )
    
    try:
        # Получаем ответ от модели
        answer = model.get_answer(user_message)
        
        # Отправляем ответ пользователю
        await update.message.reply_text(answer)
    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {e}")
        await update.message.reply_text(
            "Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже."
        )

def main():
    """Запуск бота"""
    # Создаем Application и передаем токен бота
    application = Application.builder().token("8046791599:AAGGHIJO9OFsRLqw-r_GM_6GpubyZ6nHgSk").build()
    
    # Регистрируем обработчики команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    
    # Регистрируем обработчик текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запускаем бота
    application.run_polling()

if __name__ == '__main__':
    main()