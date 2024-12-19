import os
import logging
from telegram import ForceReply, Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
 
# 몇 가지 명령 핸들러를 정의합니다. 일반적으로 업데이트와 컨텍스트의 두 인수를 받습니다.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start 명령이 실행되면 메시지를 보냅니다."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )
 
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """도움말 명령이 실행되면 메시지를 보냅니다."""
    await update.message.reply_text("Help!")
 
async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """사용자 메시지를 에코합니다."""
    await update.message.reply_text(update.message.text)
 
# 애플리케이션을 생성하고 봇의 토큰을 전달합니다.
app = ApplicationBuilder().token(os.environ.get("TELEGRAM_BOT_TOKEN")).build()
 
# 다른 명령에 대해 - 텔레그램으로 답변하기
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("help", help_command))
 
# 텔레그램에서 메시지를 에코합니다.
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
 
# 사용자가 Ctrl-C를 누를 때까지 봇을 실행합니다.
app.run_polling()