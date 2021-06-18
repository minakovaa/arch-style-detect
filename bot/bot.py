import logging
import sys
from aiogram import Bot, Dispatcher, executor, types
import aiohttp
from PIL import Image
from io import BytesIO
import random
import os
import shutil


from classifier.classifier_prediction import arch_style_predict_by_image, load_checkpoint

API_TOKEN = sys.argv[1]  # 'BOT TOKEN HERE'

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

model_loaded, styles = load_checkpoint()

styles_description = {
    'барокко': "https://ru.wikipedia.org/wiki/%D0%90%D1%80%D1%85%D0%B8%D1%82%D0%B5%D0%BA%D1%82%D1%83%D1%80%D0%B0_%D0%B1%D0%B0%D1%80%D0%BE%D0%BA%D0%BA%D0%BE",
    'классицизм': "https://ru.wikipedia.org/wiki/%D0%9A%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%86%D0%B8%D0%B7%D0%BC#%D0%90%D1%80%D1%85%D0%B8%D1%82%D0%B5%D0%BA%D1%82%D1%83%D1%80%D0%B0",
    'русское_барокко': "https://ru.wikipedia.org/wiki/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%BE%D0%B5_%D0%B1%D0%B0%D1%80%D0%BE%D0%BA%D0%BA%D0%BE",
    'узорочье': "https://ru.wikipedia.org/wiki/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%BE%D0%B5_%D1%83%D0%B7%D0%BE%D1%80%D0%BE%D1%87%D1%8C%D0%B5",
    'готика': "https://ru.wikipedia.org/wiki/%D0%93%D0%BE%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B0%D1%8F_%D0%B0%D1%80%D1%85%D0%B8%D1%82%D0%B5%D0%BA%D1%82%D1%83%D1%80%D0%B0"}

choose_styles_keyboard = types.InlineKeyboardMarkup(resize_keyboard=True,
                                                    one_time_keyboard=True,
                                                    reply=False)

for style in styles:
    button_name = style.replace('_', ' ').capitalize()
    button_style = types.InlineKeyboardButton(button_name, callback_data=style)
    choose_styles_keyboard.add(button_style)


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.reply("Привет!"
                        "\nЭтот бот умеет определять архитектурный стиль здания. "
                        "Достаточно отправить фотографию."

                        "\n\nБот различает следующие архитектурные стили:\n" +
                        ', '.join([s.replace('_', ' ').capitalize() for s in styles]) + "."

                                                                                        "\n\n/styles - подробнее узнать об архитектурных стилях"

                                                                                        "\n\nПоддержать автора 🤗"
                                                                                        "\nhttps://archwalk.ru/donate"

                                                                                        "\n\nПриходите экскурсии и лекции об архитектуре Москвы "
                                                                                        "c Галиной Минаковой"
                                                                                        "\nhttps://archwalk.ru"
                        # "\n\nПро создание бота можно прочитать на сайте "
                        # "Галины Минаковой https://archwalk.ru/about_bot"
                        ,
                        disable_web_page_preview=True,
                        reply=False)
    """
    This handler will be called when user sends `/start` or `/help` command
    """


async def download_image(file_image: types.file):
    file_id = max(file_image.photo, key=lambda p: p.file_size).file_id

    link_to_json = f"https://api.telegram.org/bot{API_TOKEN}/getFile?file_id={file_id}"

    file_path = None
    img = None

    async with aiohttp.ClientSession() as session:
        async with session.get(url=link_to_json, ssl=False) as response:
            file_path = await response.json()
            file_path = file_path['result']['file_path']

        if file_path:
            link_to_image = f"https://api.telegram.org/file/bot{API_TOKEN}/{file_path}"

            async with session.get(url=link_to_image, ssl=False) as response:
                img_stream = await response.read()
                img = Image.open(BytesIO(img_stream))

    return img


def save_image(img, folder_name, img_name):
    total, used, free = shutil.disk_usage("/")

    # TODO If disk space is less then 500Mb, send me email
    # If free memory less then 50Mb
    if free // (2 ** 20) < 50:
        logging.critical(f'Space on the disk is less then {free // (2 ** 20)} Mb!')
        return

    path_folder = os.path.join(os.getcwd(), 'dataset_from_bot')
    if not os.path.exists(path_folder):
        os.mkdir(path=path_folder)

    path_folder = os.path.join(path_folder, folder_name)
    if not os.path.exists(path_folder):
        os.mkdir(path=path_folder)

    img.save(os.path.join(path_folder, img_name), 'JPEG')


@dp.message_handler(content_types=['photo'])
async def detect_style(file_image: types.file):
    # Get image from user
    img = await download_image(file_image)

    # Predict arch styles
    top_1_style, top_3_styles_with_probabilities = arch_style_predict_by_image(img,
                                                                               model=model_loaded,
                                                                               class_names=styles)

    # Save image after classify to class folder on server
    save_image(img,
               folder_name=top_1_style,
               img_name=file_image['from'].username + '_' +
                        file_image['date'].strftime('%Y_%m_%d-%H_%M_%S')+'.jpg'
               )

    top_1_style = top_1_style.replace('_', ' ').capitalize()

    result_str = "\n\nРаспределение вероятностей по топ-3 стилям:\n"
    for style, proba in top_3_styles_with_probabilities.items():
        result_str += f"{style.replace('_', ' ').capitalize()}: {proba:.03f}\n"
        # TODO probabilities must be 1 in sum. Thus you should 1 - {proba:.03f}

    await file_image.reply(f"Архитектурный стиль: {top_1_style}"
                           f"{result_str}"
                           "\n/styles - подробнее узнать об архитектурных стилях"

                           "\n\nПоддержать автора 🤗"
                           "\nhttps://archwalk.ru/donate",
                           reply=True)


@dp.message_handler(commands=['styles'])
async def select_style(message: types.Message):
    types.ReplyKeyboardMarkup(styles)

    await message.reply("Про какой архитектурный стиль рассказать подробнее?",
                        reply_markup=choose_styles_keyboard,
                        reply=False)


@dp.callback_query_handler(lambda call: call.data in styles)
async def get_style_description(callback_query: types.CallbackQuery):
    # Скрываем кнопки после выбора стиля и возвращаем описание
    await bot.edit_message_reply_markup(chat_id=callback_query.message.chat.id,
                                        message_id=callback_query.message.message_id)

    await callback_query.message.reply(styles_description[callback_query.data],
                                       disable_web_page_preview=False,
                                       reply=False)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
