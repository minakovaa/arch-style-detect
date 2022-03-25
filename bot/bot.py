import logging
import logging.config
import sys
from io import BytesIO
import os
import shutil

import yaml
from aiogram import Bot, Dispatcher, executor, types, utils
import aiohttp
from PIL import Image

from classifier.classifier_prediction import (
    predict_image_bytes, load_checkpoint, check_memory, CLASS_REMAIN,
)

model_loaded, styles = load_checkpoint(model_name='resnet18')

# Maximum size of received image. If greater then image should be downscaled
MAX_IMG_SIZE = 1024
STATUS_CODE_OK = 200

FILEPATH_WITH_ARCHSTYLES_LINKS = "bot/archstyles_weblinks.txt"
LOGGER_FILE_CONFIG = "logging.conf.yml"

API_TOKEN = sys.argv[1]  # Bot token
LINK_TO_CLF_API = sys.argv[2]  # Link to classifier api

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# Fill once in func main()
styles_description = {}

choose_styles_keyboard = types.InlineKeyboardMarkup(resize_keyboard=True,
                                                    one_time_keyboard=True,
                                                    reply=False)


def setup_logging(logging_yaml_config_fpath):
    """setup logging via YAML if it is provided"""
    global logger

    logger = logging.getLogger("bot")
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    if logging_yaml_config_fpath:
        with open(logging_yaml_config_fpath) as config_fin:
            logging.config.dictConfig(yaml.safe_load(config_fin))


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    """
    This handler will be called when user sends `/start` or `/help` command
    """
    await message.reply("Привет!"
                        "\nЭтот бот умеет определять архитектурный стиль здания."
                        "\nДостаточно отправить фотографию." +
                        f"\n\nРазличает {len(styles_description)} архитектурный стиль и возвращает"
                        f" распределение вероятностей по топ-3 наиболее подходящим стилям."
                        # + ",\n".join([s.replace('_', ' ').capitalize() for s in styles_description.keys()]) + "."

                        "\n\n/styles - список архитектурных стилей"

                        "\n\n[Поддержать автора 🤗](https://archwalk.ru/donate)"

                        "\n\n[Приходите на экскурсии и лекции об архитектуре Москвы "
                        "c Галиной Минаковой](https://archwalk.ru)"
                        # "\n\nПро создание бота можно прочитать на сайте "
                        # "Галины Минаковой https://archwalk.ru/about_bot"
                        ,
                        parse_mode=types.ParseMode.MARKDOWN,
                        disable_web_page_preview=True,
                        reply=False)


async def download_image(file_image: types.file):
    """
    Download image sent by user to bot
    """
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

    if max(img.size) > MAX_IMG_SIZE:
        img.thumbnail((MAX_IMG_SIZE, MAX_IMG_SIZE), Image.ANTIALIAS)

    bytes_io = BytesIO()
    img.save(bytes_io, format='JPEG')
    img_bytes = bytes_io.getvalue()
    return img_bytes


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

    if isinstance(img, bytes):
        img = Image.open(BytesIO(img))

    img.save(os.path.join(path_folder, img_name), 'JPEG')

    logger.debug("Save image '%s' size %sx%s predicted: '%s'", img_name, img.size[0], img.size[1], folder_name)


@dp.message_handler(content_types=['photo'])
async def detect_style(file_image: types.file):
    # Get image bytes from user
    img_bytes = await download_image(file_image)

    # # Request arch styles with Flask-api
    # top_3_styles_with_proba = None
    # async with aiohttp.ClientSession() as session:
    #     async with session.post(url=LINK_TO_CLF_API, data=img_bytes,
    #                             headers={'content-type': 'image/jpeg'}) as response:
    #         top_3_styles_with_proba = await response.json()

    top_3_styles_with_proba = predict_image_bytes(model_loaded, styles, img_bytes)  # Request arch styles directly from model

    if top_3_styles_with_proba is None:
        return await file_image.reply("Упс.. 🥲 Неполадки на сервере, попробуйте повторить позже.", reply=False)

    check_memory("recieve flask request")

    # Delete CLASS_REMAIN='Остальные' before find maximum probability of classes
    remain_class = {CLASS_REMAIN: top_3_styles_with_proba.pop(CLASS_REMAIN)}
    sorted_arch_styles = sorted(top_3_styles_with_proba, key=lambda x: int(top_3_styles_with_proba[x]), reverse=True)
    top_3_styles_with_proba.update(remain_class)

    top_1_style = sorted_arch_styles[0]

    # Save image after classify to class folder on server
    save_image(img_bytes, folder_name=top_1_style,
               img_name=file_image['from'].username + '_time_' +
                        file_image['date'].strftime('%Y_%m_%d-%H_%M_%S') + '.jpg'
               )

    result_str = "\n\nНаиболее вероятные архитектурные стили:\n"

    global styles_description

    for style in sorted_arch_styles + [CLASS_REMAIN]:
        proba = top_3_styles_with_proba[style]
        if style in styles_description:
            result_str += f"[{style.replace('_', ' ').capitalize()}]({styles_description[style]}) ~ {proba}%\n"
        else:
            result_str += f"{style.replace('_', ' ').capitalize()} ~ {proba}%\n"

    await file_image.reply(f"{utils.markdown.bold(top_1_style.replace('_', ' ').capitalize())}"
                           f"{result_str}"
                           "\n/styles - список архитектурных стилей"
                           "\n\n[Поддержать автора 🤗](https://archwalk.ru/donate)",
                           parse_mode=types.ParseMode.MARKDOWN,
                           disable_web_page_preview=True,
                           reply=True)


@dp.message_handler(commands=['styles'])
async def select_style(message: types.Message):
    types.ReplyKeyboardMarkup(styles_description.keys())

    await message.reply("Про какой архитектурный стиль рассказать подробнее?",
                        reply_markup=choose_styles_keyboard,
                        reply=False)


@dp.callback_query_handler(lambda call: call.data in styles_description.keys())
async def get_style_description(callback_query: types.CallbackQuery):
    """
    Скрываем кнопки после выбора стиля и возвращаем описание
    """
    global styles_description

    await bot.edit_message_reply_markup(chat_id=callback_query.message.chat.id,
                                        message_id=callback_query.message.message_id)

    await callback_query.message.reply(styles_description[callback_query.data],
                                       disable_web_page_preview=False,
                                       reply=False)


def read_links_with_styles_description_from_file():
    """
    Считываем из файла FILEPATH_WITH_ARCHSTYLES_LINKS список
    архитектурных стилей и ссылки на странички с их описанием.

    styles_description = {'архстиль':'https://...'}
    """
    global styles_description

    with open(FILEPATH_WITH_ARCHSTYLES_LINKS, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

        for line in lines:
            style_name, weblink = line.strip().split(',')
            styles_description[style_name] = weblink

    if len(styles_description.keys()) > 0:
        create_bottoms_with_styles(styles_description.keys())


def create_bottoms_with_styles(styles):
    """
    Fill  global 'choose_styles_keyboard'
    """
    # Create buttoms
    for style in styles:
        button_name = style.replace('_', ' ').capitalize()
        button_style = types.InlineKeyboardButton(button_name, callback_data=style)
        choose_styles_keyboard.add(button_style)


def main():
    setup_logging(LOGGER_FILE_CONFIG)
    read_links_with_styles_description_from_file()

    executor.start_polling(dp, skip_updates=True)


if __name__ == '__main__':
    main()
