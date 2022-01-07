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

from classifier.classifier_prediction import arch_style_predict_by_image, load_checkpoint

FILEPATH_WITH_ARCHSTYLES_LINKS = "bot/archstyles_weblinks.txt"
LOGGER_FILE_CONFIG = "bot_logging.conf.yml"

logger = logging.getLogger("bot")

API_TOKEN = sys.argv[1]  # 'BOT TOKEN HERE'

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

styles_description = {}  # Fill once in func main()

model_loaded, styles = load_checkpoint(model_name='resnet18') #efficientnet-b5

choose_styles_keyboard = types.InlineKeyboardMarkup(resize_keyboard=True,
                                                    one_time_keyboard=True,
                                                    reply=False)

for style in styles:
    button_name = style.replace('_', ' ').capitalize()
    button_style = types.InlineKeyboardButton(button_name, callback_data=style)
    choose_styles_keyboard.add(button_style)


def setup_logging(logging_yaml_config_fpath):
    """setup logging via YAML if it is provided"""
    if logging_yaml_config_fpath:
        with open(logging_yaml_config_fpath) as config_fin:
            logging.config.dictConfig(yaml.safe_load(config_fin))


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    """
    This handler will be called when user sends `/start` or `/help` command
    """
    await message.reply("Привет!"
                        "\nЭтот бот умеет определять архитектурный стиль здания.\n" +
                        utils.markdown.bold("Достаточно отправить фотографию") + "."

                         f"\n\nРазличает {len(styles)} архитектурных стилей и возвращает"
                         f" распределение вероятностей по топ-3 наиболее подходящим стилям."
                        # + ",\n".join([s.replace('_', ' ').capitalize() for s in styles]) + "."

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

    logger.debug("Save image %s", img_name)


@dp.message_handler(content_types=['photo'])
async def detect_style(file_image: types.file):
    # Get image from user
    img = await download_image(file_image)

    # Predict arch styles
    top_3_styles_with_proba = arch_style_predict_by_image(img,
                                                          model=model_loaded,
                                                          class_names=styles,
                                                          logger=logger,
                                                          samples_for_voting=6,
                                                          batch_size_voting=1,
                                                          is_debug=True)

    # Delete 'Остальные' before fnd maximum probability of classes
    remain_class = {'Остальные': top_3_styles_with_proba.pop('Остальные')}

    top_1_style = max(top_3_styles_with_proba, key=lambda x: top_3_styles_with_proba[x])
    top_3_styles_with_proba.update(remain_class)

    # Save image after classify to class folder on server
    save_image(img,
               folder_name=top_1_style,
               img_name=file_image['from'].username + '_' +
                        file_image['date'].strftime('%Y_%m_%d-%H_%M_%S') + '.jpg'
               )

    top_1_style = top_1_style.replace('_', ' ').capitalize()

    result_str = "\n\nРаспределение вероятностей по топ-3 архитектурным стилям:\n"

    global styles_description
    for style, proba in top_3_styles_with_proba.items():
        if style in styles_description:
            result_str += f"[{style.replace('_', ' ').capitalize()}]({styles_description[style]}) ~ {proba:.03f}\n"
        else:
            result_str += f"{style.replace('_', ' ').capitalize()} ~ {proba:.03f}\n"

    await file_image.reply(f"{utils.markdown.bold(top_1_style)}"
                           f"{result_str}"
                           "\n/styles - список архитектурных стилей"

                           "\n\n[Поддержать автора 🤗](https://archwalk.ru/donate)"

                           "\n\n[Приходите на экскурсии и лекции об архитектуре Москвы "
                           "c Галиной Минаковой](https://archwalk.ru)"
                           ,
                           parse_mode=types.ParseMode.MARKDOWN,
                           disable_web_page_preview=True,
                           reply=True)


@dp.message_handler(commands=['styles'])
async def select_style(message: types.Message):
    types.ReplyKeyboardMarkup(styles)

    await message.reply("Про какой архитектурный стиль рассказать подробнее?",
                        reply_markup=choose_styles_keyboard,
                        reply=False)


@dp.callback_query_handler(lambda call: call.data in styles)
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

    with open(FILEPATH_WITH_ARCHSTYLES_LINKS, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            style_name, weblink = line.strip().split(',')
            styles_description[style_name] = weblink


def main():
    # setup_logging(LOGGER_FILE_CONFIG)
    read_links_with_styles_description_from_file()

    executor.start_polling(dp, skip_updates=True)


if __name__ == '__main__':
    main()
