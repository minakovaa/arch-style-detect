import io
import logging
import logging.config

from PIL import Image
from flask import Flask, jsonify, request

import yaml

from classifier_prediction import arch_style_predict_by_image, load_checkpoint, CLASS_REMAIN

model_loaded, styles = load_checkpoint(model_name='resnet50') #efficientnet-b5

app = Flask(__name__)

LOGGING_CONF_FILE = "logging.conf.yml"


@app.route('/predict/', methods=['POST'])
def predict_image():
    img_bytes = request.data

    if not img_bytes:
        return None

    img = Image.open(io.BytesIO(img_bytes))

    logger.info("Start predict image %sx%s class", img.size[0], img.size[1])
    prediction_top_3_styles_with_proba = arch_style_predict_by_image(img,
                                                                     model=model_loaded,
                                                                     class_names=styles,
                                                                     is_debug=True)
    logger.info("Finish predict image class")
    logger.info("Predictions: %s ", prediction_top_3_styles_with_proba)

    prediction_top_3_styles_with_proba = {class_name: f"{int(100 * float(proba))}" for class_name, proba
                                          in prediction_top_3_styles_with_proba.items()}

    return jsonify(prediction_top_3_styles_with_proba)


@app.route('/', methods=['GET'])
def index():
    return 'Arch style prediction'


def setup_logging(logging_yaml_config_fpath, logging_level=logging.INFO):
    """setup logging via YAML if it is provided"""
    global logger

    logger = logging.getLogger('flask_api')

    # Configure logging
    logging.basicConfig(level=logging_level)

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.INFO)

    if logging_yaml_config_fpath:
        with open(logging_yaml_config_fpath) as config_fin:
            logging.config.dictConfig(yaml.safe_load(config_fin))


def main():
    setup_logging(LOGGING_CONF_FILE, logging_level=logging.INFO)
    app.run(debug=False, host='0.0.0.0')


if __name__ == '__main__':
    main()


