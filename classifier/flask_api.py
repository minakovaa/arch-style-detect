import io

from PIL import Image
from flask import Flask, jsonify, request

from classifier_prediction import arch_style_predict_by_image, load_checkpoint, CLASS_REMAIN

model_loaded, styles = load_checkpoint(model_name='resnet50') #efficientnet-b5

app = Flask(__name__)


@app.route('/predict/', methods=['POST'])
def predict_image():
    img_bytes = request.data

    if not img_bytes:
        return None

    img = Image.open(io.BytesIO(img_bytes))

    prediction_top_3_styles_with_proba = arch_style_predict_by_image(img,
                                                                     model=model_loaded,
                                                                     class_names=styles,
                                                                     logger=None,
                                                                     samples_for_voting=1,
                                                                     batch_size_voting=1,
                                                                     is_debug=True)

    prediction_top_3_styles_with_proba = {class_name: f"{int(100 * float(proba))}" for class_name, proba
                                          in prediction_top_3_styles_with_proba.items()}

    return jsonify(prediction_top_3_styles_with_proba)


@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

