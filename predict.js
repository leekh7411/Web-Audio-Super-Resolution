const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const fs = require("fs");

async function predict(){
    model = await tf.loadModel('file://./asr-model/model.json');
}

predict();





