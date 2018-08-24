const tf = require('@tensorflow/tfjs');
const tf_core = require('@tensorflow/tfjs-core');
require('@tensorflow/tfjs-node');

function print(x){
    console.log(x);
}
class subPixel1D extends tf.layers.Layer {
    
    constructor() {
        super({});
        // TODO(bileschi): Can we point to documentation on masking here?
        this.supportsMasking = true;
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1]*2 , inputShape[2] / 2];
    }

    call(inputs, kwargs) {
        let input = inputs;
        if (Array.isArray(input)) {
            input = input[0];
        }
        this.invokeCallHook(inputs, kwargs);
        //print('--input-shape--');
        //const input_shape = input.shape;
        //print(input_shape);
        const transpose_x = tf.transpose(input, [2,1,0]);
        //const tx_shape = transpose_x.shape;
        //print(tx_shape);
        const batchnd_x = tf.batchToSpaceND(transpose_x, [2], [[0,0]]);
        //const bx_shape = batchnd_x.shape;
        //print(bx_shape);
        const x = tf.transpose(batchnd_x , [2,1,0]);  
        return x;
    }

    getClassName() {
        return 'subPixel1D';
    }
}
subPixel1D.className = 'subPixel1D'; // static variable?
tf.serialization.SerializationMap.register(subPixel1D); // Here i added serialize code

async function load_model(){
    print('load-model');
    const model = await tf.loadModel('file://./asr-model/model.json');
    return model;
}  
load_model();





