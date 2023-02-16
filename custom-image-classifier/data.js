let INPUTS = await fetch('./dataset/images.json');
INPUTS = await INPUTS.json();

let OUTPUTS = await fetch('./dataset/labels.json');
OUTPUTS = await OUTPUTS.json();

OUTPUTS =  OUTPUTS.slice(0, 1000);
INPUTS = INPUTS.slice(0, 1000);


const LABELS = [];

OUTPUTS.forEach((a) => {
  if(!LABELS.includes(a)) {
    LABELS.push(a.productDisplayName);
  }
});

const image = new Image();
image.crossOrigin = 'Anonymous';

INPUTS = await getImageDatas(INPUTS);

console.log(INPUTS)

function imageLoop(inputs, ctx, index = 0, datas = []) {
  return new Promise((resolve, reject) => {
    image.onload = async() => {
      const width = 100;
      const height = 100;

      ctx.drawImage(image, 0, 0, width, height);
      const context = ctx.getImageData(0, 0, width, height);

      datas.push(context.data);

      console.log(`${datas.length} ==> ${inputs.length}`);
  
      if(index <= inputs.length) {
        datas = await imageLoop(inputs, ctx, index + 1, datas);
        resolve(datas)
      }
    };

    if(index < inputs.length) { 
      image.src = inputs[index];
    } else {
      resolve(datas);
    }
  })
}

async function getImageDatas(inputs) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  document.body.append(canvas);


  const imageDatas = await imageLoop(inputs, ctx);

  return imageDatas;
}

function imageToGray(context) {
  return new Promise((resolve, reject) => {
    resolve(context.data)
  });
}

tf.util.shuffleCombo(INPUTS, OUTPUTS);

function normalize(tensor, min, max) {
  const result = tf.tidy(() => {
    const MIN_VALUES = tf.scalar(min);
    const MAX_VALUES = tf.scalar(max);

    // en düşük değeri her değerden çıkarak hesapla ve yeni tensore ekle
    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);

    // ARALIĞI bul
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

    return NORMALIZED_VALUES;
  });

  return result;
}

const INPUTS_TENSOR = normalize(tf.tensor2d(INPUTS), 0, 255);
const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, 'int32'), LABELS.length);


export default { INPUTS, OUTPUTS, INPUTS_TENSOR, OUTPUTS_TENSOR, LABELS, normalize };
