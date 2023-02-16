// import INPUTS from './dataset/images.json';
// import OUTPUTS from './dataset/labels.json';

let INPUTS = await fetch('./dataset/images.json');
INPUTS = await INPUTS.json();

let OUTPUTS = await fetch('./dataset/labels.json');
OUTPUTS = await OUTPUTS.json();

const LABELS = [];

OUTPUTS.forEach((a) => {
  if(!LABELS.includes(a.masterCategory)) {
    LABELS.push(a.masterCategory);
  }
});

const a = await getImageDatas(INPUTS);

function getImageData(image, ctx) {
  return new Promise(async(resolve, reject) => {
    const width = 100;
    const height = 100;
    ctx.drawImage(image, 0, 0, width, height);

    const context = ctx.getImageData(0, 0, width, height);

    let imageData = null;

    await imageToGray(context.data).then((res) => {
      imageData = res;
    });

    resolve(imageData);
  })
}

function getImageDatas(inputs) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
  
    image.crossOrigin = 'Anonymous';
    const imageDatas = [];
  
    for(let x = 0; x < inputs.length; x++) {
      image.src = inputs[x];
      
      image.onload = getImageData(image, ctx).then((imageData) => {
        imageDatas.push(imageData);
      });
    }
  
    resolve(imageDatas);
  });
}

function imageToGray(context) {
  return new Promise((resolve, reject) => {
    const imageData = [];

    for (let x = 0; x < context.length; x+= 4) {
      const avg = (context[x] + context[x + 1] + context[x + 2]) / 3;

      imageData.push(avg);
    }

    resolve(imageData);
  });
}

// tf.util.shuffleCombo(INPUTS, OUTPUTS);

// function normalize(tensor, min, max) {
//   const result = tf.tidy(() => {
//     const MIN_VALUES = tf.scalar(min);
//     const MAX_VALUES = tf.scalar(max);

//     // en düşük değeri her değerden çıkarak hesapla ve yeni tensore ekle
//     const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);

//     // ARALIĞI bul
//     const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

//     const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

//     return NORMALIZED_VALUES;
//   });

//   return result;
// }

// const INPUTS_TENSOR = normalize(tf.tensor2d(INPUTS), 0, 255);
// const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, 'int32'), LABELS.length);


// export default { INPUTS, OUTPUTS, INPUTS_TENSOR, OUTPUTS_TENSOR, LABELS, normalize };
