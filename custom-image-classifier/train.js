import data from './data.js';
import model from './model.js';

const CANVAS = document.getElementById('canvas');
const CTX = CANVAS.getContext('2d');
const PREDICTION_ELEMENT = document.getElementById('prediction');

async function train() {
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  let results = await model.fit(data.INPUTS_TENSOR, data.OUTPUTS_TENSOR, {
    shuffle: true,
    batchSize: 512,
    epochs: 50,
    callbacks: { onEpochEnd: logProcess }
  });

  data.INPUTS_TENSOR.dispose();
  data.OUTPUTS_TENSOR.dispose();

  evaluate();
}

function logProcess(epoch, logs) {
  console.log('Data for epoch '+ epoch, logs);
}

async function evaluate() {
  // Random index alıyom
  const OFFSET = Math.floor((Math.random() * data.INPUTS.length));

  let answer = tf.tidy(() => {
    let newInput = data.normalize(tf.tensor1d(data.INPUTS[OFFSET]), 0, 255);

    let output = model.predict(newInput.expandDims());
    output.print();

    return output.squeeze().argMax();
  });

  answer.array().then((index) => {
    PREDICTION_ELEMENT.innerText = JSON.stringify(data.LABELS[index]);

    PREDICTION_ELEMENT.setAttribute('class', (index === data.OUTPUTS[OFFSET]) ? 'correct' : 'wrong');

    answer.dispose();
    drawImage(data.INPUTS[OFFSET]);
  });

  const saveResult = await model.save('localstorage://my-model-1');

  console.log('train bitti', saveResult)
}

function drawImage(digit) {
  var imageData = CTX.getImageData(0, 0, 28, 28);

  for (let x = 0; x < digit.length; x+= 4) {
    imageData.data[x] = digit[x]; // RED
    imageData.data[x + 1] = digit[x]; // GREEN
    imageData.data[x + 2] = digit[x]; // BLUE
    imageData.data[x + 3] = 255;
  }

  CTX.putImageData(imageData, 0, 0);
}

train();
console.log('loaded')

document.getElementById('fileUpload').addEventListener('change', (event) => {
  const file = event.target.files[0];
  const image = new Image();
  
  image.width = 100;
  image.height = 100;
  image.crossOrigin = 'Anonymous';
  
  image.onload = async() => {
    CTX.drawImage(image, 0, 0, 100, 100);

    const context = CTX.getImageData(0, 0, 100, 100);

    let answer = tf.tidy(() => {
      let newInput = data.normalize(tf.tensor1d(context.data), 0, 255);

      let output = model.predict(newInput.expandDims());
      output.print();
  
      return output.squeeze().argMax();
    });

    answer.array().then((index) => {
      PREDICTION_ELEMENT.innerText = JSON.stringify(data.LABELS[index]);
  
  
      answer.dispose();
      drawImage(data.INPUTS[index]);
    });


    // const pre = document.getElementById('pre');
    // imageData.forEach((d) => {
    //   buffer.push(d / 255);
    //   pre.innerHTML += `${d}, `;
    // });

    // const bufferDatas = await getImageAlphas(buffer);

    // decoder.classification(tensor);

    // decoder.predict(tensor).mul(255).cast('int32');

    
    // ctx.drawImage(imageData, 0, 0, 28, 28);

    // console.log('imageLoaded')
    // ctx.drawImage(image, 0 , 0, 28, 28);
    // console.log(ctx.getImageData(0, 0, 28, 28));
    // document.getElementById('pre').append(image);
  }

  image.src = URL.createObjectURL(file);
});
