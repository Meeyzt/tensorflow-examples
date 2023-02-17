const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const image = new Image();

document.body.append(canvas);

class Dataset {
  constructor() {
    this.trainData = [];
    this.testData = [];
  }

  /** Loads training and test data. */
  async loadData() {
    console.log('Loading images...');
    this.trainData = await getData('train');
    this.testData = await getData('test');
    console.log('Images loaded successfully.')
  }

  getTrainData() {
    return {
      
      images: tf.concat(this.trainData[0]),
      labels: tf.oneHot(tf.tensor1d(this.trainData[1], 'int32'), 5).toFloat() // here 5 is class
      
    }
  }

  getTestData() {
    return {
      images: tf.concat(this.testData[0]),
      labels: tf.oneHot(tf.tensor1d(this.testData[1], 'int32'), 5).toFloat()
    }
  }
}

const dataset = new Dataset();

async function getData(type) {
  const imageDatas = [];
  const redLabels = new Array(20).fill('Kirmizi');
  const siirtLabels = new Array(20).fill('Siirt');
  const labels = [...redLabels, ...siirtLabels];

  for(let x = 1; x <= 40; x++) {
    imageDatas.push(await getImageData(`./data/${type}/${x}.jpg`));
  }

  return [imageDatas, labels];
}

function getImageData(url) {
  return new Promise((resolve, reject) => {
    image.src = url;

    image.onload = () => {
      ctx.drawImage(image, 0, 0, 28, 28);
      resolve(ctx.getImageData(0, 0, 28, 28).data);
    }
  })
}

function normalize(tensor, min, max) {
  const result = tf.tidy(() => {
    const minValues = tf.scalar(min);
    const maxValues = tf.scalar(max);

    const tensorSubtractMinValue = tf.sub(tensor, minValues);

    const rangeSize = tf.sub(maxValues, minValues);

    const normalizedValues = tf.div(tensorSubtractMinValue, rangeSize);

    return normalizedValues;
  });

  return result;
}

function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.dense({ inputShape: [3136], units: 2, activation: 'relu' }));

  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));

  model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));

  return model;
}

async function train() {
  await dataset.loadData('train');

  const model = createModel();

  const  [images, labels]  = dataset.trainData;
  
  tf.util.shuffleCombo(images, labels);

  const inputTensor = normalize(tf.tensor2d(images), 0, 255);
  const outputTensor = tf.oneHot(tf.tensor1d(labels, 'int32'), 2);

  console.log(inputTensor)

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  let results = await model.fit(inputTensor, outputTensor, {
    shuffle: true,
    batchSize: 1000,
    eopchs: 250,
    callbacks: { onEpochEnd: logProcess }
  });

  inputTensor.dispose();
  outputTensor.dispose();

  // evaluate();
}

function logProcess(epoch, logs) {
  console.log('Data for epoch '+ epoch, logs);
}

async function evaluate() {

}

train();