const tf = require('@tensorflow/tfjs-node');
const fs = require('node:fs');
const path = require('path');
const labels = require('./dataset/labels.json');

const datasetPath = './dataset/images';

// input ve outputları array olarak ayarlama
async function loadImages(dataDir, batchSize = null, start = 0) {
  const images = [];
  const outputs = [];
  const classes = [];

  var files = await fs.readdirSync(dataDir);

  batchSize = batchSize ? batchSize + start: files.length - 1; 

  for (let i = start; i <= batchSize; i++) { 
    if (!files[i].toLocaleLowerCase().endsWith('.jpg')) {
      continue;
    }

    const imageId = Number(files[i].split('.')[0]);
    const existingLabel = labels.find((lbl) => lbl.id === imageId);
    if(!existingLabel) {
      continue;
    }

    if(!classes.includes(existingLabel.masterCategory)) {
      classes.push(existingLabel.masterCategory);
    }
    
    const imagePath = path.join(dataDir, files[i]);
    const imageBuffer = await fs.readFileSync(imagePath);
    const imageTensor = tf.node.decodeImage(imageBuffer).resizeNearestNeighbor([80, 60]).div(tf.scalar(255.0));

    if(imageTensor.shape[2] !== 3) {
      continue;
    }

    images.push(imageTensor);
    outputs.push(existingLabel.masterCategory)
  }
  return [{images, outputs}, classes];
}

async function run() {
  const [dataset, classes] = await loadImages(datasetPath, 2000);

  tf.util.shuffle(dataset.images);
  tf.util.shuffle(dataset.outputs);

  const datasetImageLength = dataset.images.length;
  const datasetOutputLength = dataset.outputs.length;

  const trainDataset = { images: dataset.images.slice(0, datasetImageLength * (3 / 4)), outputs: dataset.outputs.slice(0, datasetOutputLength * (3 / 4)) };
  const testDataset = { images: dataset.images.slice(datasetImageLength * (3 / 4), datasetImageLength), outputs: dataset.outputs.slice(datasetOutputLength * (3 / 4), datasetOutputLength) };

  //kategorilerine göre numaralamak için 
  const outputValues = trainDataset.outputs.map(o => classes.findIndex((c) => c === o));
  // kategori olduğu için oneHot kullandım
  const outputTensor = tf.tidy(() => tf.oneHot(tf.tensor1d(outputValues, 'int32'), classes.length));

  const inputTensor = tf.stack(trainDataset.images);
  
  return [[inputTensor, outputTensor], testDataset, classes];
}

module.exports = run;