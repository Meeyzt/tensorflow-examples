const tf = require('@tensorflow/tfjs-node');
const fs = require('node:fs');
const path = require('path');
const inputLabels = require('./dataset/labels.json');
const notExistingLabels = [];

const TRAIN_IMAGES_DIR = './dataset/train/images';
const TEST_IMAGES_DIR = './dataset/test/images';

async function loadImages(dataDir) {
  const images = [];
  const labels = [];
  
  var files = await fs.readdirSync(dataDir);

  for (let i = 0; i < files.length; i++) { 
    if (!files[i].toLocaleLowerCase().endsWith('.jpg')) {
      continue;
    }

    console.log(i, ' to ', files.length - 1);

    const fileName = Number(files[i].split('.')[0]);
    const filePath = path.join(dataDir, files[i]);
    
    const buffer = fs.readFileSync(filePath);
    const imageTensor = tf.node.decodeImage(buffer)
      .resizeNearestNeighbor([96,96])
      .toFloat()
      .div(tf.scalar(255.0))
      .expandDims();
    images.push(imageTensor);
  
    const label = inputLabels.find((label) => label.id === fileName);
    labels.push(label.articleType);
    if(!notExistingLabels.includes(label.articleType)) {
      notExistingLabels.push(label.articleType);
    }
  }
  console.log(dataDir, 'Labels are => ', labels.length);
  console.log('notExistingLabels => ', notExistingLabels.length);
  return [images, labels];
}

/** Helper class to handle loading training and test data. */
class Dataset {
  constructor() {
    this.trainData = [];
    this.testData = [];
  }

  /** Loads training and test data. */
  async loadData() {
    console.log('Loading images...');
    // this.trainData = await loadImages(TRAIN_IMAGES_DIR);
    this.testData = await loadImages(TEST_IMAGES_DIR);
    this.notExistingLabels = notExistingLabels;
    console.log('Images loaded successfully.')
  }

  getTrainData() {
    return {
      
      images: tf.concat(this.trainData[0]),
      labels: tf.oneHot(tf.tensor1d(this.trainData[1], 'int32'), 10).toFloat() // here 5 is class
      
    }
  }

  getTestData() {
    return {
      images: this.testData[0],
      labels: tf.oneHot(tf.tensor1d(this.testData[1], 'int32'), 10).toFloat()
    }
  }
}

module.exports = new Dataset();
console.log('All done.')