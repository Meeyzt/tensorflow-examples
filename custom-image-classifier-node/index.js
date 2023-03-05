const data = require('./data.js');
const createModel = require('./model');
const tf = require('@tensorflow/tfjs-node');
const fs = require('node:fs');

async function train() {
  const [[inputs, outputTensor], testDataset, classes] = await data();

  const model = createModel(classes.length);

  const optimizer = tf.train.adam();

  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  let results = await model.fit(inputs, outputTensor, {
    batchSize: 3,
    epochs: 20,
    validationSplit: 0.2,
    callbacks: { onEpochEnd: logProcess }
  });

  // const model = await tf.loadLayersModel('file://./masterCategoryModel/model.json');
  evaluate(testDataset, classes, model);
}

function logProcess(epoch, logs) {
  // console.log('Data for epoch'+ epoch, logs);
}

async function evaluate(dataset, outputs, model) {
  let answer = tf.tidy(() => {
    let output = model.predict(tf.stack(dataset.images));
    // output.print();

    return output.squeeze();
  });

  answer.array().then((index) => {
    let accuracy = 0;

    index.forEach((x, i) => {
      const maxIndex = x.findIndex((a) => a === Math.max.apply(null, x));

      if(dataset.outputs[i] === outputs[maxIndex]) {
        accuracy += 1;
      }
    });

    console.log(accuracy / index.length -1);
    answer.dispose();
  });

  // await model.save('file://./model');

  console.log('evaluate bitti');
}

train();