const data = require('./data');
const model = require('./model');

async function train() {
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['loss', 'accuracy']
  });

  let results = await model.fit(data.trainData[0], data.trainData[1], {
    shuffle: true,
    batchSize: 10000,
    epochs: 12,
    callbacks: { onEpochEnd: logProcess }
  });

  data.trainData.dispose();

  evaluate();
}

function logProcess(epoch, logs) {
  console.log('Data for epoch'+ epoch, logs);
}

async function evaluate() {
  const testImages = data.testData[0];
  const testLabels = data.testData[1];

  let answer = tf.tidy(() => {
    let output = model.predict(testImages[0]);
    output.print();

    return output.squeeze().argMax();
  });

  answer.array().then((index) => {
    console.log(testLabels[index], testLabels[0]);

    answer.dispose();
  });

  await model.save('./model');

  console.log('evaluate bitti');
}

train();